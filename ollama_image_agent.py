"""
Core LLM driver (qwen3:8b via Ollama) with generate_image and edit_image tools.

The LLM never touches pixels directly. When it decides an image is needed, it
calls generate_image() (text-to-image) or edit_image() (image-to-image, for
a photo the user attached), which send a workflow to a local ComfyUI server
(SDXL running on your own GPU) and return the saved file path.

Requires:
  - Ollama running locally with qwen3:8b pulled (already done)
  - ComfyUI running locally on http://127.0.0.1:8188 (started separately)
  - An SDXL checkpoint in ComfyUI/models/checkpoints
"""

import json
import mimetypes
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

import ollama

COMFY_URL = "http://127.0.0.1:8188"
OLLAMA_URL = "http://127.0.0.1:11434"
CHECKPOINT_NAME = "sd_xl_base_1.0.safetensors"

if getattr(sys, "frozen", False):
    _DATA_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    _DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_DATA_DIR, "generated_images")

MODEL = "qwen3:8b"
_comfy_start_lock = threading.Lock()
_ollama_start_lock = threading.Lock()


def _ollama_ready():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=2) as response:
            return response.status == 200
    except (OSError, TimeoutError):
        return False


def ensure_ollama_running():
    """Start the local Ollama service when it isn't already running.

    Built to close a real gap: Shewbox auto-starts ComfyUI (see
    ensure_comfy_running below) but, until this was added, never did the
    same for Ollama itself -- it just assumed the service was already up
    and showed a connection error if it wasn't, even though starting it
    is as simple as running the same `ollama` binary the official app uses.
    """
    if _ollama_ready():
        return
    with _ollama_start_lock:
        if _ollama_ready():
            return
        exe = os.environ.get("SHEWBOX_OLLAMA_EXE") or shutil.which("ollama")
        if exe is None:
            raise RuntimeError("Ollama is not installed or not on PATH. Get it from https://ollama.com/download.")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [exe, "serve"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=flags,
            )
        except OSError as exc:
            raise RuntimeError(f"Ollama could not start: {exc}") from exc
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if _ollama_ready():
                return
            if process.poll() is not None:
                raise RuntimeError("Ollama closed during startup.")
            time.sleep(1)
        raise RuntimeError("Ollama did not become ready within 30 seconds.")


def _comfy_ready():
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2) as response:
            return response.status == 200
    except (OSError, TimeoutError):
        return False


def ensure_comfy_running():
    """Start the adjacent ComfyUI installation when image generation needs it."""
    if _comfy_ready():
        return
    with _comfy_start_lock:
        if _comfy_ready():
            return
        candidates = [
            os.environ.get("SHEWBOX_COMFYUI_DIR"),
            os.path.join(_DATA_DIR, "ComfyUI"),
            os.path.join(os.path.dirname(_DATA_DIR), "ComfyUI"),
        ]
        comfy_dir = next((p for p in candidates if p and os.path.isfile(os.path.join(p, "main.py"))
                          and os.path.isfile(os.path.join(p, "venv", "Scripts", "python.exe"))), None)
        if comfy_dir is None:
            raise RuntimeError("The local image engine is missing. Keep the ComfyUI folder beside Shewbox.exe or in its parent folder.")
        python = os.path.join(comfy_dir, "venv", "Scripts", "python.exe")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [python, "main.py", "--lowvram"], cwd=comfy_dir,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=flags,
            )
        except OSError as exc:
            raise RuntimeError(f"The local image engine could not start: {exc}") from exc
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if _comfy_ready():
                return
            if process.poll() is not None:
                raise RuntimeError("The local image engine closed during startup. Check the ComfyUI installation.")
            time.sleep(2)
        raise RuntimeError("The local image engine did not become ready within two minutes.")


def _comfy_post(path: str, payload: dict) -> dict:
    ensure_comfy_running()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _comfy_upload_image(local_path: str) -> str:
    """Upload a local photo to ComfyUI's own input store via its HTTP API and
    return the filename to reference from a LoadImage node. Going through the
    API (instead of guessing where ComfyUI's input/ folder lives on disk) is
    what makes this work whether this script is run directly or bundled into
    the Shewbox .exe, whose own folder is not where ComfyUI is installed."""
    ensure_comfy_running()
    boundary = uuid.uuid4().hex
    filename = os.path.basename(local_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(local_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{COMFY_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["name"]


def _comfy_get(path: str) -> dict:
    with urllib.request.urlopen(f"{COMFY_URL}{path}") as resp:
        return json.loads(resp.read())


def _build_workflow(prompt: str, negative_prompt: str, width: int, height: int, seed: int) -> dict:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7.0,
                "denoise": 1.0,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": seed,
                "steps": 25,
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT_NAME}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "agent_image", "images": ["8", 0]},
        },
    }


def _run_workflow(workflow: dict) -> str:
    """Submit a workflow to ComfyUI, wait for it to finish, and return the saved local path."""
    result = _comfy_post("/prompt", {"prompt": workflow})
    prompt_id = result["prompt_id"]

    deadline = time.monotonic() + 3600  # Low-VRAM renders on this machine can exceed 20 minutes.
    while time.monotonic() < deadline:
        history = _comfy_get(f"/history/{prompt_id}")
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("The local image engine failed while generating this image.")
            outputs = history[prompt_id]["outputs"]
            for node_output in outputs.values():
                if "images" in node_output:
                    img = node_output["images"][0]
                    url = f"{COMFY_URL}/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
                    local_path = f"{OUTPUT_DIR}/{img['filename']}"
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    urllib.request.urlretrieve(url, local_path)
                    return local_path
        time.sleep(2)

    raise TimeoutError("The local image engine did not finish within one hour.")


def generate_image(prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024) -> str:
    """Send a prompt to the local ComfyUI/SDXL backend and return the saved image path."""
    seed = random.randint(0, 2**31 - 1)
    workflow = _build_workflow(prompt, negative_prompt, width, height, seed)
    return _run_workflow(workflow)


def _build_img2img_workflow(image_filename: str, prompt: str, negative_prompt: str, denoise: float, seed: int) -> dict:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7.0,
                "denoise": denoise,
                "latent_image": ["11", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": seed,
                "steps": 25,
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT_NAME}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "agent_edit", "images": ["8", 0]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}},
    }


def edit_image(image_path: str, prompt: str, negative_prompt: str = "", denoise: float = 0.6) -> str:
    """Transform an existing photo through the local SDXL model using img2img and return the saved file path.

    image_path must be a path to a photo already attached in this conversation.
    denoise controls how much the source photo changes: close to 0 keeps it
    almost untouched, close to 1 treats it as little more than a rough sketch.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No such attached image: {image_path}")

    staged_name = _comfy_upload_image(image_path)

    seed = random.randint(0, 2**31 - 1)
    denoise = max(0.05, min(1.0, denoise))
    workflow = _build_img2img_workflow(staged_name, prompt, negative_prompt, denoise, seed)
    return _run_workflow(workflow)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt using the local SDXL model and return the saved file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of the image to generate."},
                    "negative_prompt": {"type": "string", "description": "Things to avoid in the image."},
                    "width": {"type": "integer", "description": "Image width in pixels, default 1024."},
                    "height": {"type": "integer", "description": "Image height in pixels, default 1024."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": (
                "Edit or restyle a photo the user attached to this conversation, using the local SDXL model "
                "(img2img). Use this when the user asks to change, edit, restyle, or transform an attached "
                "photo rather than create a brand new one from scratch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "File path of the attached photo to edit, exactly as given in the conversation.",
                    },
                    "prompt": {"type": "string", "description": "Description of the change to make to the photo."},
                    "negative_prompt": {"type": "string", "description": "Things to avoid in the result."},
                    "denoise": {
                        "type": "number",
                        "description": (
                            "How much to change the photo, from 0 to 1. Low (0.2-0.4) keeps it close to the "
                            "original, e.g. minor touch-ups. High (0.6-0.9) allows major changes. Default 0.6."
                        ),
                    },
                },
                "required": ["image_path", "prompt"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {"generate_image": generate_image, "edit_image": edit_image}


def chat_loop():
    messages = []
    print(f"Talking to {MODEL}. Ask for an image and it will call the local SDXL tool. Ctrl+C to quit.\n")
    while True:
        user_input = input("you> ")
        messages.append({"role": "user", "content": user_input})

        ensure_ollama_running()
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        messages.append(response["message"])

        if response["message"].get("tool_calls"):
            # 8GB VRAM can't hold qwen3:8b and SDXL at once. Unload qwen
            # immediately so ComfyUI has the full card while it renders;
            # ollama reloads qwen automatically on the next call below.
            ollama.chat(model=MODEL, messages=[], keep_alive=0)

            for call in response["message"]["tool_calls"]:
                fn_name = call["function"]["name"]
                fn_args = call["function"]["arguments"]
                print(f"[calling {fn_name}({fn_args})]")
                result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)
                messages.append({"role": "tool", "content": result, "name": fn_name})

            final = ollama.chat(model=MODEL, messages=messages)
            messages.append(final["message"])
            print(f"{MODEL}> {final['message']['content']}\n")
        else:
            print(f"{MODEL}> {response['message']['content']}\n")


if __name__ == "__main__":
    chat_loop()
