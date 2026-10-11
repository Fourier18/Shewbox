"""
Core LLM driver (qwen3:8b via Ollama) with a generate_image tool.

The LLM never touches pixels. When it decides an image is needed, it calls
generate_image(), which sends a workflow to a local ComfyUI server (SDXL
running on your own GPU) and returns the saved file path.

Requires:
  - Ollama running locally with qwen3:8b pulled (already done)
  - ComfyUI running locally on http://127.0.0.1:8188 (started separately)
  - An SDXL checkpoint in ComfyUI/models/checkpoints
"""

import json
import random
import time
import urllib.request

import ollama

COMFY_URL = "http://127.0.0.1:8188"
CHECKPOINT_NAME = "sd_xl_base_1.0.safetensors"
OUTPUT_DIR = "generated_images"

MODEL = "qwen3:8b"


def _comfy_post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


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


def generate_image(prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024) -> str:
    """Send a prompt to the local ComfyUI/SDXL backend and return the saved image path."""
    seed = random.randint(0, 2**31 - 1)
    workflow = _build_workflow(prompt, negative_prompt, width, height, seed)
    result = _comfy_post("/prompt", {"prompt": workflow})
    prompt_id = result["prompt_id"]

    for _ in range(300):  # up to 5 minutes
        history = _comfy_get(f"/history/{prompt_id}")
        if prompt_id in history:
            outputs = history[prompt_id]["outputs"]
            for node_output in outputs.values():
                if "images" in node_output:
                    img = node_output["images"][0]
                    url = f"{COMFY_URL}/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
                    local_path = f"{OUTPUT_DIR}/{img['filename']}"
                    import os

                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    urllib.request.urlretrieve(url, local_path)
                    return local_path
        time.sleep(1)

    raise TimeoutError("ComfyUI did not finish generating the image in time.")


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
    }
]

AVAILABLE_FUNCTIONS = {"generate_image": generate_image}


def chat_loop():
    messages = []
    print(f"Talking to {MODEL}. Ask for an image and it will call the local SDXL tool. Ctrl+C to quit.\n")
    while True:
        user_input = input("you> ")
        messages.append({"role": "user", "content": user_input})

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
