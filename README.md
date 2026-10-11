# Shewbox

A local desktop chat app for [Ollama](https://ollama.com) models, extended with local image generation. No cloud services, no API keys — everything runs on your own machine.

Chat with any model you have installed in Ollama. When a model supports tool calling and you ask for an image, it can call a `generate_image` tool that renders through a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) + Stable Diffusion backend and shows the result inline in the chat.

## Features

- Works with locally installed Ollama models, selectable in Settings
- Local image generation via ComfyUI/SDXL — no external API required
- Edit an attached image with the local image model
- Persistent chat history, saved locally and listed in the sidebar
- Attach text files as context or images to discuss or edit
- Save any generated image to disk from the chat window
- Seven themes, selectable in Settings and remembered across launches

## Themes

Open the gear icon in Shewbox to choose **Light**, **Dark**, **Forest Green**, **Azure Night**, **Desert Sunset**, **Arctic Dawn**, or **Plum Midnight**. Shewbox saves your choice in `settings.json` beside the executable (or beside the source files when running from Python) and restores it on the next launch. Light is the default.

## Requirements

- Windows
- [Ollama](https://ollama.com) installed, with at least one model pulled (e.g. `ollama pull qwen3:8b`)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed locally, with an SDXL checkpoint in `ComfyUI/models/checkpoints`
- Python 3.10+ (only needed if running from source rather than a built executable)

## Setup

1. Install Ollama and pull a model that supports tool calling.
2. Install ComfyUI and place an SDXL checkpoint in its `models/checkpoints` folder.
3. Install Python dependencies:
   ```
   pip install ollama pillow
   ```
4. Edit `ollama_image_agent.py` if your checkpoint filename differs from `sd_xl_base_1.0.safetensors`.
5. Run the app. Image generation starts the local ComfyUI engine automatically when needed. Keep the `ComfyUI` folder beside `Shewbox.exe` or in its parent folder (or set `SHEWBOX_COMFYUI_DIR` to its location).
   ```
   python qwen_image_app.py
   ```

## Building a standalone executable

```
pyinstaller --onefile --windowed --icon=shewbox.ico --add-data "shewbox.ico;." --name Shewbox qwen_image_app.py
```

The built executable includes the app's Python code and icon. It stores chat history, generated images, and settings beside itself. Ollama and ComfyUI remain local dependencies for chat and image generation; place the ComfyUI installation beside the executable or in its parent folder for automatic startup.

## How it works

The app itself only handles the UI and conversation state. When the active model decides to call `generate_image`, that function builds a ComfyUI workflow (checkpoint loader → CLIP text encode → KSampler → VAE decode → save image), submits it to the local ComfyUI server over HTTP, and returns the resulting file. Nothing leaves your machine.

## License

MIT
