import asyncio
import re
from random import randint
from PIL import Image
import requests
from dotenv import get_key
import os
from time import sleep

# --- Config ---
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_KEY = get_key(".env", "HuggingFaceAPIKey")
if not HF_KEY:
    raise RuntimeError("HuggingFaceAPIKey not found in .env (use get_key('.env','HuggingFaceAPIKey'))")

headers = {
    "Authorization": f"Bearer {HF_KEY}",
    "Accept": "image/png"
}

DATA_DIR = "Data"
CONTROL_FILE = os.path.join("Frontend", "Files", "ImageGeneration.data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CONTROL_FILE), exist_ok=True)

# --- Helpers ---
def sanitize_filename(s: str) -> str:
    s = s.strip()
    # keep letters, numbers, dash, underscore; replace others with underscore
    return re.sub(r'[^A-Za-z0-9_\-]', '_', s) or "prompt"

def open_images(prompt_base: str):
    """Open any saved images that match the prompt base (works with png/jpg/etc)."""
    base = sanitize_filename(prompt_base.replace(" ", "_"))
    files_found = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith(base):
            files_found.append(os.path.join(DATA_DIR, fname))

    if not files_found:
        print(f"No files found for prompt base '{base}' in {DATA_DIR}")
        return

    for path in sorted(files_found):
        try:
            print(f"Opening image: {path}")
            img = Image.open(path)
            img.show()
            sleep(1)
        except Exception as exc:
            print(f"Unable to open {path}: {exc}")

# --- Networking ---
async def query(payload):
    # send blocking requests.post inside a thread so this async func doesn't block the event loop
    try:
        resp = await asyncio.to_thread(requests.post, API_URL, headers=headers, json=payload, timeout=60)
        return resp
    except Exception as exc:
        return exc  # caller will handle it

async def generate_image(prompt: str, n: int = 4):
    tasks = []
    for _ in range(n):
        payload = {
            "inputs": f"{prompt}, quality = 4K, sharpness=maximum, Ultra High details, high resolution",
            "options": {"wait_for_model": True},
            "parameters": {"seed": randint(0, 1000000)}
        }
        tasks.append(asyncio.create_task(query(payload)))

    responses = await asyncio.gather(*tasks)

    saved_files = []
    base = sanitize_filename(prompt.replace(" ", "_"))
    for i, resp in enumerate(responses, start=1):
        if isinstance(resp, Exception):
            print(f"Image {i} generation failed: request exception: {resp}")
            continue

        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and ctype.startswith("image/"):
            ext = ctype.split("/")[-1].split(";")[0]
            if ext == "jpeg":
                ext = "jpg"
            filename = os.path.join(DATA_DIR, f"{base}{i}.{ext}")
            try:
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"Saved {filename}")
                saved_files.append(filename)
            except Exception as exc:
                print(f"Failed to save {filename}: {exc}")
        else:
            # Print server JSON/text for debugging (invalid key, model error, etc.)
            text = resp.text if hasattr(resp, "text") else str(resp)
            print(f"Image {i} generation failed: status={getattr(resp,'status_code',None)} ctype={ctype} text={text}")

    return saved_files

def GenerateImage(prompt: str):
    saved = asyncio.run(generate_image(prompt))
    if saved:
        open_images(prompt)
    else:
        print("No images saved.")

# --- Control file helpers ---
def read_control_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
    except FileNotFoundError:
        return None, False
    except Exception as exc:
        print(f"Error reading control file: {exc}")
        return None, False

    if not data:
        return None, False

    # split on last comma so prompt can contain commas
    try:
        prompt, status = data.rsplit(",", 1)
    except ValueError:
        # malformed file
        return None, False

    return prompt.strip(), status.strip().lower() in ("true", "1", "yes")

def write_control_file(path: str, prompt: str = "", status: bool = False):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{prompt},{str(status)}")
    except Exception as exc:
        print(f"Error writing control file: {exc}")

# --- Main loop ---
if __name__ == "__main__":
    while True:
        Prompt, Status = read_control_file(CONTROL_FILE)
        if Prompt is None:
            sleep(1)
            continue

        if Status:
            print("Generating Image for prompt:", Prompt)
            try:
                GenerateImage(prompt=Prompt)
            except Exception as exc:
                print("Unexpected error during generation:", exc)
            # mark as done but keep the prompt (so you can see what was run)
            write_control_file(CONTROL_FILE, Prompt, False)
            break
        else:
            sleep(1)
