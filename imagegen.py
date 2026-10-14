"""AI scene generation: Pollinations (free default), OpenAI or Stability AI (API key from env vars)."""
import base64
import io
import logging
import os
import time
import urllib.parse
from pathlib import Path

import requests
from PIL import Image

log = logging.getLogger("singer_studio.imagegen")

TIMEOUT = 120  # seconds, no infinite waits
RETRY_STATUS = {402, 429, 502, 503}  # Pollinations' free tier rate-limits anonymous requests
RETRY_WAIT = 20  # seconds between retries
MAX_TRIES = 4
PROVIDERS = ["Pollinations (free)", "OpenAI", "Stability AI"]
ENV_KEYS = {"OpenAI": "OPENAI_API_KEY", "Stability AI": "STABILITY_API_KEY"}

STYLES = {
    "Cinematic": "cinematic film still, dramatic lighting, highly detailed",
    "Anime": "anime style illustration, vibrant colours, detailed background",
    "Dreamy painting": "dreamy oil painting, soft glowing light, rich colours",
    "Neon cyberpunk": "neon cyberpunk city, glowing lights, rain, futuristic",
    "Nature": "breathtaking nature landscape, golden hour, photorealistic",
}


def key_available(provider: str) -> bool:
    env = ENV_KEYS.get(provider)
    return env is None or bool(os.environ.get(env))


def build_prompts(title: str, lyrics: list[str], theme: str, style: str, mood: str, scenes: int) -> list[str]:
    """One prompt per scene. Lyrics are split evenly across scenes; with no lyrics, use title/theme."""
    scenes = max(1, scenes)
    base = f"{STYLES[style]}, {mood}, no text, no letters, no watermark"
    if lyrics:
        size = max(1, -(-len(lyrics) // scenes))  # ceil division
        groups = [" ".join(lyrics[i:i + size]) for i in range(0, len(lyrics), size)][:scenes]
        return [f"Scene illustrating the song lyric: {g[:200]}. {theme}. {base}" for g in groups]
    subject = theme or title or "a singer performing on a stage"
    variety = ["wide establishing shot", "close-up", "from above", "silhouette at sunset", "dramatic sky", "dreamlike abstract"]
    return [f"{subject}, {variety[i % len(variety)]}. {base}" for i in range(scenes)]


def _pollinations(prompt: str, width: int, height: int, seed: int) -> bytes:
    url = (
        "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:600])
        + f"?width={width}&height={height}&nologo=true&seed={seed}"
    )
    for attempt in range(MAX_TRIES):
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code in RETRY_STATUS and attempt < MAX_TRIES - 1:
            time.sleep(RETRY_WAIT)
            continue
        r.raise_for_status()
        return _strip_watermark(r.content)
    raise RuntimeError("unreachable")


def _strip_watermark(data: bytes, fraction: float = 0.09) -> bytes:
    """The free tier stamps a logo in the bottom-right corner; crop that strip off (scene is re-fitted later)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.crop((0, 0, img.width, int(img.height * (1 - fraction))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _openai(prompt: str) -> bytes:
    r = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": "gpt-image-1", "prompt": prompt[:3000], "size": "1536x1024", "n": 1},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return base64.b64decode(r.json()["data"][0]["b64_json"])


def _stability(prompt: str) -> bytes:
    r = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={"Authorization": f"Bearer {os.environ['STABILITY_API_KEY']}", "Accept": "image/*"},
        files={"none": ""},
        data={"prompt": prompt[:9000], "aspect_ratio": "16:9", "output_format": "jpeg"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.content


def generate_scenes(prompts: list[str], provider: str, out_dir: Path, width: int, height: int, progress=None) -> list[Path]:
    """Generate one image per prompt. Failed scenes are skipped (logged without secrets)."""
    paths: list[Path] = []
    for i, prompt in enumerate(prompts):
        try:
            if provider == "OpenAI":
                data = _openai(prompt)
            elif provider == "Stability AI":
                data = _stability(prompt)
            else:
                data = _pollinations(prompt, width, height, seed=1000 + i)
            p = out_dir / f"scene_{i}.jpg"
            p.write_bytes(data)
            paths.append(p)
        except Exception as exc:  # network/API errors must not stop the whole video
            log.error("scene %d failed with %s: %s", i, provider, type(exc).__name__)
        if progress:
            progress((i + 1) / len(prompts))
    return paths
