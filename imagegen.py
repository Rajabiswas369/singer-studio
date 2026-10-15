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
    "Classic Romantic Painting": (
        "classical romantic oil painting in the style of the 19th-century masters "
        "(Bouguereau, Alma-Tadema, Leighton), a couple in period dress, soft candlelight or golden hour, "
        "rich warm palette, fine brushwork, fully clothed elegant attire, museum quality, "
        "no nudity, no text, no watermark"
    ),
    "Renaissance Fine Art (Stability AI)": (
        "Renaissance fine art oil painting, academic nude figure study, "
        "in the style of Titian, Raphael and Botticelli, classical mythology scene, "
        "soft natural light from a single window, warm golden tones, "
        "museum collection masterpiece, painterly brushwork, no text, no watermark"
    ),
    "Cinematic": "cinematic film still, dramatic lighting, highly detailed",
    "Anime": "anime style illustration, vibrant colours, detailed background",
    "Dreamy painting": "dreamy oil painting, soft glowing light, rich colours",
    "Neon cyberpunk": "neon cyberpunk city, glowing lights, rain, futuristic",
    "Nature": "breathtaking nature landscape, golden hour, photorealistic",
    "Impressionist": (
        "impressionist painting in the style of Monet or Renoir, soft dappled light, "
        "romantic garden or riverside setting, loose expressive brushstrokes, warm pastel colours, "
        "fully clothed elegant figures, no text, no watermark"
    ),
    "Baroque portrait": (
        "baroque oil portrait, dramatic chiaroscuro lighting, opulent 17th-century costumes, "
        "velvet and lace, rich jewel tones, classical architectural backdrop, "
        "museum quality, fully clothed, no nudity, no text, no watermark"
    ),
}

# Styles that require Stability AI (other providers will block these prompts)
_STABILITY_ONLY_STYLES = {"Renaissance Fine Art (Stability AI)"}


def get_key(provider: str) -> str:
    """API key from env var, else Streamlit secrets (top-level or [section]). Never logged."""
    name = ENV_KEYS.get(provider)
    if not name:
        return ""
    val = os.environ.get(name, "")
    if not val:
        try:
            import streamlit as st

            val = str(st.secrets.get(name, "") or "")
        except Exception:
            val = ""
    return val.strip().strip('"').strip("'")


def key_available(provider: str) -> bool:
    return provider not in ENV_KEYS or bool(get_key(provider))


def _api_error(r: requests.Response) -> str:
    """Short, secret-free description of a failed API response."""
    try:
        body = r.json()
        msg = "; ".join(body.get("errors", [])) or body.get("name") or body.get("error", {}).get("message", "")
    except Exception:
        msg = ""
    return f"HTTP {r.status_code}" + (f" - {str(msg)[:200]}" if msg else "")


# Cinematic variety angles for non-couple styles
_VARIETY = ["wide establishing shot", "close-up", "from above", "silhouette at sunset", "dramatic sky", "dreamlike abstract"]

# For painterly couple styles, each scene shows a different tender moment
_COUPLE_MOMENTS = [
    "a couple gazing into each other's eyes by candlelight",
    "lovers walking hand-in-hand through a flower garden",
    "a tender embrace on a moonlit balcony",
    "a man offering a rose to a woman in a grand hall",
    "a couple sitting together reading by a window",
    "two figures sharing a quiet moment by a river at dusk",
    "a couple dancing in a lantern-lit courtyard",
    "a woman waiting by a window while her beloved returns",
    "two silhouettes against a sunset sky",
    "a man serenading a woman on a balcony",
    "a couple sheltering together under a blossoming tree",
    "a tender farewell at a harbour",
]

_PAINTERLY_STYLES = {"Classic Romantic Painting", "Impressionist", "Baroque portrait"}


def build_prompts(title: str, lyrics: list[str], theme: str, style: str, mood: str, scenes: int) -> list[str]:
    """One prompt per scene. Lyrics are split evenly across scenes; with no lyrics, use title/theme."""
    scenes = max(1, scenes)
    style_desc = STYLES[style]
    base = f"{style_desc}, {mood}"

    if style in _PAINTERLY_STYLES:
        # For romantic painting styles use curated couple moments regardless of lyrics
        extra = f" inspired by: {theme}" if theme else ""
        extra += f" from the song '{title}'" if title else ""
        return [
            f"{_COUPLE_MOMENTS[i % len(_COUPLE_MOMENTS)]}{extra}. {base}"
            for i in range(scenes)
        ]

    base_with_guard = f"{base}, no text, no letters, no watermark"
    if lyrics:
        size = max(1, -(-len(lyrics) // scenes))  # ceil division
        groups = [" ".join(lyrics[i:i + size]) for i in range(0, len(lyrics), size)][:scenes]
        return [f"Scene illustrating the song lyric: {g[:200]}. {theme}. {base_with_guard}" for g in groups]
    subject = theme or title or "a singer performing on a stage"
    return [f"{subject}, {_VARIETY[i % len(_VARIETY)]}. {base_with_guard}" for i in range(scenes)]


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
        if not r.ok:
            raise RuntimeError(_api_error(r))
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
        headers={"Authorization": f"Bearer {get_key('OpenAI')}"},
        json={"model": "gpt-image-1", "prompt": prompt[:3000], "size": "1536x1024", "n": 1},
        timeout=TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(_api_error(r))
    return base64.b64decode(r.json()["data"][0]["b64_json"])


def _stability(prompt: str) -> bytes:
    r = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={"Authorization": f"Bearer {get_key('Stability AI')}", "Accept": "image/*"},
        files={"none": ""},
        data={"prompt": prompt[:9000], "aspect_ratio": "16:9", "output_format": "jpeg"},
        timeout=TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(_api_error(r))
    return r.content


def generate_scenes(
    prompts: list[str], provider: str, out_dir: Path, width: int, height: int, progress=None, errors: list | None = None
) -> list[Path]:
    """Generate one image per prompt. Failed scenes are skipped; reasons are appended to `errors` (no secrets)."""
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
            reason = str(exc)[:250] if isinstance(exc, RuntimeError) else type(exc).__name__
            log.error("scene %d failed with %s: %s", i, provider, reason)
            if errors is not None:
                errors.append(f"Scene {i + 1}: {reason}")
        if progress:
            progress((i + 1) / len(prompts))
    return paths
