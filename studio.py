"""Audio + video processing for Singer Studio (auto-tune, reverb, mix, visualizer render)."""
import subprocess
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.signal import fftconvolve, get_window

SR = 44100
FPS = 25
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Minor": [0, 2, 3, 5, 7, 8, 10],
    "Chromatic (any note)": list(range(12)),
}

# name -> (background top, background bottom, bar colour A, bar colour B)
PALETTES = {
    "Neon Night": ((10, 5, 30), (40, 10, 70), (0, 220, 255), (255, 60, 200)),
    "Sunset": ((30, 8, 20), (110, 30, 40), (255, 190, 60), (255, 80, 90)),
    "Ocean": ((3, 12, 30), (8, 50, 80), (80, 240, 220), (80, 140, 255)),
    "Forest": ((5, 20, 12), (15, 55, 30), (190, 255, 100), (60, 220, 140)),
}

MAX_SECONDS = 600
CLOUD_MAX_SECONDS = 180  # free hosting has ~1 GB RAM and slow CPUs


def convert_to_wav(src: Path, dst: Path) -> None:
    """Decode any audio/video file to 44.1 kHz stereo WAV using the bundled ffmpeg."""
    subprocess.run(
        [FFMPEG, "-y", "-i", str(src), "-vn", "-ar", str(SR), "-ac", "2", str(dst)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
    )


def load_stereo(path: Path, max_seconds: float | None = None) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=False, duration=max_seconds)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y


def _snap(midi: float, key: int, scale: list[int]) -> float:
    base = int(round(midi))
    candidates = [n for n in range(base - 2, base + 3) if (n - key) % 12 in scale]
    return float(min(candidates, key=lambda n: abs(n - midi)))


def autotune(y: np.ndarray, key: int, scale: list[int], strength: float) -> np.ndarray:
    """Snap sung notes toward the chosen scale. y is mono. strength 0..1."""
    if strength <= 0:
        return y
    sr_f0 = 22050
    y_f0 = librosa.resample(y, orig_sr=SR, target_sr=sr_f0)
    f0, voiced, _ = librosa.pyin(
        y_f0, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sr_f0, frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr_f0, hop_length=512)

    n = 8192
    hop = n // 2
    win = get_window("hann", n, fftbins=True)
    yp = np.pad(y, (n, n))
    out = np.zeros_like(yp)
    for start in range(0, len(yp) - n + 1, hop):
        chunk = yp[start:start + n]
        t0 = (start - n) / SR
        mask = (times >= t0) & (times < t0 + n / SR) & voiced
        shift = 0.0
        if mask.sum() >= 3:
            midi = float(librosa.hz_to_midi(np.nanmedian(f0[mask])))
            shift = float(np.clip(strength * (_snap(midi, key, scale) - midi), -2.0, 2.0))
        if abs(shift) > 0.02:
            chunk = librosa.effects.pitch_shift(chunk, sr=SR, n_steps=shift)
        out[start:start + n] += chunk * win
    return out[n:n + len(y)]


def add_reverb(y: np.ndarray, amount: float, seconds: float = 1.8) -> np.ndarray:
    """Add a synthetic room reverb. Returns signal with a tail appended."""
    tail = int(seconds * SR)
    y = np.pad(y, (0, tail))
    if amount <= 0:
        return y
    rng = np.random.default_rng(0)
    t = np.arange(tail) / SR
    ir = rng.standard_normal(tail) * np.exp(-6.0 * t / seconds)
    wet = fftconvolve(y, ir)[: len(y)]
    wet *= (np.max(np.abs(y)) + 1e-9) / (np.max(np.abs(wet)) + 1e-9)
    return y * (1 - 0.5 * amount) + wet * amount


def mix(vocal: np.ndarray, backing: np.ndarray | None, vocal_gain: float, backing_gain: float) -> np.ndarray:
    """vocal mono (n,), backing stereo (2, m) or None. Returns stereo (2, n)."""
    n = len(vocal)
    out = np.stack([vocal, vocal]) * vocal_gain
    if backing is not None:
        reps = int(np.ceil(n / backing.shape[1]))
        out = out + np.tile(backing, (1, reps))[:, :n] * backing_gain
    return out / (np.max(np.abs(out)) + 1e-9) * 0.95


# ---------------------------------------------------------------- visuals

def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


ZOOM = 1.12  # slow push-in per scene (Ken Burns)
FADE_SECONDS = 1.0


def _gradient(width: int, height: int, palette: str) -> np.ndarray:
    top, bottom, _, _ = PALETTES[palette]
    g = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    col = np.array(top, np.float32) * (1 - g) + np.array(bottom, np.float32) * g
    return np.broadcast_to(col, (height, width, 3)).copy()


def _load_scenes(paths: list[Path], width: int, height: int) -> list[Image.Image]:
    """Pre-size scenes slightly larger than the frame so we can zoom without upscaling per frame."""
    big = (int(width * ZOOM), int(height * ZOOM))
    return [ImageOps.fit(Image.open(p).convert("RGB"), big) for p in paths]


def _scene_view(img: Image.Image, p: float, width: int, height: int) -> Image.Image:
    """Crop a centred window that shrinks to 1/ZOOM of the image as p goes 0..1, then resize to the frame."""
    z = 1.0 + (ZOOM - 1.0) * p
    cw, ch = img.width / z, img.height / z
    left, top = (img.width - cw) / 2, (img.height - ch) / 2
    return img.crop((left, top, left + cw, top + ch)).resize((width, height), Image.BILINEAR)


def _scene_frame(scenes: list[Image.Image], i: int, total: int, width: int, height: int) -> np.ndarray:
    n = len(scenes)
    pos = i / max(total, 1) * n
    k = min(int(pos), n - 1)
    local = pos - k
    view = _scene_view(scenes[k], local, width, height)
    fade = FADE_SECONDS / (max(total, 1) / FPS / n)  # fade length as a fraction of one scene
    if k < n - 1 and local > 1 - fade:
        nxt = _scene_view(scenes[k + 1], 0.0, width, height)
        view = Image.blend(view, nxt, (local - (1 - fade)) / fade)
    return np.asarray(view, dtype=np.float32) * 0.8


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, width: int, font) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    x = (width - (box[2] - box[0])) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))


def render_video(
    audio: np.ndarray, out_path: Path, work_dir: Path, *, title: str, lyrics: list[str],
    palette: str, scene_paths: list[Path], width: int = 1280, height: int = 720, progress=None,
) -> None:
    wav_path = work_dir / "final_mix.wav"
    sf.write(str(wav_path), audio.T, SR, subtype="PCM_16")

    mono = audio.mean(axis=0)
    hop = SR // FPS
    mel = librosa.feature.melspectrogram(
        y=mono, sr=SR, n_fft=2048, hop_length=hop, n_mels=64, fmin=40, fmax=12000
    )
    db = librosa.power_to_db(mel, ref=np.max)
    feats = np.clip((db + 60.0) / 60.0, 0, 1)  # (64, frames)

    total = int(len(mono) / SR * FPS)
    scenes = _load_scenes(scene_paths, width, height) if scene_paths else []
    base = None if scenes else _gradient(width, height, palette)
    _, _, ca, cb = PALETTES[palette]
    bars = 128
    t = np.abs(np.linspace(-1, 1, bars))  # mirrored colour ramp
    colours = [tuple(int(ca[k] * (1 - v) + cb[k] * v) for k in range(3)) for v in t]
    angles = np.linspace(0, 2 * np.pi, bars, endpoint=False) - np.pi / 2
    cos, sin = np.cos(angles), np.sin(angles)
    cx, cy = width // 2, height // 2
    title_font, lyric_font = _font(44), _font(40)

    cmd = [
        FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(FPS), "-i", "-", "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    smooth = np.zeros(64, np.float32)
    try:
        for i in range(total):
            cur = feats[:, min(i, feats.shape[1] - 1)]
            smooth = np.maximum(cur, smooth * 0.85)  # fast attack, slow decay
            bass = float(smooth[:6].mean())

            bg = _scene_frame(scenes, i, total, width, height) if scenes else base
            frame = np.clip(bg * (0.75 + 0.45 * bass), 0, 255).astype(np.uint8)
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img)
            r0 = 140 + 30 * bass
            vals = np.concatenate([smooth, smooth[::-1]])  # mirror 64 -> 128 bars
            for k in range(bars):
                ln = 12 + 110 * float(vals[k])
                d.line(
                    [(cx + cos[k] * r0, cy + sin[k] * r0), (cx + cos[k] * (r0 + ln), cy + sin[k] * (r0 + ln))],
                    fill=colours[k], width=6,
                )
            d.ellipse([cx - r0 + 8, cy - r0 + 8, cx + r0 - 8, cy + r0 - 8], outline=(255, 255, 255), width=3)
            if title:
                _draw_centered(d, title, 30, width, title_font)
            if lyrics:
                line = lyrics[min(int(i / max(total, 1) * len(lyrics)), len(lyrics) - 1)]
                _draw_centered(d, line, height - 90, width, lyric_font)
            proc.stdin.write(img.tobytes())
            if progress and i % 25 == 0:
                progress(i / total)
        proc.stdin.close()
        if proc.wait(timeout=300) != 0:
            raise RuntimeError("ffmpeg failed to encode the video")
    finally:
        if proc.poll() is None:
            proc.kill()
    if progress:
        progress(1.0)
