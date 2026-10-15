"""Singer Studio - turn your vocal into a YouTube-ready AI music video."""
import logging
import tempfile
from pathlib import Path

import streamlit as st

import cloud
import imagegen
import studio

logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger("singer_studio")

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "aac", "mp4"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

st.set_page_config(page_title="Singer Studio", layout="centered")
st.title("Singer Studio")
st.caption("Record or upload your singing -> auto-tune + reverb -> AI-generated scenes -> video ready for YouTube.")

if not cloud.login_gate():
    st.stop()

if cloud.library_enabled():
    with st.expander("My saved videos"):
        saved = cloud.list_videos()
        if not saved:
            st.caption("Nothing saved yet. Videos you create are saved here automatically.")
        for item in saved:
            if st.button(f"Load {item['name']}", key=item["name"]):
                blob = cloud.download_video(item["name"])
                if blob:
                    st.video(blob)
                    st.download_button("Download", blob, file_name=item["name"], mime="video/mp4", key="dl_" + item["name"])

# ------------------------------------------------------------- inputs
st.subheader("1. Your voice")
recorded = st.audio_input("Record with your microphone")
uploaded = st.file_uploader("...or upload a vocal recording", type=AUDIO_TYPES)
backing_up = st.file_uploader(
    "Backing track / karaoke music (optional - must be royalty-free or yours)", type=AUDIO_TYPES
)

st.subheader("2. Sound")
c1, c2 = st.columns(2)
key_name = c1.selectbox("Song key", studio.NOTE_NAMES, index=0)
scale_name = c2.selectbox("Scale", list(studio.SCALES), index=0)
tune = st.slider("Auto-tune strength", 0.0, 1.0, 0.7, 0.05, help="0 = off, 1 = robotic snap")
reverb = st.slider("Reverb (room / echo)", 0.0, 1.0, 0.3, 0.05)
c3, c4 = st.columns(2)
vocal_gain = c3.slider("Vocal volume", 0.2, 2.0, 1.0, 0.1)
backing_gain = c4.slider("Backing volume", 0.0, 2.0, 0.6, 0.1)

st.subheader("3. AI visuals")
title = st.text_input("Song title (shown on video)", max_chars=80)
lyrics_text = st.text_area(
    "Lyrics (one line per row). AI draws a scene for each part of the song.", height=120
)
theme = st.text_input(
    "Story / theme (optional)", max_chars=200,
    placeholder="e.g. a lonely girl walking through a rainy city at night",
)
mood = st.text_input("Mood", value="emotional, atmospheric", max_chars=80)
style = st.selectbox("Art style", list(imagegen.STYLES))
provider = st.selectbox("AI image service", imagegen.PROVIDERS)
if not imagegen.key_available(provider):
    st.warning(
        f"{provider} needs {imagegen.ENV_KEYS[provider]} set (environment variable locally, "
        "or in Streamlit Cloud -> Settings -> Secrets). Pollinations needs no key."
    )
if style in imagegen._STABILITY_ONLY_STYLES and provider != "Stability AI":
    st.warning(
        "**Renaissance Fine Art** style requires **Stability AI** as the image service. "
        "Please change the 'AI image service' above to Stability AI."
    )
scenes_n = st.slider("Number of AI scenes", 2, 12, 6, help="More scenes = more variety, slower to generate")
bg_up = st.file_uploader("...or use your own background images instead of AI", type=IMAGE_TYPES, accept_multiple_files=True)
palette = st.selectbox("Visualizer colours", list(studio.PALETTES))
resolution = st.radio("Resolution", ["1280x720 (fast)", "1920x1080 (slower)"], horizontal=True)
quick = st.checkbox("Quick test - only process the first 20 seconds", value=True)

# ------------------------------------------------------------- run
if st.button("Create video", type="primary"):
    source = uploaded if uploaded is not None else recorded
    if source is None:
        st.warning("Record or upload your vocal first.")
        st.stop()
    if not bg_up and not imagegen.key_available(provider):
        st.error(f"Set {imagegen.ENV_KEYS[provider]} or choose Pollinations (free).")
        st.stop()

    width, height = (1920, 1080) if resolution.startswith("1920") else (1280, 720)
    max_len = float(studio.MAX_SECONDS) if cloud.is_local() else studio.CLOUD_MAX_SECONDS
    limit = 20.0 if quick else max_len
    lyrics = [ln.strip() for ln in lyrics_text.splitlines() if ln.strip()]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def save(upload, stem: str) -> Path:
                suffix = Path(upload.name).suffix.lower() if getattr(upload, "name", None) else ".wav"
                if suffix.lstrip(".") not in AUDIO_TYPES + IMAGE_TYPES:
                    suffix = ".wav"
                p = tmp / f"{stem}{suffix}"
                p.write_bytes(upload.getvalue())
                return p

            status = st.status("Working...", expanded=True)
            status.write("Reading audio")
            studio.convert_to_wav(save(source, "vocal_in"), tmp / "vocal.wav")
            vocal = studio.load_stereo(tmp / "vocal.wav", limit).mean(axis=0)

            backing = None
            if backing_up is not None:
                studio.convert_to_wav(save(backing_up, "backing_in"), tmp / "backing.wav")
                backing = studio.load_stereo(tmp / "backing.wav", limit)

            status.write("Auto-tuning (this is the slow part)")
            vocal = studio.autotune(
                vocal, studio.NOTE_NAMES.index(key_name), studio.SCALES[scale_name], tune
            )
            status.write("Adding reverb and mixing")
            vocal = studio.add_reverb(vocal, reverb)
            audio = studio.mix(vocal, backing, vocal_gain, backing_gain)

            if bg_up:
                scene_paths = [save(f, f"bg{i}") for i, f in enumerate(bg_up)]
            else:
                n_scenes = min(scenes_n, 3) if quick else scenes_n
                status.write(f"AI is drawing {n_scenes} scenes with {provider}")
                prompts = imagegen.build_prompts(title.strip(), lyrics, theme.strip(), style, mood.strip(), n_scenes)
                gen_bar = st.progress(0.0)
                gen_errors: list[str] = []
                scene_paths = imagegen.generate_scenes(
                    prompts, provider, tmp, width, height, gen_bar.progress, gen_errors
                )
                if gen_errors:
                    st.warning(f"{provider} problems:\n\n" + "\n".join(f"- {e}" for e in gen_errors[:5]))
                if not scene_paths:
                    st.warning("AI image service did not respond - using the animated gradient background instead.")

            status.write("Rendering video")
            bar = st.progress(0.0)
            out = tmp / "singer_studio.mp4"
            studio.render_video(
                audio, out, tmp, title=title.strip(), lyrics=lyrics,
                palette=palette, scene_paths=scene_paths, width=width, height=height,
                progress=bar.progress,
            )
            data = out.read_bytes()
            status.update(label="Done", state="complete")
        st.video(data)
        st.download_button("Download MP4", data, file_name="singer_studio.mp4", mime="video/mp4")
        if cloud.library_enabled():
            if len(data) > 50 * 1024 * 1024:
                st.warning("Video is over 50 MB, so it was not saved to your library. Download it above.")
            elif cloud.save_video(data, title):
                st.success("Saved to your library.")
            else:
                st.warning("Could not save to your library. Download it above.")
        if not cloud.is_local():
            st.caption(f"Hosted version processes up to {int(studio.CLOUD_MAX_SECONDS)} seconds of audio.")
        st.info("Untick 'Quick test' to process the whole song before uploading to YouTube.")
    except Exception:
        log.error("video creation failed", exc_info=True)
        st.error("Something went wrong while creating the video. Check the file formats and try again.")
