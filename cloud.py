"""Supabase login (Auth) + private video library (Storage) for the hosted version of Singer Studio.

Secrets (Streamlit Cloud -> Settings -> Secrets):  supabase_url, supabase_key  (the *anon* key).
Local runs set SINGER_LOCAL=1 (done by START_SINGER_STUDIO.bat) and skip login because the server
only listens on 127.0.0.1. Anywhere else the app FAILS CLOSED: no Supabase config -> no access.
"""
import logging
import os
import re
import time

import streamlit as st

log = logging.getLogger("singer_studio.cloud")
BUCKET = "videos"


def is_local() -> bool:
    return os.environ.get("SINGER_LOCAL") == "1"


def configured() -> bool:
    try:
        return "supabase_url" in st.secrets and "supabase_key" in st.secrets
    except Exception:
        return False


def _client():
    """One Supabase client PER browser session (never shared between users)."""
    if "sb" not in st.session_state:
        from supabase import create_client

        st.session_state["sb"] = create_client(st.secrets["supabase_url"], st.secrets["supabase_key"])
    return st.session_state["sb"]


def login_gate() -> bool:
    """Render the login form if needed. Returns True when the user may use the app."""
    if is_local():
        return True
    if not configured():
        st.error("This app is not configured for sign-in, so access is blocked.")
        return False
    if st.session_state.get("user"):
        with st.sidebar:
            st.caption(f"Signed in as {st.session_state['user']['email']}")
            if st.button("Sign out"):
                try:
                    _client().auth.sign_out()
                except Exception:
                    log.error("sign out failed", exc_info=True)
                st.session_state.clear()
                st.rerun()
        return True

    st.subheader("Sign in")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            res = _client().auth.sign_in_with_password({"email": email.strip(), "password": password})
            st.session_state["user"] = {"id": res.user.id, "email": res.user.email}
            st.rerun()
        except Exception:
            log.warning("failed sign-in attempt")
            st.error("Sign-in failed. Check your email and password.")
    return False


def library_enabled() -> bool:
    return not is_local() and bool(st.session_state.get("user"))


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:40] or "song"


def save_video(data: bytes, title: str) -> bool:
    try:
        path = f"{st.session_state['user']['id']}/{int(time.time())}_{_safe(title)}.mp4"
        _client().storage.from_(BUCKET).upload(path, data, {"content-type": "video/mp4"})
        return True
    except Exception:
        log.error("video upload failed", exc_info=True)
        return False


def list_videos() -> list[dict]:
    try:
        items = _client().storage.from_(BUCKET).list(
            st.session_state["user"]["id"],
            {"limit": 50, "sortBy": {"column": "created_at", "order": "desc"}},
        )
        return [i for i in items if i.get("name", "").endswith(".mp4")]
    except Exception:
        log.error("video list failed", exc_info=True)
        return []


def download_video(name: str) -> bytes | None:
    try:
        return _client().storage.from_(BUCKET).download(f"{st.session_state['user']['id']}/{name}")
    except Exception:
        log.error("video download failed", exc_info=True)
        return None
