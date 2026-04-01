import subprocess
import streamlit as st
import tempfile
import requests
import os


def convert_to_mp4_bytes(video_bytes: bytes, original_suffix: str = ".mp4") -> bytes:
    """
    Convert any video format to a fixed-size MP4 (854x480) using FFmpeg.
    """
    in_file = tempfile.NamedTemporaryFile(suffix=original_suffix, delete=False)
    in_file.write(video_bytes)
    in_file.close()

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i", in_file.name,
        "-vf", "scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2:black",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_file.name
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(out_file.name, "rb") as f:
            mp4_bytes = f.read()
    finally:
        if os.path.exists(in_file.name):
            os.remove(in_file.name)
        if os.path.exists(out_file.name):
            os.remove(out_file.name)

    return mp4_bytes


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nepali Caption AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/*@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');*/

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #000000 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 90%, #000000 0%, transparent 60%) !important;
    background-blend-mode: screen !important;
    min-height: 100vh;
}

/* hide default header/footer/menu */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Custom Header ── */
.page-header {
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.page-header .label {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #7c6fff;
    margin-bottom: 0.5rem;
}
.page-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2rem, 3.2vw, 3rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: #f0ece4 !important;
    line-height: 1.1 !important;
}
.page-header h1 span { color: #7c6fff; }
.page-header .sub {
    font-size: 1.5rem;
    color: #ffffff;
    margin-top: 0.7rem;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Column Divider ── */
.col-divider {
    width: 1px;
    background: linear-gradient(to bottom, transparent, rgba(124,111,255,0.3) 30%, rgba(124,111,255,0.3) 70%, transparent);
    margin: 0 auto;
}

/* ── Panel Labels ── */
.panel-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #ffffff;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(124,111,255,0.25) !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    transition: border-color 0.25s ease, background 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(124,111,255,0.55) !important;
    background: rgba(124,111,255,0.04) !important;
}
[data-testid="stFileUploader"] label {
    color: #9490a8 !important;
    font-size: 1.2rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploader"] small {
    color: #4a4760 !important;
    font-size: 1rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: #6b6878 !important;
}

/* ── File info badge ── */
.file-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(124,111,255,0.1);
    border: 1px solid rgba(124,111,255,0.2);
    border-radius: 8px;
    padding: 0.45rem 0.85rem;
    font-size: 0.78rem;
    color: #a09cc0;
    font-family: 'DM Sans', sans-serif;
    margin: 0.8rem 0 1.2rem;
}
.file-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7c6fff;
    box-shadow: 0 0 6px #7c6fff;
}

/* ── Generate Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7c6fff 0%, #5b4fcf 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(124,111,255,0.25) !important;
    margin-top: 0.5rem !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,111,255,0.4) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Caption Card ── */
.caption-section {
    margin-top: 1.8rem;
}
.caption-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #ffffff;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.caption-header::after {
    content: ''; flex: 1;
    height: 1px; background: rgba(255,255,255,0.05);
}
.caption-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.65rem;
    position: relative;
    overflow: hidden;
}
.caption-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(to bottom, #7c6fff, #3ecfb2);
    border-radius: 3px 0 0 3px;
}
.caption-idx {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c6fff;
    margin-bottom: 0.3rem;
}
.caption-text {
    font-size: 2rem;
    color: #ccc8be;
    line-height: 1.65;
    font-weight: 300;
}

/* ── Video Preview ── */
.preview-wrap {
    background: #07070d;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    overflow: hidden;
    aspect-ratio: 16/9;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
.preview-wrap video {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    border-radius: 0 !important;
    border: none !important;
}
[data-testid="stVideo"] {
    width: 100% !important;
}
[data-testid="stVideo"] video {
    width: 100% !important;
    height: auto !important;
    max-height: 360px !important;
    object-fit: contain !important;
    border-radius: 12px !important;
    background: #07070d !important;
}

/* ── Empty Preview Placeholder ── */
.empty-preview {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 3rem 2rem;
    background: rgba(255,255,255,0.015);
    border: 1.5px dashed rgba(255,255,255,0.07);
    border-radius: 16px;
    text-align: center;
}
.empty-preview .icon {
    font-size: 2.5rem;
    opacity: 0.3;
    filter: grayscale(1);
}
.empty-preview p {
    font-size: 0.82rem;
    color: #3d3a4d;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Alert overrides ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #9490a8 !important;
    font-size: 0.84rem !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #6b6878 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2740; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="label">AI · Video Understanding</div>
    <h1>Video <span>Caption</span> Generator In <span>Nepali</span> Language</h1>
    <div class="sub">Upload a video and let AI transcribe it into Nepali captions</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ─── Layout ───────────────────────────────────────────────────────────────────
left_col, spacer, right_col = st.columns([1, 0.05, 1.3])

# ════════════════════════════════════════════════════════
# LEFT PANEL — Upload + Controls + Captions
# ════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="panel-label">Input</div>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Drop a video file here",
        type=["mp4", "avi", "mov", "mkv", "webm", "flv"],
        label_visibility="collapsed"
    )

    # File badge
    if uploaded_video is not None:
        ext = os.path.splitext(uploaded_video.name)[1].upper().lstrip(".")
        size_kb = len(uploaded_video.getvalue()) // 1024
        size_str = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb} KB"
        st.markdown(f"""
        <div class="file-badge">
            <div class="dot"></div>
            <span>{uploaded_video.name}</span>
            <span style="color:#4a4760;margin-left:0.25rem">· {ext} · {size_str}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    generate_clicked = st.button("⚡ Generate Caption")

    # ── Caption output ──
    if generate_clicked:
        if uploaded_video is None:
            st.warning("Please upload a video file first.")
        else:
            with st.spinner("Analysing video and generating captions…"):
                uploaded_video.seek(0)
                files = {
                    "file": (
                        uploaded_video.name,
                        uploaded_video,
                        uploaded_video.type
                    )
                }
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/generate_caption/",
                        files=files
                    )
                    if response.status_code == 200:
                        data = response.json()
                        captions = data["caption"]

                        st.markdown("""
                        <div class="caption-section">
                            <div class="caption-header">Captions</div>
                        """, unsafe_allow_html=True)

                        for idx, item in enumerate(captions):
                            st.markdown(f"""
                            <div class="caption-card">
                                <div class="caption-idx">Caption {idx + 1}</div>
                                <div class="caption-text">{item}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Caption generation failed. Check if the API server is running.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API server at localhost:8000.")

# ════════════════════════════════════════════════════════
# SPACER — vertical line
# ════════════════════════════════════════════════════════
with spacer:
    st.markdown("""
    <div style="display:flex;justify-content:center;height:100%;min-height:400px">
        <div style="width:1px;background:linear-gradient(to bottom,transparent,rgba(124,111,255,0.25) 20%,rgba(124,111,255,0.25) 80%,transparent);flex-shrink:0;"></div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# RIGHT PANEL — Video Preview
# ════════════════════════════════════════════════════════
with right_col:
    st.markdown('<div class="panel-label"> Video Preview</div>', unsafe_allow_html=True)

    if uploaded_video is not None:
        with st.spinner("Processing video…"):
            video_bytes = uploaded_video.getvalue()
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            try:
                mp4_bytes = convert_to_mp4_bytes(video_bytes, original_suffix=suffix)
                st.video(mp4_bytes)
            except Exception as e:
                # Fallback: try to play original
                st.video(video_bytes)
    else:
        st.markdown("""
        <div class="empty-preview">
            <div class="icon">🎞️</div>
            <p>Video preview will appear here<br>after you upload a file</p>
        </div>
        """, unsafe_allow_html=True)