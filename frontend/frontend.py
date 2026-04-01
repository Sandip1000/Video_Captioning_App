import subprocess
import streamlit as st
import tempfile
import requests
import os


def avi_to_mp4_bytes(avi_bytes: bytes) -> bytes:
    """
    Convert AVI video bytes to MP4 bytes using FFmpeg on Windows.
    """
    # Create temp input file
    in_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    in_file.write(avi_bytes)
    in_file.close()  # Close so FFmpeg can access it

    # Create temp output file
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i", in_file.name,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        out_file.name
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(out_file.name, "rb") as f:
            mp4_bytes = f.read()
    finally:
        # Clean up temp files
        if os.path.exists(in_file.name):
            os.remove(in_file.name)
        if os.path.exists(out_file.name):
            os.remove(out_file.name)

    return mp4_bytes





st.set_page_config(
    page_title="Nepali Video Caption Generator",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 Nepali Video Caption Generator")
st.write("Upload a video and generate its caption")

# CSS for fixed video size
st.markdown("""
<style>
.video-box video {
    width: 480px !important;
    height: 270px !important;
    object-fit: contain;
    border-radius: 10px;
    border: 2px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov"]
)

# ✅ VIDEO PREVIEW (THIS WILL PLAY)
if uploaded_video is not None:
    st.subheader("🎬 Video Preview")

    avi_bytes = uploaded_video.read()
    mp4_bytes = avi_to_mp4_bytes(avi_bytes)


    st.markdown('<div class="video-box">', unsafe_allow_html=True)
    st.video(mp4_bytes)
    st.markdown('</div>', unsafe_allow_html=True)

    # IMPORTANT: reset pointer for API upload
    uploaded_video.seek(0)

if st.button("Generate Caption"):
    if uploaded_video is None:
        st.warning("⚠️ Please upload a video first")
    else:
        with st.spinner("⏳ Generating caption..."):

            files = {
                "file": (
                    uploaded_video.name,
                    uploaded_video,
                    uploaded_video.type
                )
            }

            response = requests.post(
                "http://127.0.0.1:8000/generate_caption/",
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                captions = data["caption"]
                st.success("✅ Caption generated")
                st.subheader("Caption")
                for idx, item in enumerate(captions):
                    st.write(f"Caption {idx}: {item}")
            else:
                st.error("❌ Failed to generate caption")