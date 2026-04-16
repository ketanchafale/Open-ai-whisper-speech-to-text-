# -speech-to-text-
import streamlit as st
import openai
import os
import tempfile
from pathlib import Path

st.set_page_config(page_title="Whisper Transcribe", page_icon="🎙️")
st.title("🎙️ Audio Transcription with Whisper")

api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")


def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> str:
    client = openai.OpenAI(api_key=api_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    os.unlink(tmp_path)
    return result


def show_result(text: str):
    st.success("Done!")
    st.text_area("Transcript", text, height=300)
    st.download_button("⬇️ Download transcript", text, file_name="transcript.txt")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab_record, tab_upload = st.tabs(["🎤 Record Audio", "📁 Upload File"])

# ── TAB 1: Live recorder ──────────────────────────────────────────────────────
with tab_record:
    st.markdown("Click **Start** to record, then **Stop** when done.")

    try:
        from audiorecorder import audiorecorder

        audio_segment = audiorecorder("▶ Start Recording", "⏹ Stop Recording")

        if len(audio_segment) > 0:
            st.audio(audio_segment.export().read(), format="audio/wav")

            if st.button("Transcribe Recording", key="btn_record"):
                if not api_key:
                    st.error("Please enter your OpenAI API key.")
                else:
                    with st.spinner("Transcribing…"):
                        try:
                            wav_bytes = audio_segment.export(format="wav").read()
                            result = transcribe(wav_bytes, suffix=".wav")
                            show_result(result)
                        except Exception as e:
                            st.error(f"Error: {e}")

    except ImportError:
        st.warning(
            "`streamlit-audiorecorder` is not installed.\n\n"
            "Run: `pip install streamlit-audiorecorder` then restart the app."
        )

# ── TAB 2: File upload ────────────────────────────────────────────────────────
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
    )

    if uploaded_file:
        st.audio(uploaded_file)

    if st.button("Transcribe Upload", key="btn_upload"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif not uploaded_file:
            st.error("Please upload an audio file.")
        else:
            with st.spinner("Transcribing…"):
                try:
                    suffix = Path(uploaded_file.name).suffix
                    result = transcribe(uploaded_file.read(), suffix=suffix)
                    show_result(result)
                except Exception as e:
                    st.error(f"Error: {e}")
