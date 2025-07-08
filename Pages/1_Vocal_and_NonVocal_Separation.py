import streamlit as st
from pytube import YouTube
from pytube.exceptions import PytubeError
import os
from pydub import AudioSegment
import shutil
from tempfile import NamedTemporaryFile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
import warnings

warnings.filterwarnings("ignore")

os.environ["PATH"] += os.pathsep + r"C:\Users\sampa\OneDrive\Desktop\ffmpeg-2025-04-23-git-25b0a8e295-full_build\bin"
AudioSegment.converter = r"C:\Users\sampa\OneDrive\Desktop\ffmpeg-2025-04-23-git-25b0a8e295-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\sampa\OneDrive\Desktop\ffmpeg-2025-04-23-git-25b0a8e295-full_build\bin\ffprobe.exe"

st.set_page_config(
    page_title="Vocal & Non-Vocal Studio",
    layout="centered",
    page_icon="🎙️"
)
st.title("🎙️ Vocal & Non-Vocal Studio")
st.write("Separate, Preview, Control Volume, Jump to Timestamps 🎛️🎧")


def clean_youtube_url(url):
    """Clean and standardize YouTube URLs"""
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url.split("&")[0] if "&" in url else url


def convert_to_wav(audio_file):
    """Convert any audio file to WAV format"""
    try:
        sound = AudioSegment.from_file(audio_file)
        temp_wav = NamedTemporaryFile(delete=False, suffix=".wav")
        sound.export(temp_wav.name, format="wav")
        return temp_wav.name
    except Exception as e:
        st.error(f"Error converting file: {str(e)}")
        return None


def separate_audio(input_wav):
    """Separate audio using Demucs"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(name='htdemucs')
        model.to(device)

        with st.spinner("Separating audio components..."):
            stems = apply_model(model, input_wav, device=device)

        output_dir = os.path.splitext(input_wav)[0] + "_output"
        os.makedirs(output_dir, exist_ok=True)

        vocals_path = os.path.join(output_dir, "vocals.wav")
        accompaniment_path = os.path.join(output_dir, "accompaniment.wav")

        stems[0].save(vocals_path)
        accompaniment = stems[1] + stems[2] + stems[3]
        accompaniment.save(accompaniment_path)

        return vocals_path, accompaniment_path
    except Exception as e:
        st.error(f"Error during separation: {str(e)}")
        return None, None


def download_button(label, filepath):
    """Create a download button for audio files"""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            st.download_button(
                label=label,
                data=f,
                file_name=os.path.basename(filepath),
                mime="audio/wav"
            )


def plot_waveform(filepath, title):
    """Plot waveform of an audio file"""
    try:
        y, sr = librosa.load(filepath, sr=None)
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting waveform: {str(e)}")


def process_audio_player(filepath, label):
    """Create an interactive audio player"""
    if not os.path.exists(filepath):
        st.warning(f"Audio file not found: {filepath}")
        return

    col1, col2 = st.columns(2)
    with col1:
        volume = st.slider(f"{label} Volume (dB)", -20, 20, 0, 1)
    with col2:
        start_time = st.slider(f"{label} Start Time (sec)", 0, 300, 0, 1)

    try:
        audio = AudioSegment.from_wav(filepath)
        audio = audio + volume
        trimmed = audio[start_time * 1000:]

        buffer = BytesIO()
        trimmed.export(buffer, format="wav")
        buffer.seek(0)

        st.audio(buffer, format="audio/wav")
        st.caption(f"{label} - Volume: {volume}dB | Start at {start_time}s")
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")


def process_youtube_video(url):
    """Handle YouTube video processing with better error handling"""
    try:
        clean_url = clean_youtube_url(url)

        with st.spinner("Connecting to YouTube..."):
            yt = YouTube(clean_url)
            st.write(f"Processing: {yt.title}")

            stream = yt.streams.filter(only_audio=True).order_by('abr').last()
            if not stream:
                raise PytubeError("No suitable audio stream found")

            with st.spinner("Downloading audio..."):
                temp_file = f"yt_audio_{yt.video_id}.mp4"
                stream.download(filename=temp_file)

                wav_file = convert_to_wav(temp_file)
                if not wav_file:
                    raise Exception("Failed to convert audio")

                vocals, accompaniment = separate_audio(wav_file)
                if not vocals or not accompaniment:
                    raise Exception("Failed to separate audio")

                return vocals, accompaniment

    except PytubeError as e:
        st.error(f"YouTube processing error: {str(e)}")
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    return None, None

# UI
tab1, tab2 = st.tabs(["📁 Upload Audio File", "📹 YouTube Link"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload an audio file (MP3, WAV, M4A)",
        type=["mp3", "wav", "m4a"]
    )

    if uploaded_file:
        wav_file = convert_to_wav(uploaded_file)
        if wav_file:
            vocals, accompaniment = separate_audio(wav_file)
            if vocals and accompaniment:
                st.success("Separation complete! 🎉")

                st.subheader("🔊 Vocals")
                process_audio_player(vocals, "Vocals")
                plot_waveform(vocals, "Vocals Waveform")

                st.subheader("🎸 Non-Vocals")
                process_audio_player(accompaniment, "Non-Vocals")
                plot_waveform(accompaniment, "Non-Vocals Waveform")

                st.markdown("---")
                download_button("🎤 Download Vocals", vocals)
                download_button("🎸 Download Non-Vocals", accompaniment)

with tab2:
    yt_url = st.text_input("Paste YouTube URL:", key="yt_url")
    if yt_url and st.button("Process YouTube Video"):
        vocals, accompaniment = process_youtube_video(yt_url)
        if vocals and accompaniment:
            st.success("Separation complete! 🎉")

            st.subheader("🔊 Vocals")
            process_audio_player(vocals, "Vocals")
            plot_waveform(vocals, "Vocals Waveform")

            st.subheader("🎸 Non-Vocals")
            process_audio_player(accompaniment, "Non-Vocals")
            plot_waveform(accompaniment, "Non-Vocals Waveform")

            st.markdown("---")
            download_button("🎤 Download Vocals", vocals)
            download_button("🎸 Download Non-Vocals", accompaniment)


if st.button("🧹 Clear Temporary Files"):
    try:
        for file in os.listdir("."):
            if file.endswith((".wav", ".mp4", ".mp3")) or "_output" in file:
                try:
                    os.remove(file)
                except:
                    pass
        st.success("Temporary files cleared!")
    except Exception as e:
        st.error(f"Error clearing files: {str(e)}")