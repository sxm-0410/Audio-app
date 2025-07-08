# app.py
import streamlit as st
import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import noisereduce as nr
from pydub import AudioSegment
from pytube import YouTube
from pytube.exceptions import PytubeError
from tempfile import NamedTemporaryFile
from io import BytesIO
import soundfile as sf
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings("ignore")

# ✅ UPDATE THIS to your actual ffmpeg/bin folder path
ffmpeg_path = r"C:\Users\sampa\Downloads\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")

st.set_page_config(page_title="AuraLift", layout="wide")

@st.cache_resource
def load_musicgen_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_musicgen_model()

def convert_to_wav(audio_file):
    try:
        sound = AudioSegment.from_file(audio_file)
        temp_wav = NamedTemporaryFile(delete=False, suffix=".wav")
        sound.export(temp_wav.name, format="wav")
        return temp_wav.name
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
        return None

def separate_audio(input_wav):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(name='htdemucs')
        model.to(device)

        # Load and resample audio to 44.1kHz
        wav, sr = torchaudio.load(input_wav)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
            wav = resampler(wav)
            sr = 44100

        wav = wav.unsqueeze(0).to(device)  # [1, channels, samples]

        with st.spinner("Separating audio..."):
            stems = apply_model(model, wav, device=device)

        output_dir = os.path.splitext(input_wav)[0] + "_output"
        os.makedirs(output_dir, exist_ok=True)

        vocals_path = os.path.join(output_dir, "vocals.wav")
        accompaniment_path = os.path.join(output_dir, "accompaniment.wav")

        # 💥 FIXED: remove batch dim for torchaudio.save()
        torchaudio.save(vocals_path, stems[0].squeeze(0).cpu(), sample_rate=sr)
        acc = stems[1] + stems[2] + stems[3]
        torchaudio.save(accompaniment_path, acc.squeeze(0).cpu(), sample_rate=sr)

        return vocals_path, accompaniment_path

    except Exception as e:
        st.error(f"Error during separation: {str(e)}")
        return None, None


def plot_waveform(filepath, title):
    y, sr = librosa.load(filepath, sr=None)
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def process_audio_player(filepath, label):
    col1, col2 = st.columns(2)
    with col1:
        volume = st.slider(f"{label} Volume (dB)", -20, 20, 0, 1)
    with col2:
        start_time = st.slider(f"{label} Start Time (sec)", 0, 300, 0, 1)
    audio = AudioSegment.from_wav(filepath)
    audio = audio + volume
    trimmed = audio[start_time * 1000:]
    buffer = BytesIO()
    trimmed.export(buffer, format="wav")
    buffer.seek(0)
    st.audio(buffer, format="audio/wav")
    st.caption(f"{label} - Volume: {volume}dB | Start at {start_time}s")

def apply_reverb(audio, sr, reverb_amount=0.3):
    ir = np.zeros(int(0.3 * sr))
    ir[0] = 1.0
    ir[int(0.03 * sr)] = 0.6
    ir[int(0.06 * sr)] = 0.3
    convolved = scipy.signal.convolve(audio, ir, mode='full')[:len(audio)]
    return (1 - reverb_amount) * audio + reverb_amount * convolved

def apply_eq(audio, sr, bass_gain=1.0, low_mid_gain=1.0, high_mid_gain=1.0, treble_gain=1.0):
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    fft[freqs < 100] *= bass_gain
    fft[(freqs >= 100) & (freqs < 500)] *= low_mid_gain
    fft[(freqs >= 500) & (freqs < 5000)] *= high_mid_gain
    fft[freqs >= 5000] *= treble_gain
    return np.fft.irfft(fft, n=len(audio))

def apply_chorus(audio, sr, depth=0.003, rate=0.25, mix=0.5):
    n_samples = len(audio)
    delay_samples = int(depth * sr)
    mod = (np.sin(2 * np.pi * rate * np.arange(n_samples) / sr) * delay_samples).astype(int)
    chorus_audio = np.zeros_like(audio)
    for i in range(delay_samples, n_samples):
        chorus_audio[i] = audio[i] + audio[i - mod[i]]
    chorus_audio /= np.max(np.abs(chorus_audio))
    return (1 - mix) * audio + mix * chorus_audio

def apply_autotune(audio, sr):
    try:
        pitches, mags = librosa.piptrack(y=audio, sr=sr)
        pitch = np.mean(pitches[mags > np.median(mags)]) if np.any(mags > np.median(mags)) else 0
        if pitch <= 0 or np.isnan(pitch): return audio
        target_hz = librosa.note_to_hz(librosa.hz_to_note(pitch))
        n_steps = 12 * np.log2(target_hz / pitch)
        return librosa.effects.pitch_shift(audio, sr, n_steps)
    except: return audio

def apply_preset(audio, sr, preset_name):
    if preset_name == "Radio":
        return apply_eq(audio, sr, 0.6, 1.5, 1.3, 1.2)
    elif preset_name == "Concert Hall":
        return apply_reverb(audio, sr, 0.7)
    elif preset_name == "Lo-fi":
        audio = librosa.effects.time_stretch(audio, 0.8)
        return apply_eq(audio, sr, 1.2, 0.9, 0.7, 0.6)
    return audio

def apply_all_effects(y, sr, opts):
    if opts['preset'] != 'Custom':
        y = apply_preset(y, sr, opts['preset'])
    if opts['autotune']: y = apply_autotune(y, sr)
    if opts['noise_reduction']: y = nr.reduce_noise(y=y, sr=sr)
    if opts['time_stretch'] != 1.0: y = librosa.effects.time_stretch(y, rate=opts['time_stretch'])
    if opts['reverb'] > 0: y = apply_reverb(y, sr, opts['reverb'])
    y = apply_eq(y, sr, opts['eq_bass'], opts['eq_low_mid'], opts['eq_high_mid'], opts['eq_treble'])
    if opts['chorus'] > 0: y = apply_chorus(y, sr, mix=opts['chorus'])
    return y / np.max(np.abs(y))

# UI
st.title("🎧 AuraLift")
tab1, tab2, tab3 = st.tabs(["🔀 Vocal Split", "🎛️ Vocal FX", "🎶 AI Music"])

with tab1:
    st.subheader("Upload or Paste YouTube Link")

    if 'separation_done' not in st.session_state:
        st.session_state.separation_done = False

    uploaded_file = st.file_uploader("Upload MP3/WAV/M4A", type=["mp3", "wav", "m4a"])
    yt_link = st.text_input("Or paste YouTube link")

    if uploaded_file and not st.session_state.separation_done:
        wav = convert_to_wav(uploaded_file)
        if wav:
            vocals, nonvocals = separate_audio(wav)
            if vocals and nonvocals:
                st.session_state.vocals = vocals
                st.session_state.nonvocals = nonvocals
                st.session_state.separation_done = True

    elif yt_link and st.button("Process YouTube"):
        yt = YouTube(yt_link)
        stream = yt.streams.filter(only_audio=True).last()
        stream.download(filename="yt_audio.mp4")
        wav = convert_to_wav("yt_audio.mp4")
        vocals, nonvocals = separate_audio(wav)
        if vocals and nonvocals:
            st.session_state.vocals = vocals
            st.session_state.nonvocals = nonvocals
            st.session_state.separation_done = True

    # ✅ Once done, show results
    if st.session_state.separation_done:
        st.success("Separation complete! 🎉")

        st.subheader("🎤 Vocals")
        process_audio_player(st.session_state.vocals, "Vocals")
        plot_waveform(st.session_state.vocals, "Vocals")

        st.subheader("🎸 Non-Vocals")
        process_audio_player(st.session_state.nonvocals, "Non-Vocals")
        plot_waveform(st.session_state.nonvocals, "Non-Vocals")

        if st.button("🔁 Process another file"):
            st.session_state.separation_done = False
            st.experimental_rerun()


with tab2:
    st.sidebar.header("FX Controls")
    fx_file = st.sidebar.file_uploader("Upload vocals", type=["wav", "mp3"])
    preset = st.sidebar.selectbox("Preset", ["Custom", "Radio", "Concert Hall", "Lo-fi"])
    autotune = st.sidebar.checkbox("Autotune")
    noise = st.sidebar.checkbox("Noise Reduction")
    stretch = st.sidebar.slider("Time Stretch", 0.5, 2.0, 1.0, 0.1)
    reverb = st.sidebar.slider("Reverb", 0.0, 1.0, 0.0, 0.05)
    chorus = st.sidebar.slider("Chorus", 0.0, 1.0, 0.0, 0.05)
    eq_bass = st.sidebar.slider("Bass", 0.5, 2.0, 1.0, 0.1)
    eq_low = st.sidebar.slider("Low-Mid", 0.5, 2.0, 1.0, 0.1)
    eq_high = st.sidebar.slider("High-Mid", 0.5, 2.0, 1.0, 0.1)
    eq_treble = st.sidebar.slider("Treble", 0.5, 2.0, 1.0, 0.1)

    if fx_file:
        st.audio(fx_file)
        y, sr = librosa.load(fx_file, sr=None)
        opts = {
            "preset": preset, "autotune": autotune, "noise_reduction": noise,
            "time_stretch": stretch, "reverb": reverb, "chorus": chorus,
            "eq_bass": eq_bass, "eq_low_mid": eq_low, "eq_high_mid": eq_high, "eq_treble": eq_treble
        }
        if st.button("🎚️ Apply Effects"):
            out = apply_all_effects(y, sr, opts)
            buffer = BytesIO()
            sf.write(buffer, out, sr, format='WAV')
            buffer.seek(0)
            st.success("✅ Processed!")
            st.audio(buffer, format="audio/wav")
            st.download_button("⬇️ Download", buffer, "processed.wav", "audio/wav")

with tab3:
    prompt = st.text_area("🎼 Describe your music", "Lo-fi beat with rain sounds")
    duration = st.slider("🎵 Duration (s)", 5, 30, 10)
    temperature = st.slider("🎨 Creativity", 0.1, 2.0, 1.0)
    guidance = st.slider("🎯 Guidance scale", 1.0, 10.0, 3.0)
    if st.button("🚀 Generate Music"):
        with st.spinner("Generating..."):
            inputs = processor(text=[prompt], return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=duration * 50,
                                    do_sample=True, temperature=temperature, guidance_scale=guidance)
            audio = output[0, 0].cpu().numpy()
            audio = np.clip(audio, -1, 1)
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            buffer = BytesIO()
            torchaudio.save(buffer, audio_tensor, sample_rate=model.config.audio_encoder.sampling_rate, format="wav")
            buffer.seek(0)
            st.success("✅ Done!")
            st.audio(buffer, format="audio/wav")
            st.download_button("⬇ Download", buffer, "music.wav", "audio/wav")
