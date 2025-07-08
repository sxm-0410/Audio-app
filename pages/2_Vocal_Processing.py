import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import scipy.signal
import noisereduce as nr
from io import BytesIO

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

    bass = freqs < 100
    low_mid = (freqs >= 100) & (freqs < 500)
    high_mid = (freqs >= 500) & (freqs < 5000)
    treble = freqs >= 5000

    fft[bass] *= bass_gain
    fft[low_mid] *= low_mid_gain
    fft[high_mid] *= high_mid_gain
    fft[treble] *= treble_gain

    return np.fft.irfft(fft, n=len(audio))

def apply_chorus(audio, sr, depth=0.003, rate=0.25, mix=0.5):
    n_samples = len(audio)
    delay_samples = int(depth * sr)
    mod = np.sin(2 * np.pi * rate * np.arange(n_samples) / sr) * delay_samples
    mod = mod.astype(int)
    chorus_audio = np.zeros_like(audio)
    for i in range(delay_samples, n_samples):
        chorus_audio[i] = audio[i] + audio[i - mod[i]]
    chorus_audio /= np.max(np.abs(chorus_audio))
    return (1 - mix) * audio + mix * chorus_audio

def apply_autotune(audio, sr):
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

        if pitch <= 0 or np.isnan(pitch):
            return audio

        note = librosa.hz_to_note(pitch)
        target_hz = librosa.note_to_hz(note)
        n_steps = 12 * np.log2(target_hz / pitch)

        return librosa.effects.pitch_shift(audio, sr, n_steps)
    except Exception as e:
        print("Autotune failed:", str(e))
        return audio

def apply_preset(audio, sr, preset_name):
    if preset_name == "Radio":
        audio = apply_eq(audio, sr, bass_gain=0.6, low_mid_gain=1.5, high_mid_gain=1.3, treble_gain=1.2)
    elif preset_name == "Concert Hall":
        audio = apply_reverb(audio, sr, reverb_amount=0.7)
    elif preset_name == "Lo-fi":
        audio = librosa.effects.time_stretch(audio, rate=0.8)
        audio = apply_eq(audio, sr, bass_gain=1.2, low_mid_gain=0.9, high_mid_gain=0.7, treble_gain=0.6)
    return audio

def apply_all_effects(y, sr, opts):
    if opts['preset'] != 'Custom':
        y = apply_preset(y, sr, opts['preset'])

    if opts['autotune']:
        y = apply_autotune(y, sr)

    if opts['noise_reduction']:
        y = nr.reduce_noise(y=y, sr=sr)

    if opts['time_stretch'] != 1.0:
        y = librosa.effects.time_stretch(y, rate=opts['time_stretch'])

    if opts['reverb'] > 0:
        y = apply_reverb(y, sr, reverb_amount=opts['reverb'])

    y = apply_eq(y, sr, opts['eq_bass'], opts['eq_low_mid'], opts['eq_high_mid'], opts['eq_treble'])

    if opts['chorus'] > 0:
        y = apply_chorus(y, sr, mix=opts['chorus'])

    y = y / np.max(np.abs(y))
    return y

# UI
st.set_page_config(page_title="Vocal FX Rack", layout="wide")
st.title("🎤 Vocal FX Rack")

with st.sidebar:
    st.header("Upload Vocal")
    uploaded_file = st.file_uploader("Select an audio file", type=["wav", "mp3"])

    st.header("FX Controls")
    preset = st.selectbox("Preset", ["Custom", "Radio", "Concert Hall", "Lo-fi"])
    autotune = st.checkbox("Autotune (basic)")
    noise_reduction = st.checkbox("Noise Reduction")
    time_stretch = st.slider("Time Stretch", 0.5, 2.0, 1.0, 0.1)
    reverb = st.slider("Reverb Amount", 0.0, 1.0, 0.0, 0.05)
    chorus = st.slider("Chorus / Flanger", 0.0, 1.0, 0.0, 0.05)

    st.subheader("Equalizer")
    eq_bass = st.slider("Bass (<100 Hz)", 0.5, 2.0, 1.0, 0.1)
    eq_low_mid = st.slider("Low-Mid (100–500 Hz)", 0.5, 2.0, 1.0, 0.1)
    eq_high_mid = st.slider("High-Mid (500–5k Hz)", 0.5, 2.0, 1.0, 0.1)
    eq_treble = st.slider("Treble (>5k Hz)", 0.5, 2.0, 1.0, 0.1)

if uploaded_file:
    st.subheader("Original Audio")
    st.audio(uploaded_file, format='audio/wav')

    y, sr = librosa.load(uploaded_file, sr=None)
    opts = {
        "preset": preset,
        "autotune": autotune,
        "noise_reduction": noise_reduction,
        "time_stretch": time_stretch,
        "reverb": reverb,
        "chorus": chorus,
        "eq_bass": eq_bass,
        "eq_low_mid": eq_low_mid,
        "eq_high_mid": eq_high_mid,
        "eq_treble": eq_treble
    }

    if st.button("🧪 Apply Effects"):
        with st.spinner("Processing..."):
            try:
                y_fx = apply_all_effects(y, sr, opts)
                buffer = BytesIO()
                sf.write(buffer, y_fx, sr, format='WAV')
                buffer.seek(0)

                st.success("✅ Done!")
                st.subheader("Processed Audio")
                st.audio(buffer, format='audio/wav')
                st.download_button("⬇️ Download", buffer, file_name="processed.wav", mime="audio/wav")
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
else:
    st.info("Upload a vocal track to begin.")
