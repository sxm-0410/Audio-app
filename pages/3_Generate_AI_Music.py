import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio
from io import BytesIO
import numpy as np

st.set_page_config(page_title="🎶 Local AI Music Generator", layout="wide")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# UI
st.title("🎶 AI Music Generator (Local)")
col1, col2 = st.columns(2)
with col1:
    prompt = st.text_area("🎼 Describe your music:",
        "An upbeat lo-fi beat with mellow piano, soft drums, and vinyl crackle", height=100)
    duration = st.slider("🎵 Duration (seconds)", 5, 30, 10)
with col2:
    temperature = st.slider("🎨 Creativity (temperature)", 0.1, 2.0, 1.0, 0.1)
    guidance_scale = st.slider("🎯 Guidance scale", 1.0, 10.0, 3.0, 0.1)

if st.button("🚀 Generate Music"):
    with st.spinner("Generating music..."):
        try:
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(device)

            max_new_tokens = duration * 50

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    guidance_scale=guidance_scale
                )

            audio_values = output[0, 0].cpu().numpy()
            audio_values = np.clip(audio_values, -1.0, 1.0)
            audio_tensor = torch.tensor(audio_values).unsqueeze(0)

            sample_rate = model.config.audio_encoder.sampling_rate
            buffer = BytesIO()
            torchaudio.save(buffer, audio_tensor, sample_rate=sample_rate, format="wav")
            buffer.seek(0)

            # Output UI
            st.success("✅ Music generated!")
            st.audio(buffer, format="audio/wav")
            st.download_button("⬇ Download Music", buffer, file_name="music.wav", mime="audio/wav")

        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")