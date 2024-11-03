import torch
from TTS.api import TTS
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_audio(text="A journey of a thousand miles begins with a single step."):
    tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch').to(device)
    tts.tts_to_file(text=text,file_path="outputs/output.wav")
    return "outputs/output.wav"

demo = gr.Interface(
    fn=generate_audio,
    inputs=[gr.Text(label="Text"),],
    outputs=[gr.Audio(label="Audio"),],
)

print(generate_audio())