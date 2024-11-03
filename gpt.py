from gpt4all import GPT4All
from TTS.api import TTS
import torch
import gradio as gr

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_response_and_audio(text_input="What is the definition of human computer interaction?"):
    with model.chat_session():
        llm_response = model.generate(text_input, max_tokens=1024)

    tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch').to(device)

    output_file = "LLM output/output.wav"
    tts.tts_to_file(text=llm_response, file_path=output_file)

    return llm_response, output_file

demo = gr.Interface(
    fn=generate_response_and_audio,
    inputs=[gr.Textbox(label="Text Input")],  
    outputs=[gr.Textbox(label="LLM Output"), gr.Audio(label="Generated Audio")],  
)

print(generate_response_and_audio())