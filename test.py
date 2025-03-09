import whisper_encoder
from whisper_encoder import ModelDimensions, Whisper, load_audio, log_mel_spectrogram, pad_or_trim
from whisper_encoder.audio import N_SAMPLES
import torch
import numpy as np
import os
import whisper

model = whisper_encoder.load_model("tiny")
model1 = whisper.load_model("tiny")
options = dict(language="Chinese", beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)

base_path = "D:\\Scientific Research\\MyClap\\test_audio\\"
audio_names = ['5s.wav', '11s.wav', '39s.wav']
for audio_name in audio_names:
    audio_path = os.path.join(base_path, audio_name)

    audio_embeds = model.embed_audio(audio_path)

    transcription = model1.transcribe(audio_path, **transcribe_options)["text"]
    
    if torch.equal(model1.audio_features, audio_embeds):
        print(f"{audio_name}: Embeddings are equal.")

    model1.whether_is_first_chunk = False