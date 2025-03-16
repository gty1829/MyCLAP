import whisper_MoE
from whisper_MoE import ModelDimensions, Whisper_MoE, load_audio, log_mel_spectrogram, pad_or_trim
from whisper_MoE.audio import N_SAMPLES
import os


model = whisper_MoE.load_model("tiny")              # 这是我扒出来的encoder代码

base_path = "D:\\Scientific Research\\MyClap\\test_audio\\"
audio_names = ['5s.wav', '11s.wav', '39s.wav']
audio_paths = [os.path.join(base_path, audio_name) for audio_name in audio_names]

audio_feature = model.embed_audio(audio_paths)
print(audio_feature.shape)  # (batch_size, n_frames, n_audio_states) = (batch_size, 1500, n_audio_states)
                            # 1500(frame)  = 30(s) / 0.02(s/frame)







