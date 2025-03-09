import torch, warnings, tqdm
from typing import TYPE_CHECKING, Union, List, Tuple
import numpy as np
from .audio import (
    load_audio, log_mel_spectrogram, pad_or_trim, exact_div,
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,               # mel频谱有3000个
    N_SAMPLES,              # 30s样本有48000个
    SAMPLE_RATE
)
if TYPE_CHECKING:   # 避免循环引用
    from.audio_encoder import Whisper

def get_audio_features(
        model: "Whisper",
        audio: Union[str, List[str], np.ndarray, torch.Tensor],
        *,
        dtype = torch.float16,
    ):

    # 设置数值格式
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    # Pad 30-seconds of silence to the input audio, for slicing
    # 得到一个list, 每个元素是mel频谱的torch.tensor
    mel_list = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)  # 计算mel spectrogram
    mel_segment_list = []
    for mel in mel_list:
        if dtype == torch.float16:
            mel = mel.half()
        content_frames = mel.shape[-1] - N_FRAMES 
        mel = mel[:, : content_frames] # mel是整个音频的mel频谱表示, mel_segment表示当前音频片段窗口的mel频谱表示

        mel_segment_list.append(pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype))
    mel_segments = torch.stack(mel_segment_list)
    #print("mel_segments: ", mel_segments[0])
    #print("mel_segments.shape(batch_size, n_mels, n_frames): ", mel_segments.shape)

    if mel_segments.ndim == 2:
        mel_segments = mel_segments.unsqueeze(0)
    audio_embeds = model.encoder(mel_segments)
    return audio_embeds

