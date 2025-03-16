import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F

def exact_div(x, y):
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000     # 样本采样率，每秒采样多少个样本点
N_FFT = 400             # FFT点数
HOP_LENGTH = 160        # 每帧之间时间间隔(帧移长度)160个点，HOP_LENGTH / SAMPLE_RATE = 每帧间隔时间
CHUNK_LENGTH = 30       # 音频窗口长度30s
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk 一个音频窗口内的采样点数
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input 音频窗口对应mel频谱帧数(横轴大小)

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2 卷积步幅为2，每个token对应2帧
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame 每秒帧数 1帧10ms, HOP_LENGHT / SAMPLE_RATE = 每帧间隔时间
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token 每秒token数，每个token20ms

'''
def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
'''

def load_audio(file_or_dir: Union[str, List[str]], sr: int = SAMPLE_RATE):
    """
    读取单个音频文件或文件夹下的所有音频文件，并返回对应的音频数据。
    
    参数:
    file_or_dir: str 或 List[str]
        - 单个音频文件路径 (str)
        - 文件夹路径 (str)，读取所有音频文件
        - 传入多个文件路径的列表 (List[str])
    
    sr: int
        - 采样率，默认 `SAMPLE_RATE`

    返回:
    - 单个 NumPy 数组（如果是单个音频）
    - 多个音频文件的 NumPy 数组列表（如果是文件夹）
    """

    def process_audio(file_path: str):
        """读取并处理单个音频文件"""
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file_path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        try:
            out = run(cmd, capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # 处理单个文件
    if isinstance(file_or_dir, str):
        if os.path.isfile(file_or_dir):  # 单个音频文件
            return process_audio(file_or_dir)
        elif os.path.isdir(file_or_dir):  # 文件夹
            audio_files = [
                os.path.join(file_or_dir, f)
                for f in os.listdir(file_or_dir)
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
            ]
            if not audio_files:
                raise RuntimeError("No audio files found in the directory.")
            return [process_audio(f) for f in sorted(audio_files)]  # 处理所有音频并返回列表

    # 处理多个文件
    elif isinstance(file_or_dir, list):
        return [process_audio(f) for f in file_or_dir]

    else:
        raise ValueError("Invalid input. Expected a file path, directory path, or a list of file paths.")

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    补全或截取音频到指定长度, 适应编码器的输入要求。输入形状可以为(batch_size, n_mels, n_frames)或(n_mels, n_frames)。
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    #print(array.shape)
    if torch.is_tensor(array):
        # 默认axis=-1, 表示时间维度
        if array.shape[axis] > length:
            # 截取前length个长度
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            # 所有维度的填充数量
            pad_widths = [(0, 0)] * array.ndim 
            # 计算沿着时间轴需要补0的数量
            pad_widths[axis] = (0, length - array.shape[axis])
            # F.pad是倒叙填充, 填充顺序为[最后一维的填充数量, 倒数第二维的填充数量, ..., 第一维的填充数量]
            # 每个元素是一个元组，元组的第一个元素表示在维度上需要后面填充的数量，第二个元素表示在维度上需要在前面填充的数量。
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    list, element -> torch.Tensor, shape = (n_mels, n_frames)
    torch.Tensor, shape = (batch_size, n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """

    # 处理单一音频
    def process_mel_spectrogram(audio):
        if device is not None:
            audio = audio.to(device)
        if padding > 0:
            # 注意这里torch.nn.Functional是追加填充padding个点数，也就是说原来有m点，填充后是padding + m个点
            # audio是一维的，形状为(采样点数,), 填充后是形状为(采样点数 + padding,)
            # 先填充padding再进行计算的原因是为了确保音频信号在进行短时傅里叶变换（STFT）时，能够覆盖到音频信号的末尾部分。
            # 填充后的音频信号长度增加，可以避免由于信号长度不足而导致的频谱信息丢失。
            audio = F.pad(audio, (0, padding)) 
    
        # 定义用于STFT的窗函数
        window = torch.hann_window(N_FFT).to(audio.device)
        # 短时傅里叶变换
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        # 提取STFT的幅度谱，并计算其平方（能量谱）
        # stft[..., :-1] 表示去掉最后一个频率分量（奈奎斯特频率），因为它是冗余的
        # 奈奎斯特频率是采样频率的一半，在实际应用中通常不包含有用的信息，最后一个频率分量可以理解为下一个周期的开始的频率分量
        magnitudes = stft[..., :-1].abs() ** 2

        # 获得mel滤波器组
        filters = mel_filters(audio.device, n_mels)
    
        # 将能量谱通过梅尔滤波器组，得到梅尔频谱
        mel_spec = filters @ magnitudes

        # torch.clamp将最小值限制在1e-10以上，并计算对数mel频谱
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        # 下面的8.0和4.0是经验常数
        # 动态范围压缩：目的是限制频谱的极值，避免某些频率点的能量过高或过低，从而改F善频谱的数值稳定性，使其更适合后续的模型处理。
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        # 归一化
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    # 处理单个文件
    if not isinstance(audio, List):
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio) 

            return [process_mel_spectrogram(audio)]       
    # 处理多个文件
    else:
        audio_list = []
        for path in audio:
            audio_list.append(process_mel_spectrogram(torch.from_numpy(load_audio(path))))
        return audio_list

    
