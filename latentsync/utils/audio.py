# Adapted from https://github.com/Rudrabha/Wav2Lip/blob/master/audio.py

import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch


class AudioConfig:
    """Audio configuration class that replaces the YAML config"""
    def __init__(self):
        # Audio processing parameters
        self.num_mels = 80  # Number of mel-spectrogram channels and local conditioning dimensionality
        self.rescale = True  # Whether to rescale audio prior to preprocessing
        self.rescaling_max = 0.9  # Rescaling value
        self.n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
        self.hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
        self.win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
        self.sample_rate = 16000  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
        self.frame_shift_ms = None
        self.signal_normalization = True
        self.allow_clipping_in_normalization = True
        self.symmetric_mels = True
        self.max_abs_value = 4.0
        self.preemphasize = True  # whether to apply filter
        self.preemphasis = 0.97  # filter coefficient.
        self.min_level_db = -100
        self.ref_level_db = 20
        self.fmin = 55
        self.fmax = 7600


# Create a global config instance
AUDIO_CONFIG = AudioConfig()


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = AUDIO_CONFIG.hop_size
    if hop_size is None:
        assert AUDIO_CONFIG.frame_shift_ms is not None
        hop_size = int(AUDIO_CONFIG.frame_shift_ms / 1000 * AUDIO_CONFIG.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, AUDIO_CONFIG.preemphasis, AUDIO_CONFIG.preemphasize))
    S = _amp_to_db(np.abs(D)) - AUDIO_CONFIG.ref_level_db

    if AUDIO_CONFIG.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, AUDIO_CONFIG.preemphasis, AUDIO_CONFIG.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - AUDIO_CONFIG.ref_level_db

    if AUDIO_CONFIG.signal_normalization:
        return _normalize(S)
    return S


def _stft(y):
    # 使用librosa进行短时傅里叶变换
    return librosa.stft(y=y, n_fft=AUDIO_CONFIG.n_fft, hop_length=get_hop_size(), win_length=AUDIO_CONFIG.win_size)


##########################################################
# 通用帧数计算函数
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram"""
    pad = fsize - fshift
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding"""
    M = num_frames(len(x), fsize, fshift)
    pad = fsize - fshift
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert AUDIO_CONFIG.fmax <= AUDIO_CONFIG.sample_rate // 2
    return librosa.filters.mel(
        sr=AUDIO_CONFIG.sample_rate,
        n_fft=AUDIO_CONFIG.n_fft,
        n_mels=AUDIO_CONFIG.num_mels,
        fmin=AUDIO_CONFIG.fmin,
        fmax=AUDIO_CONFIG.fmax,
    )


def _amp_to_db(x):
    min_level = np.exp(AUDIO_CONFIG.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if AUDIO_CONFIG.allow_clipping_in_normalization:
        if AUDIO_CONFIG.symmetric_mels:
            return np.clip(
                (2 * AUDIO_CONFIG.max_abs_value) * ((S - AUDIO_CONFIG.min_level_db) / (-AUDIO_CONFIG.min_level_db))
                - AUDIO_CONFIG.max_abs_value,
                -AUDIO_CONFIG.max_abs_value,
                AUDIO_CONFIG.max_abs_value,
            )
        else:
            return np.clip(
                AUDIO_CONFIG.max_abs_value * ((S - AUDIO_CONFIG.min_level_db) / (-AUDIO_CONFIG.min_level_db)),
                0,
                AUDIO_CONFIG.max_abs_value,
            )

    assert S.max() <= 0 and S.min() - AUDIO_CONFIG.min_level_db >= 0
    if AUDIO_CONFIG.symmetric_mels:
        return (2 * AUDIO_CONFIG.max_abs_value) * (
            (S - AUDIO_CONFIG.min_level_db) / (-AUDIO_CONFIG.min_level_db)
        ) - AUDIO_CONFIG.max_abs_value
    else:
        return AUDIO_CONFIG.max_abs_value * ((S - AUDIO_CONFIG.min_level_db) / (-AUDIO_CONFIG.min_level_db))


def _denormalize(D):
    if AUDIO_CONFIG.allow_clipping_in_normalization:
        if AUDIO_CONFIG.symmetric_mels:
            return (
                (np.clip(D, -AUDIO_CONFIG.max_abs_value, AUDIO_CONFIG.max_abs_value) + AUDIO_CONFIG.max_abs_value)
                * -AUDIO_CONFIG.min_level_db
                / (2 * AUDIO_CONFIG.max_abs_value)
            ) + AUDIO_CONFIG.min_level_db
        else:
            return (
                np.clip(D, 0, AUDIO_CONFIG.max_abs_value) * -AUDIO_CONFIG.min_level_db / AUDIO_CONFIG.max_abs_value
            ) + AUDIO_CONFIG.min_level_db

    if AUDIO_CONFIG.symmetric_mels:
        return (
            (D + AUDIO_CONFIG.max_abs_value) * -AUDIO_CONFIG.min_level_db / (2 * AUDIO_CONFIG.max_abs_value)
        ) + AUDIO_CONFIG.min_level_db
    else:
        return (D * -AUDIO_CONFIG.min_level_db / AUDIO_CONFIG.max_abs_value) + AUDIO_CONFIG.min_level_db


def get_melspec_overlap(audio_samples, melspec_length=52):
    mel_spec_overlap = melspectrogram(audio_samples.numpy())
    mel_spec_overlap = torch.from_numpy(mel_spec_overlap)
    i = 0
    mel_spec_overlap_list = []
    while i + melspec_length < mel_spec_overlap.shape[1] - 3:
        mel_spec_overlap_list.append(mel_spec_overlap[:, i : i + melspec_length].unsqueeze(0))
        i += 3
    mel_spec_overlap = torch.stack(mel_spec_overlap_list)
    return mel_spec_overlap 