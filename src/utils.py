from scipy.fft import fft
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import librosa


def load_audio_file(file_path):
    """
    This is a wrapper function around librosa.load which loads audio file given its path.
    :param file_path: File path.
    :return: audio signal and its sampling rate.
    """
    audio_file, sampling_rate = librosa.load(file_path, sr=None, mono=True, offset=0.0)
    return audio_file, sampling_rate


def fft_plot(audio_signal, sampling_rate, title=None):
    """
    This function returns fft plot.
    :param audio_signal: Input audio signal.
    :param sampling_rate: Sampling frequency.
    :param title: Plot title.
    :return: Plot.
    """
    n = len(audio_signal)
    T = 1 / sampling_rate

    yf = fft(audio_signal)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(n / 2))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(xf, 2.0 / n * np.abs(yf[: n // 2]))
    plt.grid()
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    return plt.show()


def wave_plot(audio_signal, sampling_rate, title=None):
    """
    This function returns waveform plot.
    :param audio_signal: Input audio signal.
    :param sampling_rate: Sampling frequency.
    :param title: Plot title.
    :return: Plot.
    """
    plt.figure(figsize=(15, 5))
    display.waveplot(audio_signal, sampling_rate)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    return plt.show()
