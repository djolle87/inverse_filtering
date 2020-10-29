from scipy.signal import lfilter
import librosa


def inverse_filter(audio_signal, filter_order):
    """
    This function returns inverese filtered signal.
    :param audio_signal: Input audio signal.
    :param filter_order: LPC filter order.
    :return: Inverse filtered signal.
    """

    # Find Linear Prediction Filter Coefficients
    a = librosa.lpc(y=audio_signal, order=filter_order)

    # Create inverse filter and apply filtering
    filtered_signal = lfilter(b=a, a=1, x=audio_signal)

    return filtered_signal
