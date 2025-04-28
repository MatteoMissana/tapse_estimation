import numpy as np
from scipy.signal import butter, lfilter, find_peaks

def bandpass_filter(signal, fs, lowcut=5.0, highcut=15.0, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def pan_tompkins_detector(ecg_signal: np.ndarray, fs: float):
    """
    Implements Pan-Tompkins algorithm to detect R-peaks in an ECG signal.
    
    Parameters:
        ecg_signal (np.ndarray): The ECG signal.
        fs (float): Sampling frequency in Hz.
    
    Returns:
        r_peaks (np.ndarray): Detected R-peak indices.
        diastoles (np.ndarray): Approximated diastole indices.
    """
    # 1. Bandpass filter (5–15 Hz)
    filtered = bandpass_filter(ecg_signal, fs, lowcut=5.0, highcut=15.0)

    # 2. Derivative
    derivative = np.diff(filtered)
    derivative = np.append(derivative, 0)

    # 3. Squaring
    squared = derivative ** 2

    # 4. Moving window integration (~150 ms window)
    window_size = int(0.150 * fs)
    mwa = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

    # 5. Peak detection with adaptive thresholding
    distance = int(0.25 * fs)  # ~240 BPM max
    threshold = 0.5 * np.max(mwa)
    peaks, _ = find_peaks(mwa, height=threshold, distance=distance)

    # Optional: adjust peak position to actual R-peak in original ECG
    r_peaks = []
    search_window = int(0.1 * fs)  # +/- 100 ms window
    for peak in peaks:
        start = max(peak - search_window, 0)
        end = min(peak + search_window, len(ecg_signal))
        local_max = np.argmax(ecg_signal[start:end])
        r_peaks.append(start + local_max)
    
    r_peaks = np.array(r_peaks)

    return r_peaks
