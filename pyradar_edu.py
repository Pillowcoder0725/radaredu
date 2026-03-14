import numpy as np

class RadarProcessor:
    def __init__(self, sample_rate, sweep_time, num_samples):
        self.sample_rate = sample_rate
        self.sweep_time = sweep_time
        self.num_samples = num_samples

    def apply_range_fft(self, beat_signal):
        """Applies 1D FFT to extract range information."""
        # Applying windowing to reduce leakage
        window = np.hanning(self.num_samples)
        windowed_signal = beat_signal * window
        
        # Compute Fast Fourier Transform
        fft_result = np.fft.fft(windowed_signal)
        
        # Take the positive half of the frequencies
        half_length = self.num_samples // 2
        range_profile = np.abs(fft_result[:half_length])
        
        return range_profile

    def apply_doppler_fft(self, radar_data_matrix):
        """Applies 2D FFT to extract range and velocity (micro-Doppler)."""
        num_chirps = radar_data_matrix.shape[0]
        
        # 1D FFT along the fast-time (range) axis
        range_fft = np.fft.fft(radar_data_matrix, axis=1)
        
        # 1D FFT along the slow-time (Doppler) axis
        range_doppler = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
