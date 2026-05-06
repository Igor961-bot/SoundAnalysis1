import csv
import math
import os
import sys
import wave
from dataclasses import dataclass


VENDOR_DIR = os.path.join(os.path.dirname(__file__), ".vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import numpy as np


WINDOW_NAME_MAP = {
    "prostokatne": "rectangular",
    "rectangular": "rectangular",
    "rectangle": "rectangular",
    "triangular": "triangular",
    "trojkatne": "triangular",
    "bartlett": "triangular",
    "hamming": "hamming",
    "hann": "hann",
    "hanning": "hann",
    "van hann": "hann",
    "van hanna": "hann",
    "blackman": "blackman",
}


@dataclass
class AudioData:
    path: str
    sample_rate: int
    channels: int
    sample_width: int
    sample_count: int
    duration_seconds: float
    samples: np.ndarray


@dataclass
class FrameFeatures:
    index: int
    start_time: float
    end_time: float
    volume: float
    normalized_volume: float
    ste: float
    zcr: float
    silent_flag: int
    f0_autocorrelation: float
    f0_amdf: float
    dominant_frequency_fft: float
    spectral_centroid: float
    effective_bandwidth: float
    ersb1: float
    ersb2: float
    ersb3: float
    spectral_flatness: float
    spectral_crest: float
    f0_cepstrum: float
    voicing_label: str
    speech_music_label: str


@dataclass
class ClipFeatures:
    mean_volume: float
    vstd: float
    vdr: float
    vu: float
    lster: float
    energy_entropy: float
    zstd: float
    hzcrr: float
    silent_ratio: float
    mean_f0_autocorrelation: float
    mean_f0_amdf: float
    mean_dominant_frequency_fft: float
    mean_spectral_centroid: float
    mean_effective_bandwidth: float
    mean_ersb1: float
    mean_ersb2: float
    mean_ersb3: float
    mean_spectral_flatness: float
    mean_spectral_crest: float
    mean_f0_cepstrum: float
    overall_label: str


@dataclass
class SpectrumSnapshot:
    start_time: float
    duration_seconds: float
    sample_rate: int
    window_name: str
    time_axis: np.ndarray
    raw_samples: np.ndarray
    windowed_samples: np.ndarray
    frequencies: np.ndarray
    raw_magnitude_db: np.ndarray
    windowed_magnitude_db: np.ndarray
    spectral_centroid: float
    effective_bandwidth: float
    band_energies: tuple[float, float, float, float]
    band_ratios: tuple[float, float, float, float]
    spectral_flatness: float
    spectral_crest: float
    cepstrum_quefrencies_ms: np.ndarray
    cepstrum_values: np.ndarray
    f0_cepstrum: float


@dataclass
class SpectrogramData:
    frame_ms: float
    overlap_percent: float
    sample_rate: int
    window_name: str
    times: np.ndarray
    frequencies: np.ndarray
    magnitude_db: np.ndarray


@dataclass
class AnalysisResult:
    audio_data: AudioData
    analysis_sample_rate: int
    downsample_factor: int
    frame_ms: float
    hop_ms: float
    frame_size_samples: int
    hop_size_samples: int
    silence_volume_threshold: float
    silence_zcr_threshold: float
    frames: list[FrameFeatures]
    clip: ClipFeatures
    voicing_segments: list[tuple[float, float, str]]
    speech_music_segments: list[tuple[float, float, str]]


def load_wav_file(path: str) -> AudioData:
    with wave.open(path, "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Obslugiwane sa tylko nieskompresowane pliki WAV PCM.")

        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        sample_count = wav_file.getnframes()
        raw_bytes = wav_file.readframes(sample_count)

    samples = decode_pcm_samples(raw_bytes, sample_width)

    if channels > 1:
        samples = samples.reshape(-1, channels)
        mono_samples = np.mean(samples, axis=1)
    else:
        mono_samples = samples

    mono_samples = mono_samples.astype(np.float64)
    duration_seconds = 0.0
    if sample_rate > 0:
        duration_seconds = len(mono_samples) / sample_rate

    return AudioData(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        sample_count=len(mono_samples),
        duration_seconds=duration_seconds,
        samples=mono_samples,
    )


def decode_pcm_samples(raw_bytes: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64)
        return (data - 128.0) / 128.0

    if sample_width == 2:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
        return data / 32768.0

    if sample_width == 3:
        sample_total = len(raw_bytes) // 3
        values = np.empty(sample_total, dtype=np.int32)
        for index in range(sample_total):
            offset = index * 3
            chunk = raw_bytes[offset:offset + 3]
            sign_byte = b"\xff" if chunk[2] & 0x80 else b"\x00"
            values[index] = int.from_bytes(chunk + sign_byte, byteorder="little", signed=True)
        return values.astype(np.float64) / 8388608.0

    if sample_width == 4:
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float64)
        return data / 2147483648.0

    raise ValueError(f"Nieobslugiwana szerokosc probki WAV: {sample_width} bajty.")


def frame_signal(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> tuple[list[np.ndarray], list[float], list[float], int, int]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, int(sample_rate * hop_ms / 1000.0))

    frames = []
    start_times = []
    end_times = []

    if len(samples) == 0:
        frames.append(np.zeros(frame_size, dtype=np.float64))
        start_times.append(0.0)
        end_times.append(frame_size / sample_rate if sample_rate > 0 else 0.0)
        return frames, start_times, end_times, frame_size, hop_size

    start_index = 0
    while start_index < len(samples):
        end_index = start_index + frame_size
        frame = samples[start_index:end_index]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        frames.append(frame.astype(np.float64))
        start_times.append(start_index / sample_rate)
        end_times.append(min(end_index, len(samples)) / sample_rate)

        if end_index >= len(samples):
            break
        start_index += hop_size

    return frames, start_times, end_times, frame_size, hop_size


def downsample_signal(samples: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or len(samples) == 0:
        return samples.astype(np.float64)

    usable_length = len(samples) - (len(samples) % factor)
    reduced_parts = []

    if usable_length > 0:
        trimmed = samples[:usable_length].reshape(-1, factor)
        reduced_parts.append(np.mean(trimmed, axis=1))

    if usable_length < len(samples):
        reduced_parts.append(np.array([float(np.mean(samples[usable_length:]))], dtype=np.float64))

    if not reduced_parts:
        return samples.astype(np.float64)

    return np.concatenate(reduced_parts).astype(np.float64)


def prepare_analysis_signal(
    audio_data: AudioData,
    max_sample_rate: int = 16000,
) -> tuple[np.ndarray, int, int]:
    analysis_sample_rate = audio_data.sample_rate
    downsample_factor = 1
    analysis_samples = audio_data.samples

    if audio_data.sample_rate > max_sample_rate:
        downsample_factor = math.ceil(audio_data.sample_rate / max_sample_rate)
        analysis_sample_rate = int(round(audio_data.sample_rate / downsample_factor))
        analysis_samples = downsample_signal(audio_data.samples, downsample_factor)

    return analysis_samples.astype(np.float64), analysis_sample_rate, downsample_factor


def sign_value(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def mean_value(values: list[float]) -> float:
    if not values:
        return 0.0

    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def std_dev(values: list[float]) -> float:
    if not values:
        return 0.0

    average = mean_value(values)
    variance_sum = 0.0
    for value in values:
        difference = value - average
        variance_sum += difference * difference

    variance = variance_sum / len(values)
    return math.sqrt(variance)


def normalize_window_name(name: str | None) -> str:
    if not name:
        return "rectangular"

    normalized = name.strip().lower()
    return WINDOW_NAME_MAP.get(normalized, normalized)


def create_window(name: str | None, size: int) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=np.float64)

    normalized = normalize_window_name(name)
    if normalized == "rectangular":
        return np.ones(size, dtype=np.float64)
    if normalized == "triangular":
        return np.bartlett(size).astype(np.float64)
    if normalized == "hamming":
        return np.hamming(size).astype(np.float64)
    if normalized == "hann":
        return np.hanning(size).astype(np.float64)
    if normalized == "blackman":
        return np.blackman(size).astype(np.float64)
    return np.ones(size, dtype=np.float64)


def calculate_volume(frame: np.ndarray) -> float:
    if len(frame) == 0:
        return 0.0

    energy_sum = 0.0
    for sample in frame:
        value = float(sample)
        energy_sum += value * value

    return math.sqrt(energy_sum / len(frame))


def calculate_ste(frame: np.ndarray) -> float:
    if len(frame) == 0:
        return 0.0

    energy_sum = 0.0
    for sample in frame:
        value = float(sample)
        energy_sum += value * value

    return energy_sum / len(frame)


def calculate_zcr(frame: np.ndarray) -> float:
    if len(frame) < 2:
        return 0.0

    zero_crossings = 0.0
    previous_sign = sign_value(float(frame[0]))

    for index in range(1, len(frame)):
        current_sign = sign_value(float(frame[index]))
        zero_crossings += abs(current_sign - previous_sign)
        previous_sign = current_sign

    return zero_crossings / (2.0 * len(frame))


def calculate_autocorrelation_f0(
    frame: np.ndarray,
    sample_rate: int,
    min_frequency: float = 50.0,
    max_frequency: float = 500.0,
) -> float:
    if len(frame) < 2 or sample_rate <= 0:
        return 0.0

    centered = frame - float(np.mean(frame))
    reference_energy = float(np.dot(centered, centered))
    if reference_energy <= 1e-12:
        return 0.0

    min_lag = max(1, int(sample_rate / max_frequency))
    max_lag = min(len(centered) - 1, int(sample_rate / min_frequency))
    if min_lag >= max_lag:
        return 0.0

    best_lag = 0
    best_value = -1e30

    for lag in range(min_lag, max_lag + 1):
        value = float(np.dot(centered[:-lag], centered[lag:]))
        if value > best_value:
            best_value = value
            best_lag = lag

    normalized_peak = best_value / reference_energy
    if normalized_peak < 0.30 or best_lag == 0:
        return 0.0

    return sample_rate / best_lag


def calculate_amdf_f0(
    frame: np.ndarray,
    sample_rate: int,
    min_frequency: float = 50.0,
    max_frequency: float = 500.0,
) -> float:
    if len(frame) < 2 or sample_rate <= 0:
        return 0.0

    centered = frame - float(np.mean(frame))
    average_absolute_value = float(np.mean(np.abs(centered)))
    if average_absolute_value <= 1e-12:
        return 0.0

    min_lag = max(1, int(sample_rate / max_frequency))
    max_lag = min(len(centered) - 1, int(sample_rate / min_frequency))
    if min_lag >= max_lag:
        return 0.0

    lag_values = []
    for lag in range(min_lag, max_lag + 1):
        difference = float(np.mean(np.abs(centered[lag:] - centered[:-lag])))
        lag_values.append((lag, difference))

    if not lag_values:
        return 0.0

    best_lag = 0
    best_value = min(value for _, value in lag_values)

    if len(lag_values) >= 3:
        average_difference = mean_value([value for _, value in lag_values])
        for index in range(1, len(lag_values) - 1):
            previous_value = lag_values[index - 1][1]
            current_lag, current_value = lag_values[index]
            next_value = lag_values[index + 1][1]

            is_local_minimum = current_value <= previous_value and current_value < next_value
            if is_local_minimum and current_value < average_difference * 0.85:
                best_lag = current_lag
                best_value = current_value
                break

    if best_lag == 0:
        accepted_limit = best_value * 1.10
        for lag, value in lag_values:
            if value <= accepted_limit:
                best_lag = lag
                best_value = value
                break

    normalized_difference = best_value / average_absolute_value
    if normalized_difference > 1.10 or best_lag == 0:
        return 0.0

    return sample_rate / best_lag


def calculate_fft_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    window_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(signal) == 0 or sample_rate <= 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    samples = signal.astype(np.float64)
    window = create_window(window_name, len(samples))
    windowed = samples * window
    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)
    power = magnitudes * magnitudes
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return windowed, frequencies, magnitudes, power


def calculate_dominant_frequency_from_spectrum(frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
    if len(magnitudes) <= 1 or len(frequencies) != len(magnitudes):
        return 0.0

    magnitudes_copy = magnitudes.copy()
    magnitudes_copy[0] = 0.0
    best_index = int(np.argmax(magnitudes_copy))
    return float(frequencies[best_index])


def calculate_dominant_frequency_fft(frame: np.ndarray, sample_rate: int, window_name: str = "hamming") -> float:
    _windowed, frequencies, magnitudes, _power = calculate_fft_spectrum(frame, sample_rate, window_name)
    return calculate_dominant_frequency_from_spectrum(frequencies, magnitudes)


def calculate_spectral_centroid(frequencies: np.ndarray, power: np.ndarray) -> float:
    if len(frequencies) == 0 or len(frequencies) != len(power):
        return 0.0

    total_power = float(np.sum(power))
    if total_power <= 1e-12:
        return 0.0

    return float(np.sum(frequencies * power) / total_power)


def calculate_effective_bandwidth(frequencies: np.ndarray, power: np.ndarray, centroid: float) -> float:
    if len(frequencies) == 0 or len(frequencies) != len(power):
        return 0.0

    total_power = float(np.sum(power))
    if total_power <= 1e-12:
        return 0.0

    deviations = (frequencies - centroid) ** 2
    return float(math.sqrt(float(np.sum(deviations * power) / total_power)))


def calculate_band_energy_ratios(
    frequencies: np.ndarray,
    power: np.ndarray,
    sample_rate: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    if len(frequencies) == 0 or len(frequencies) != len(power) or sample_rate <= 0:
        zeros = (0.0, 0.0, 0.0, 0.0)
        return zeros, zeros

    nyquist = sample_rate / 2.0
    band_limits = [0.0, 630.0, 1720.0, 4400.0, min(11025.0, nyquist)]
    band_energies = []

    for band_index in range(4):
        low = band_limits[band_index]
        high = band_limits[band_index + 1]
        if high <= low:
            band_energies.append(0.0)
            continue

        if band_index == 3:
            mask = (frequencies >= low) & (frequencies <= high)
        else:
            mask = (frequencies >= low) & (frequencies < high)
        band_energies.append(float(np.sum(power[mask])))

    total_energy = float(np.sum(power))
    if total_energy <= 1e-12:
        band_ratios = (0.0, 0.0, 0.0, 0.0)
    else:
        band_ratios = tuple(energy / total_energy for energy in band_energies)

    return tuple(band_energies), band_ratios


def calculate_spectral_flatness(power: np.ndarray) -> float:
    if len(power) == 0:
        return 1.0

    arithmetic_mean = float(np.mean(power))
    if arithmetic_mean <= 1e-12:
        return 1.0

    positive_power = np.maximum(power, 1e-12)
    geometric_mean = float(np.exp(np.mean(np.log(positive_power))))
    return geometric_mean / arithmetic_mean


def calculate_spectral_crest(power: np.ndarray) -> float:
    if len(power) == 0:
        return 0.0

    arithmetic_mean = float(np.mean(power))
    if arithmetic_mean <= 1e-12:
        return 0.0

    return float(np.max(power) / arithmetic_mean)


def calculate_real_cepstrum(
    frame: np.ndarray,
    sample_rate: int,
    window_name: str = "hamming",
) -> tuple[np.ndarray, np.ndarray]:
    if len(frame) == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    centered_frame = frame.astype(np.float64) - float(np.mean(frame))
    if float(np.mean(centered_frame * centered_frame)) <= 1e-12:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    window = create_window(window_name, len(centered_frame))
    windowed_frame = centered_frame * window
    spectrum = np.fft.fft(windowed_frame)
    magnitudes = np.abs(spectrum)
    if len(magnitudes) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    log_magnitude = np.log(np.maximum(magnitudes, 1e-12))
    cepstrum = np.abs(np.fft.ifft(log_magnitude).real)
    quefrencies = np.arange(len(cepstrum), dtype=np.float64) / sample_rate
    return quefrencies, cepstrum


def detect_cepstrum_f0_from_curve(
    quefrencies: np.ndarray,
    cepstrum: np.ndarray,
    min_frequency: float = 50.0,
    max_frequency: float = 400.0,
    reference_frequency: float | None = None,
) -> float:
    if len(quefrencies) == 0 or len(cepstrum) == 0 or len(quefrencies) != len(cepstrum):
        return 0.0

    min_quefrency = 1.0 / max_frequency
    max_quefrency = 1.0 / min_frequency
    mask = (quefrencies >= min_quefrency) & (quefrencies <= max_quefrency)

    if not np.any(mask):
        return 0.0

    search_values = cepstrum[mask]
    search_quefrencies = quefrencies[mask]
    if len(search_values) < 3:
        return 0.0

    candidate_peaks = []
    for index in range(1, len(search_values) - 1):
        current_value = float(search_values[index])
        previous_value = float(search_values[index - 1])
        next_value = float(search_values[index + 1])
        if current_value > previous_value and current_value >= next_value:
            prominence = current_value - max(previous_value, next_value)
            frequency = 1.0 / float(search_quefrencies[index])
            candidate_peaks.append((frequency, current_value, prominence))

    if not candidate_peaks:
        return 0.0

    average_value = float(np.mean(search_values))
    std_value = float(np.std(search_values))

    if reference_frequency is not None and min_frequency <= reference_frequency <= max_frequency:
        nearby_candidates = []
        for frequency, current_value, prominence in candidate_peaks:
            relative_distance = abs(frequency - reference_frequency) / max(reference_frequency, 1e-6)
            if relative_distance <= 0.35:
                nearby_candidates.append((frequency, current_value, prominence))

        if nearby_candidates:
            nearby_candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
            frequency, current_value, _prominence = nearby_candidates[0]
            if current_value >= average_value + (0.20 * std_value):
                return frequency

    candidate_peaks.sort(key=lambda item: (item[2], item[1]), reverse=True)
    frequency, current_value, _prominence = candidate_peaks[0]
    if current_value < average_value + (0.50 * std_value):
        return 0.0

    return frequency


def calculate_cepstrum_f0(
    frame: np.ndarray,
    sample_rate: int,
    min_frequency: float = 50.0,
    max_frequency: float = 400.0,
    window_name: str = "hamming",
    reference_frequency: float | None = None,
) -> float:
    if len(frame) == 0 or sample_rate <= 0:
        return 0.0

    energy = float(np.mean(frame * frame))
    if energy <= 1e-12:
        return 0.0

    quefrencies, cepstrum = calculate_real_cepstrum(frame, sample_rate, window_name)
    if len(cepstrum) == 0:
        return 0.0

    return detect_cepstrum_f0_from_curve(
        quefrencies,
        cepstrum,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        reference_frequency=reference_frequency,
    )


def calculate_frequency_features(
    frame: np.ndarray,
    sample_rate: int,
    cepstrum_reference_frequency: float | None = None,
    cepstrum_min_frequency: float = 50.0,
    cepstrum_max_frequency: float = 400.0,
) -> dict[str, float]:
    _windowed, frequencies, magnitudes, power = calculate_fft_spectrum(frame, sample_rate, "hamming")
    centroid = calculate_spectral_centroid(frequencies, power)
    bandwidth = calculate_effective_bandwidth(frequencies, power, centroid)
    _band_energies, band_ratios = calculate_band_energy_ratios(frequencies, power, sample_rate)

    return {
        "dominant_frequency_fft": calculate_dominant_frequency_from_spectrum(frequencies, magnitudes),
        "spectral_centroid": centroid,
        "effective_bandwidth": bandwidth,
        "ersb1": band_ratios[0],
        "ersb2": band_ratios[1],
        "ersb3": band_ratios[2],
        "spectral_flatness": calculate_spectral_flatness(power),
        "spectral_crest": calculate_spectral_crest(power),
        "f0_cepstrum": calculate_cepstrum_f0(
            frame,
            sample_rate,
            min_frequency=cepstrum_min_frequency,
            max_frequency=cepstrum_max_frequency,
            reference_frequency=cepstrum_reference_frequency,
        ),
    }


def extract_audio_segment(
    samples: np.ndarray,
    sample_rate: int,
    start_time: float = 0.0,
    duration_seconds: float | None = None,
    pad_to_duration: bool = False,
) -> np.ndarray:
    if len(samples) == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float64)

    if duration_seconds is None:
        return samples.astype(np.float64)

    safe_start_time = max(0.0, start_time)
    safe_duration = max(1.0 / sample_rate, duration_seconds)

    start_index = int(round(safe_start_time * sample_rate))
    frame_length = max(1, int(round(safe_duration * sample_rate)))
    start_index = min(start_index, len(samples) - 1)

    end_index = min(len(samples), start_index + frame_length)
    segment = samples[start_index:end_index].astype(np.float64)

    if pad_to_duration and len(segment) < frame_length:
        segment = np.pad(segment, (0, frame_length - len(segment)))

    return segment


def extract_centered_audio_segment(
    samples: np.ndarray,
    sample_rate: int,
    center_time: float,
    duration_seconds: float,
) -> np.ndarray:
    if len(samples) == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float64)

    segment_length = max(1, int(round(duration_seconds * sample_rate)))
    center_index = int(round(center_time * sample_rate))
    start_index = center_index - (segment_length // 2)
    end_index = start_index + segment_length

    left_pad = max(0, -start_index)
    right_pad = max(0, end_index - len(samples))
    clamped_start = max(0, start_index)
    clamped_end = min(len(samples), end_index)

    segment = samples[clamped_start:clamped_end].astype(np.float64)
    if left_pad > 0 or right_pad > 0:
        segment = np.pad(segment, (left_pad, right_pad))

    if len(segment) < segment_length:
        segment = np.pad(segment, (0, segment_length - len(segment)))

    return segment


def build_relative_db(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=np.float64)

    safe_values = np.maximum(values.astype(np.float64), 1e-12)
    max_value = float(np.max(safe_values))
    if max_value <= 1e-12:
        return np.zeros(len(values), dtype=np.float64)

    return 20.0 * np.log10(safe_values / max_value)


def compute_spectrum_snapshot(
    audio_data: AudioData,
    start_time: float = 0.0,
    duration_seconds: float | None = None,
    window_name: str = "hamming",
    cepstrum_reference_frequency: float | None = None,
    cepstrum_min_frequency: float = 50.0,
    cepstrum_max_frequency: float = 400.0,
) -> SpectrumSnapshot:
    if duration_seconds is None:
        segment = audio_data.samples.astype(np.float64)
        actual_start = 0.0
        actual_duration = audio_data.duration_seconds
    else:
        segment = extract_audio_segment(
            audio_data.samples,
            audio_data.sample_rate,
            start_time=start_time,
            duration_seconds=duration_seconds,
            pad_to_duration=True,
        )
        actual_start = max(0.0, start_time)
        actual_duration = max(duration_seconds, 1.0 / max(1, audio_data.sample_rate))

    if len(segment) == 0:
        segment = np.zeros(1, dtype=np.float64)

    time_axis = np.arange(len(segment), dtype=np.float64) / max(1, audio_data.sample_rate)
    raw_windowed, raw_frequencies, raw_magnitudes, raw_power = calculate_fft_spectrum(
        segment,
        audio_data.sample_rate,
        "rectangular",
    )
    windowed_samples, windowed_frequencies, windowed_magnitudes, windowed_power = calculate_fft_spectrum(
        segment,
        audio_data.sample_rate,
        window_name,
    )

    centroid = calculate_spectral_centroid(windowed_frequencies, windowed_power)
    bandwidth = calculate_effective_bandwidth(windowed_frequencies, windowed_power, centroid)
    band_energies, band_ratios = calculate_band_energy_ratios(
        windowed_frequencies,
        windowed_power,
        audio_data.sample_rate,
    )
    quefrencies, cepstrum = calculate_real_cepstrum(segment, audio_data.sample_rate, window_name)

    return SpectrumSnapshot(
        start_time=actual_start,
        duration_seconds=actual_duration,
        sample_rate=audio_data.sample_rate,
        window_name=normalize_window_name(window_name),
        time_axis=time_axis,
        raw_samples=segment,
        windowed_samples=windowed_samples,
        frequencies=windowed_frequencies,
        raw_magnitude_db=build_relative_db(raw_magnitudes),
        windowed_magnitude_db=build_relative_db(windowed_magnitudes),
        spectral_centroid=centroid,
        effective_bandwidth=bandwidth,
        band_energies=band_energies,
        band_ratios=band_ratios,
        spectral_flatness=calculate_spectral_flatness(windowed_power),
        spectral_crest=calculate_spectral_crest(windowed_power),
        cepstrum_quefrencies_ms=quefrencies * 1000.0,
        cepstrum_values=cepstrum,
        f0_cepstrum=calculate_cepstrum_f0(
            segment,
            audio_data.sample_rate,
            min_frequency=cepstrum_min_frequency,
            max_frequency=cepstrum_max_frequency,
            window_name=window_name,
            reference_frequency=cepstrum_reference_frequency,
        ),
    )


def reduce_matrix_mean(matrix: np.ndarray, axis: int, target_size: int) -> np.ndarray:
    current_size = matrix.shape[axis]
    if current_size <= target_size or target_size <= 0:
        return matrix

    index_slices = np.array_split(np.arange(current_size), target_size)
    reduced_parts = []

    for group in index_slices:
        if axis == 0:
            reduced_parts.append(np.mean(matrix[group, :], axis=0))
        else:
            reduced_parts.append(np.mean(matrix[:, group], axis=1))

    if axis == 0:
        return np.vstack(reduced_parts)
    return np.column_stack(reduced_parts)


def reduce_axis_mean(values: np.ndarray, target_size: int) -> np.ndarray:
    if len(values) <= target_size or target_size <= 0:
        return values

    index_slices = np.array_split(np.arange(len(values)), target_size)
    reduced = []
    for group in index_slices:
        reduced.append(float(np.mean(values[group])))
    return np.array(reduced, dtype=np.float64)


def compute_spectrogram(
    audio_data: AudioData,
    frame_ms: float = 40.0,
    overlap_percent: float = 50.0,
    window_name: str = "hann",
    max_frequency_hz: float | None = None,
    max_time_bins: int = 1400,
    max_frequency_bins: int = 256,
) -> SpectrogramData:
    if audio_data.sample_rate <= 0:
        return SpectrogramData(
            frame_ms=frame_ms,
            overlap_percent=overlap_percent,
            sample_rate=audio_data.sample_rate,
            window_name=normalize_window_name(window_name),
            times=np.zeros(0, dtype=np.float64),
            frequencies=np.zeros(0, dtype=np.float64),
            magnitude_db=np.zeros((0, 0), dtype=np.float64),
        )

    frame_size = max(16, int(audio_data.sample_rate * frame_ms / 1000.0))
    overlap_samples = int(frame_size * max(0.0, min(overlap_percent, 95.0)) / 100.0)
    hop_size = max(1, frame_size - overlap_samples)
    window = create_window(window_name, frame_size)

    if len(audio_data.samples) == 0:
        frames = [np.zeros(frame_size, dtype=np.float64)]
        start_indices = [0]
    else:
        frames = []
        start_indices = []
        start_index = 0

        while start_index < len(audio_data.samples):
            end_index = start_index + frame_size
            frame = audio_data.samples[start_index:end_index]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))

            frames.append(frame.astype(np.float64))
            start_indices.append(start_index)

            if end_index >= len(audio_data.samples):
                break
            start_index += hop_size

    spectra = []
    times = []
    for start_index, frame in zip(start_indices, frames):
        windowed = frame * window
        magnitudes = np.abs(np.fft.rfft(windowed))
        spectra.append(magnitudes)
        times.append((start_index + (frame_size / 2.0)) / audio_data.sample_rate)

    magnitude = np.array(spectra, dtype=np.float64).T
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / audio_data.sample_rate)

    if max_frequency_hz is not None:
        valid_mask = frequencies <= max_frequency_hz
        magnitude = magnitude[valid_mask, :]
        frequencies = frequencies[valid_mask]

    magnitude_db = build_relative_db(magnitude.flatten()).reshape(magnitude.shape)

    if magnitude_db.shape[1] > max_time_bins:
        magnitude_db = reduce_matrix_mean(magnitude_db, axis=1, target_size=max_time_bins)
        times = reduce_axis_mean(np.array(times, dtype=np.float64), max_time_bins)
    else:
        times = np.array(times, dtype=np.float64)

    if magnitude_db.shape[0] > max_frequency_bins:
        magnitude_db = reduce_matrix_mean(magnitude_db, axis=0, target_size=max_frequency_bins)
        frequencies = reduce_axis_mean(frequencies, max_frequency_bins)

    return SpectrogramData(
        frame_ms=frame_ms,
        overlap_percent=overlap_percent,
        sample_rate=audio_data.sample_rate,
        window_name=normalize_window_name(window_name),
        times=times,
        frequencies=frequencies,
        magnitude_db=magnitude_db,
    )


def calculate_volume_undulation(volumes: list[float]) -> float:
    if len(volumes) < 2:
        return 0.0

    extrema = [volumes[0]]
    for index in range(1, len(volumes) - 1):
        previous_value = volumes[index - 1]
        current_value = volumes[index]
        next_value = volumes[index + 1]

        is_peak = current_value >= previous_value and current_value > next_value
        is_valley = current_value <= previous_value and current_value < next_value

        if is_peak or is_valley:
            extrema.append(current_value)

    extrema.append(volumes[-1])

    if len(extrema) < 2:
        return 0.0

    difference_sum = 0.0
    for index in range(1, len(extrema)):
        difference_sum += abs(extrema[index] - extrema[index - 1])

    return difference_sum / (len(extrema) - 1)


def calculate_energy_entropy(energies: list[float]) -> float:
    if not energies:
        return 0.0

    total_energy = 0.0
    for energy in energies:
        total_energy += energy

    if total_energy <= 1e-12:
        return 0.0

    entropy = 0.0
    for energy in energies:
        if energy <= 1e-12:
            continue

        probability = energy / total_energy
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_local_means(values: list[float], window_size: int) -> list[float]:
    if not values:
        return []

    if window_size <= 1:
        return [float(value) for value in values]

    half_window = window_size // 2
    local_means = []

    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        local_means.append(mean_value(values[start:end]))

    return local_means


def calculate_local_ratios(flags: list[int], window_size: int) -> list[float]:
    if not flags:
        return []

    if window_size <= 1:
        return [float(flag) for flag in flags]

    half_window = window_size // 2
    ratios = []

    for index in range(len(flags)):
        start = max(0, index - half_window)
        end = min(len(flags), index + half_window + 1)
        window = flags[start:end]
        ratios.append(sum(window) / len(window))

    return ratios


def build_segments(frame_times: list[float], frame_duration: float, labels: list[str]) -> list[tuple[float, float, str]]:
    if not labels:
        return []

    segments = []
    current_label = labels[0]
    segment_start = frame_times[0]

    for index in range(1, len(labels)):
        if labels[index] != current_label:
            segment_end = frame_times[index - 1] + frame_duration
            segments.append((segment_start, segment_end, current_label))
            current_label = labels[index]
            segment_start = frame_times[index]

    last_end = frame_times[-1] + frame_duration
    segments.append((segment_start, last_end, current_label))
    return segments


def remove_short_non_silence_runs(labels: list[str], max_run_length: int) -> list[str]:
    if not labels or max_run_length <= 0:
        return labels

    cleaned_labels = labels[:]
    start_index = 0

    while start_index < len(labels):
        end_index = start_index + 1
        while end_index < len(labels) and labels[end_index] == labels[start_index]:
            end_index += 1

        run_label = labels[start_index]
        run_length = end_index - start_index
        left_label = labels[start_index - 1] if start_index > 0 else None
        right_label = labels[end_index] if end_index < len(labels) else None

        if (
            run_label != "silence"
            and run_length <= max_run_length
            and left_label == "silence"
            and right_label == "silence"
        ):
            for index in range(start_index, end_index):
                cleaned_labels[index] = "silence"

        start_index = end_index

    return cleaned_labels


def merge_short_middle_runs(labels: list[str], max_run_length: int) -> list[str]:
    if not labels or max_run_length <= 0:
        return labels

    cleaned_labels = labels[:]
    start_index = 0

    while start_index < len(cleaned_labels):
        end_index = start_index + 1
        while end_index < len(cleaned_labels) and cleaned_labels[end_index] == cleaned_labels[start_index]:
            end_index += 1

        run_length = end_index - start_index
        left_label = cleaned_labels[start_index - 1] if start_index > 0 else None
        right_label = cleaned_labels[end_index] if end_index < len(cleaned_labels) else None

        if run_length <= max_run_length and left_label is not None and left_label == right_label:
            for index in range(start_index, end_index):
                cleaned_labels[index] = left_label

        start_index = end_index

    return cleaned_labels


def merge_short_middle_non_silence_runs(labels: list[str], max_run_length: int) -> list[str]:
    if not labels or max_run_length <= 0:
        return labels

    cleaned_labels = labels[:]
    start_index = 0

    while start_index < len(cleaned_labels):
        end_index = start_index + 1
        while end_index < len(cleaned_labels) and cleaned_labels[end_index] == cleaned_labels[start_index]:
            end_index += 1

        run_label = cleaned_labels[start_index]
        run_length = end_index - start_index
        left_label = cleaned_labels[start_index - 1] if start_index > 0 else None
        right_label = cleaned_labels[end_index] if end_index < len(cleaned_labels) else None

        if (
            run_label != "silence"
            and run_length <= max_run_length
            and left_label is not None
            and left_label == right_label
            and left_label != "silence"
        ):
            for index in range(start_index, end_index):
                cleaned_labels[index] = left_label

        start_index = end_index

    return cleaned_labels


def smooth_frequency_track(values: list[float], active_labels: list[str], active_label: str = "voiced", radius: int = 2) -> list[float]:
    if not values:
        return []

    smoothed_values = list(values)
    for index, value in enumerate(values):
        if value <= 0.0 or active_labels[index] != active_label:
            continue

        neighborhood = []
        for neighbor_index in range(max(0, index - radius), min(len(values), index + radius + 1)):
            neighbor_value = values[neighbor_index]
            if neighbor_value > 0.0 and active_labels[neighbor_index] == active_label:
                neighborhood.append(neighbor_value)

        if len(neighborhood) < 3:
            continue

        sorted_neighborhood = sorted(neighborhood)
        median_value = sorted_neighborhood[len(sorted_neighborhood) // 2]
        relative_difference = abs(value - median_value) / max(median_value, 1e-6)
        if relative_difference > 0.35:
            smoothed_values[index] = median_value

    return smoothed_values


def choose_overall_label(labels: list[str], clip_hzcrr: float = 0.0) -> str:
    speech_count = 0
    music_count = 0

    for label in labels:
        if label == "speech":
            speech_count += 1
        elif label == "music":
            music_count += 1

    if speech_count == 0 and music_count == 0:
        return "silence"

    if speech_count > 0 and music_count > 0:
        smaller = min(speech_count, music_count)
        larger = max(speech_count, music_count)
        if larger > 0 and (smaller / larger) >= 0.60:
            if clip_hzcrr < 0.12 and music_count >= speech_count * 0.60:
                return "music"
            return "mixed"
        if clip_hzcrr < 0.12 and music_count >= speech_count * 0.60:
            return "music"

    if speech_count >= music_count:
        return "speech"
    return "music"


def analyze_audio(audio_data: AudioData, frame_ms: float = 20.0, hop_ms: float = 10.0) -> AnalysisResult:
    analysis_samples, analysis_sample_rate, downsample_factor = prepare_analysis_signal(audio_data)

    frames, start_times, end_times, frame_size, hop_size = frame_signal(
        analysis_samples,
        analysis_sample_rate,
        frame_ms,
        hop_ms,
    )

    volumes = []
    ste_values = []
    zcr_values = []
    f0_autocorrelation_values = []
    f0_amdf_values = []
    dominant_frequencies = []
    spectral_centroids = []
    effective_bandwidths = []
    ersb1_values = []
    ersb2_values = []
    ersb3_values = []
    spectral_flatness_values = []
    spectral_crest_values = []
    f0_cepstrum_values = []
    cepstrum_window_seconds = max(0.04, frame_ms / 1000.0)

    for frame_index, frame in enumerate(frames):
        volume = calculate_volume(frame)
        ste = calculate_ste(frame)
        zcr = calculate_zcr(frame)
        f0_autocorrelation = calculate_autocorrelation_f0(frame, analysis_sample_rate)
        f0_amdf = calculate_amdf_f0(frame, analysis_sample_rate)
        cepstrum_reference_candidates = []
        if 70.0 <= f0_autocorrelation <= 350.0:
            cepstrum_reference_candidates.append(f0_autocorrelation)
        if 70.0 <= f0_amdf <= 350.0:
            cepstrum_reference_candidates.append(f0_amdf)

        cepstrum_reference_frequency = None
        if cepstrum_reference_candidates:
            cepstrum_reference_frequency = mean_value(cepstrum_reference_candidates)

        frame_center_time = (start_times[frame_index] + end_times[frame_index]) * 0.5
        cepstrum_frame = extract_centered_audio_segment(
            analysis_samples,
            analysis_sample_rate,
            frame_center_time,
            cepstrum_window_seconds,
        )
        frequency_features = calculate_frequency_features(
            frame,
            analysis_sample_rate,
            cepstrum_reference_frequency=cepstrum_reference_frequency,
            cepstrum_min_frequency=70.0,
            cepstrum_max_frequency=350.0,
        )
        frequency_features["f0_cepstrum"] = calculate_cepstrum_f0(
            cepstrum_frame,
            analysis_sample_rate,
            min_frequency=70.0,
            max_frequency=350.0,
            reference_frequency=cepstrum_reference_frequency,
        )

        volumes.append(volume)
        ste_values.append(ste)
        zcr_values.append(zcr)
        f0_autocorrelation_values.append(f0_autocorrelation)
        f0_amdf_values.append(f0_amdf)
        dominant_frequencies.append(frequency_features["dominant_frequency_fft"])
        spectral_centroids.append(frequency_features["spectral_centroid"])
        effective_bandwidths.append(frequency_features["effective_bandwidth"])
        ersb1_values.append(frequency_features["ersb1"])
        ersb2_values.append(frequency_features["ersb2"])
        ersb3_values.append(frequency_features["ersb3"])
        spectral_flatness_values.append(frequency_features["spectral_flatness"])
        spectral_crest_values.append(frequency_features["spectral_crest"])
        f0_cepstrum_values.append(frequency_features["f0_cepstrum"])

    max_volume = max(volumes) if volumes else 0.0
    normalized_volumes = []
    for volume in volumes:
        if max_volume > 1e-12:
            normalized_volumes.append(volume / max_volume)
        else:
            normalized_volumes.append(0.0)

    mean_normalized_volume = mean_value(normalized_volumes)
    mean_zcr = mean_value(zcr_values)

    silence_volume_threshold = max(0.03, min(0.12, mean_normalized_volume * 0.50))
    silence_zcr_threshold = max(0.02, min(0.10, mean_zcr * 0.80))
    hard_silence_volume_threshold = silence_volume_threshold * 0.50

    silent_flags = []
    for index in range(len(frames)):
        is_silent = (
            normalized_volumes[index] < hard_silence_volume_threshold
            or (
                normalized_volumes[index] < silence_volume_threshold
                and zcr_values[index] < silence_zcr_threshold
            )
        )
        silent_flags.append(1 if is_silent else 0)

    voicing_labels = []
    for index in range(len(frames)):
        if silent_flags[index] == 1:
            voicing_labels.append("silence")
            continue

        valid_f0_autocorrelation = 70.0 <= f0_autocorrelation_values[index] <= 350.0
        valid_f0_amdf = 70.0 <= f0_amdf_values[index] <= 350.0
        valid_f0_cepstrum = 70.0 <= f0_cepstrum_values[index] <= 350.0
        has_valid_f0 = valid_f0_autocorrelation or valid_f0_amdf or valid_f0_cepstrum
        strong_low_zcr_condition = (
            normalized_volumes[index] > max(silence_volume_threshold * 3.0, 0.10)
            and zcr_values[index] < 0.05
        )

        voiced_condition = (
            normalized_volumes[index] > max(silence_volume_threshold * 1.2, 0.035)
            and zcr_values[index] < 0.13
            and (has_valid_f0 or strong_low_zcr_condition)
        )

        if voiced_condition:
            voicing_labels.append("voiced")
        else:
            voicing_labels.append("unvoiced")

    voicing_labels = merge_short_middle_runs(voicing_labels, max_run_length=2)
    voicing_labels = merge_short_middle_runs(voicing_labels, max_run_length=2)
    voicing_labels = remove_short_non_silence_runs(voicing_labels, max_run_length=2)

    for index in range(len(frames)):
        valid_f0_autocorrelation = 70.0 <= f0_autocorrelation_values[index] <= 350.0
        valid_f0_amdf = 70.0 <= f0_amdf_values[index] <= 350.0
        valid_f0_cepstrum = 70.0 <= f0_cepstrum_values[index] <= 350.0

        if voicing_labels[index] != "voiced":
            f0_autocorrelation_values[index] = 0.0
            f0_amdf_values[index] = 0.0
            f0_cepstrum_values[index] = 0.0
            continue

        if not valid_f0_autocorrelation:
            f0_autocorrelation_values[index] = 0.0
        if not valid_f0_amdf:
            f0_amdf_values[index] = 0.0
        if not valid_f0_cepstrum:
            f0_cepstrum_values[index] = 0.0

    f0_cepstrum_values = smooth_frequency_track(f0_cepstrum_values, voicing_labels, active_label="voiced", radius=2)

    frame_rate = 1
    if hop_size > 0:
        frame_rate = max(1, int(round(analysis_sample_rate / hop_size)))

    local_ste_means = calculate_local_means(ste_values, frame_rate)
    local_zcr_means = calculate_local_means(zcr_values, frame_rate)
    local_volume_means = calculate_local_means(normalized_volumes, frame_rate)
    local_silent_ratios = calculate_local_ratios(silent_flags, frame_rate)
    voiced_flags = []
    unvoiced_flags = []

    for label in voicing_labels:
        voiced_flags.append(1 if label == "voiced" else 0)
        unvoiced_flags.append(1 if label == "unvoiced" else 0)

    local_voiced_ratios = calculate_local_ratios(voiced_flags, frame_rate)
    local_unvoiced_ratios = calculate_local_ratios(unvoiced_flags, frame_rate)

    low_ste_flags = []
    high_zcr_flags = []
    for index in range(len(frames)):
        low_ste = 0
        high_zcr = 0

        if ste_values[index] < 0.5 * local_ste_means[index]:
            low_ste = 1

        if zcr_values[index] > 1.5 * local_zcr_means[index]:
            high_zcr = 1

        low_ste_flags.append(low_ste)
        high_zcr_flags.append(high_zcr)

    local_lster_ratios = calculate_local_ratios(low_ste_flags, frame_rate)
    local_hzcrr_ratios = calculate_local_ratios(high_zcr_flags, frame_rate)

    speech_music_labels = []
    for index in range(len(frames)):
        if silent_flags[index] == 1:
            speech_music_labels.append("silence")
            continue

        local_vstd = 0.0
        start = max(0, index - (frame_rate // 2))
        end = min(len(normalized_volumes), index + (frame_rate // 2) + 1)
        local_window_volumes = normalized_volumes[start:end]
        if local_window_volumes:
            local_max_volume = max(local_window_volumes)
            if local_max_volume > 1e-12:
                local_vstd = std_dev(local_window_volumes) / local_max_volume

        tonal_music_condition = (
            voicing_labels[index] == "voiced"
            and local_voiced_ratios[index] > 0.45
            and local_unvoiced_ratios[index] < 0.12
            and zcr_values[index] < max(0.06, mean_zcr * 0.85)
            and local_hzcrr_ratios[index] < 0.18
            and local_volume_means[index] > max(silence_volume_threshold * 1.4, 0.05)
            and spectral_flatness_values[index] < 0.50
        )

        speech_condition = (
            voicing_labels[index] == "unvoiced"
            or local_unvoiced_ratios[index] > 0.16
            or local_hzcrr_ratios[index] > 0.18
            or (local_silent_ratios[index] > 0.18 and local_unvoiced_ratios[index] > 0.08)
            or (
                voicing_labels[index] == "voiced"
                and zcr_values[index] > max(0.085, mean_zcr)
                and local_voiced_ratios[index] < 0.45
            )
            or (local_lster_ratios[index] > 0.40 and local_voiced_ratios[index] < 0.45)
            or spectral_flatness_values[index] > 0.65
        )

        music_condition = (
            tonal_music_condition
            or (
                voicing_labels[index] == "voiced"
                and local_voiced_ratios[index] > 0.55
                and local_unvoiced_ratios[index] < 0.10
                and local_vstd < 0.18
                and local_hzcrr_ratios[index] < 0.20
                and local_volume_means[index] > max(silence_volume_threshold * 1.2, 0.05)
                and spectral_flatness_values[index] < 0.55
            )
        )

        if music_condition and not speech_condition:
            speech_music_labels.append("music")
        elif speech_condition and not music_condition:
            speech_music_labels.append("speech")
        elif music_condition:
            speech_music_labels.append("music")
        elif voicing_labels[index] == "unvoiced":
            speech_music_labels.append("speech")
        elif local_lster_ratios[index] >= local_hzcrr_ratios[index]:
            speech_music_labels.append("speech")
        else:
            speech_music_labels.append("music")

    speech_music_labels = merge_short_middle_non_silence_runs(speech_music_labels, max_run_length=3)
    speech_music_labels = remove_short_non_silence_runs(speech_music_labels, max_run_length=4)

    frame_items = []
    for index in range(len(frames)):
        frame_items.append(
            FrameFeatures(
                index=index,
                start_time=start_times[index],
                end_time=end_times[index],
                volume=volumes[index],
                normalized_volume=normalized_volumes[index],
                ste=ste_values[index],
                zcr=zcr_values[index],
                silent_flag=silent_flags[index],
                f0_autocorrelation=f0_autocorrelation_values[index],
                f0_amdf=f0_amdf_values[index],
                dominant_frequency_fft=dominant_frequencies[index],
                spectral_centroid=spectral_centroids[index],
                effective_bandwidth=effective_bandwidths[index],
                ersb1=ersb1_values[index],
                ersb2=ersb2_values[index],
                ersb3=ersb3_values[index],
                spectral_flatness=spectral_flatness_values[index],
                spectral_crest=spectral_crest_values[index],
                f0_cepstrum=f0_cepstrum_values[index],
                voicing_label=voicing_labels[index],
                speech_music_label=speech_music_labels[index],
            )
        )

    clip_mean_volume = mean_value(normalized_volumes)
    clip_vstd = 0.0
    if normalized_volumes:
        clip_max_volume = max(normalized_volumes)
        if clip_max_volume > 1e-12:
            clip_vstd = std_dev(normalized_volumes) / clip_max_volume

    clip_vdr = 0.0
    if normalized_volumes:
        clip_max_volume = max(normalized_volumes)
        clip_min_volume = min(normalized_volumes)
        if clip_max_volume > 1e-12:
            clip_vdr = (clip_max_volume - clip_min_volume) / clip_max_volume

    clip_vu = calculate_volume_undulation(normalized_volumes)
    clip_lster = mean_value([float(flag) for flag in low_ste_flags])
    clip_energy_entropy = calculate_energy_entropy(ste_values)
    clip_zstd = std_dev(zcr_values)
    clip_hzcrr = mean_value([float(flag) for flag in high_zcr_flags])
    clip_silent_ratio = mean_value([float(flag) for flag in silent_flags])

    non_zero_f0_autocorrelation = [value for value in f0_autocorrelation_values if value > 0.0]
    non_zero_f0_amdf = [value for value in f0_amdf_values if value > 0.0]
    non_zero_fft_frequencies = [value for value in dominant_frequencies if value > 0.0]
    non_zero_f0_cepstrum = [value for value in f0_cepstrum_values if value > 0.0]
    non_silent_indices = [index for index, flag in enumerate(silent_flags) if flag == 0]

    if non_silent_indices:
        clip_spectral_centroid = mean_value([spectral_centroids[index] for index in non_silent_indices])
        clip_effective_bandwidth = mean_value([effective_bandwidths[index] for index in non_silent_indices])
        clip_ersb1 = mean_value([ersb1_values[index] for index in non_silent_indices])
        clip_ersb2 = mean_value([ersb2_values[index] for index in non_silent_indices])
        clip_ersb3 = mean_value([ersb3_values[index] for index in non_silent_indices])
        clip_spectral_flatness = mean_value([spectral_flatness_values[index] for index in non_silent_indices])
        clip_spectral_crest = mean_value([spectral_crest_values[index] for index in non_silent_indices])
    else:
        clip_spectral_centroid = 0.0
        clip_effective_bandwidth = 0.0
        clip_ersb1 = 0.0
        clip_ersb2 = 0.0
        clip_ersb3 = 0.0
        clip_spectral_flatness = 0.0
        clip_spectral_crest = 0.0

    clip_features = ClipFeatures(
        mean_volume=clip_mean_volume,
        vstd=clip_vstd,
        vdr=clip_vdr,
        vu=clip_vu,
        lster=clip_lster,
        energy_entropy=clip_energy_entropy,
        zstd=clip_zstd,
        hzcrr=clip_hzcrr,
        silent_ratio=clip_silent_ratio,
        mean_f0_autocorrelation=mean_value(non_zero_f0_autocorrelation),
        mean_f0_amdf=mean_value(non_zero_f0_amdf),
        mean_dominant_frequency_fft=mean_value(non_zero_fft_frequencies),
        mean_spectral_centroid=clip_spectral_centroid,
        mean_effective_bandwidth=clip_effective_bandwidth,
        mean_ersb1=clip_ersb1,
        mean_ersb2=clip_ersb2,
        mean_ersb3=clip_ersb3,
        mean_spectral_flatness=clip_spectral_flatness,
        mean_spectral_crest=clip_spectral_crest,
        mean_f0_cepstrum=mean_value(non_zero_f0_cepstrum),
        overall_label=choose_overall_label(speech_music_labels, clip_hzcrr),
    )

    frame_duration = 0.0
    if analysis_sample_rate > 0:
        frame_duration = frame_size / analysis_sample_rate

    voicing_segments = build_segments(start_times, frame_duration, voicing_labels)
    speech_music_segments = build_segments(start_times, frame_duration, speech_music_labels)

    return AnalysisResult(
        audio_data=audio_data,
        analysis_sample_rate=analysis_sample_rate,
        downsample_factor=downsample_factor,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        frame_size_samples=frame_size,
        hop_size_samples=hop_size,
        silence_volume_threshold=silence_volume_threshold,
        silence_zcr_threshold=silence_zcr_threshold,
        frames=frame_items,
        clip=clip_features,
        voicing_segments=voicing_segments,
        speech_music_segments=speech_music_segments,
    )


def build_summary_lines(result: AnalysisResult) -> list[str]:
    lines = []
    audio = result.audio_data
    clip = result.clip
    frames = result.frames

    voiced_frames = sum(1 for frame in frames if frame.voicing_label == "voiced")
    unvoiced_frames = sum(1 for frame in frames if frame.voicing_label == "unvoiced")
    silent_frames = sum(1 for frame in frames if frame.voicing_label == "silence")
    speech_frames = sum(1 for frame in frames if frame.speech_music_label == "speech")
    music_frames = sum(1 for frame in frames if frame.speech_music_label == "music")

    lines.extend(
        [
            "Podsumowanie klipu",
            "",
            f"Plik: {audio.path}",
            f"Czestotliwosc probkowania: {audio.sample_rate} Hz",
            f"Czestotliwosc analizy: {result.analysis_sample_rate} Hz",
            f"Downsample factor: {result.downsample_factor}",
            f"Liczba kanalow: {audio.channels}",
            f"Dlugosc: {audio.duration_seconds:.3f} s",
            f"Frame/Hop: {result.frame_ms:.2f} ms / {result.hop_ms:.2f} ms",
            f"Liczba ramek: {len(frames)}",
            "",
            "Progi ciszy:",
            f"- volume_norm < {result.silence_volume_threshold:.4f}",
            f"- zcr < {result.silence_zcr_threshold:.4f}",
            "",
            "Cechy clip-level w dziedzinie czasu:",
            f"- Mean Volume: {clip.mean_volume:.6f}",
            f"- VSTD: {clip.vstd:.6f}",
            f"- VDR: {clip.vdr:.6f}",
            f"- VU: {clip.vu:.6f}",
            f"- LSTER: {clip.lster:.6f}",
            f"- Energy Entropy: {clip.energy_entropy:.6f}",
            f"- ZSTD: {clip.zstd:.6f}",
            f"- HZCRR: {clip.hzcrr:.6f}",
            f"- Silent Ratio: {clip.silent_ratio:.6f}",
            "",
            "Cechy clip-level w dziedzinie czestotliwosci:",
            f"- Mean F0 (autokorelacja): {clip.mean_f0_autocorrelation:.3f} Hz",
            f"- Mean F0 (AMDF): {clip.mean_f0_amdf:.3f} Hz",
            f"- Mean F0 (cepstrum): {clip.mean_f0_cepstrum:.3f} Hz",
            f"- Mean dominant FFT frequency: {clip.mean_dominant_frequency_fft:.3f} Hz",
            f"- Mean spectral centroid: {clip.mean_spectral_centroid:.3f} Hz",
            f"- Mean effective bandwidth: {clip.mean_effective_bandwidth:.3f} Hz",
            f"- Mean ERSB1: {clip.mean_ersb1:.6f}",
            f"- Mean ERSB2: {clip.mean_ersb2:.6f}",
            f"- Mean ERSB3: {clip.mean_ersb3:.6f}",
            f"- Mean spectral flatness: {clip.mean_spectral_flatness:.6f}",
            f"- Mean spectral crest: {clip.mean_spectral_crest:.6f}",
            f"- Etykieta ogolna: {clip.overall_label}",
            "",
            "Liczba ramek wg etykiet:",
            f"- voiced: {voiced_frames}",
            f"- unvoiced: {unvoiced_frames}",
            f"- silence: {silent_frames}",
            f"- speech: {speech_frames}",
            f"- music: {music_frames}",
            "",
            "Segmenty voiced/unvoiced:",
        ]
    )

    for start_time, end_time, label in result.voicing_segments:
        lines.append(f"- {start_time:.3f}s - {end_time:.3f}s: {label}")

    lines.append("")
    lines.append("Segmenty speech/music:")
    for start_time, end_time, label in result.speech_music_segments:
        lines.append(f"- {start_time:.3f}s - {end_time:.3f}s: {label}")

    return lines


def export_frames_to_csv(result: AnalysisResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "frame_index",
                "start_time_s",
                "end_time_s",
                "volume",
                "normalized_volume",
                "ste",
                "zcr",
                "silent_flag",
                "f0_autocorrelation_hz",
                "f0_amdf_hz",
                "dominant_frequency_fft_hz",
                "spectral_centroid_hz",
                "effective_bandwidth_hz",
                "ersb1",
                "ersb2",
                "ersb3",
                "spectral_flatness",
                "spectral_crest",
                "f0_cepstrum_hz",
                "voicing_label",
                "speech_music_label",
            ]
        )

        for frame in result.frames:
            writer.writerow(
                [
                    frame.index,
                    f"{frame.start_time:.6f}",
                    f"{frame.end_time:.6f}",
                    f"{frame.volume:.6f}",
                    f"{frame.normalized_volume:.6f}",
                    f"{frame.ste:.6f}",
                    f"{frame.zcr:.6f}",
                    frame.silent_flag,
                    f"{frame.f0_autocorrelation:.6f}",
                    f"{frame.f0_amdf:.6f}",
                    f"{frame.dominant_frequency_fft:.6f}",
                    f"{frame.spectral_centroid:.6f}",
                    f"{frame.effective_bandwidth:.6f}",
                    f"{frame.ersb1:.6f}",
                    f"{frame.ersb2:.6f}",
                    f"{frame.ersb3:.6f}",
                    f"{frame.spectral_flatness:.6f}",
                    f"{frame.spectral_crest:.6f}",
                    f"{frame.f0_cepstrum:.6f}",
                    frame.voicing_label,
                    frame.speech_music_label,
                ]
            )


def get_analysis_frames_for_export(
    result: AnalysisResult,
) -> tuple[np.ndarray, int, list[np.ndarray], list[float], list[float]]:
    analysis_samples, analysis_sample_rate, _downsample_factor = prepare_analysis_signal(result.audio_data)
    frames, start_times, end_times, _frame_size, _hop_size = frame_signal(
        analysis_samples,
        analysis_sample_rate,
        result.frame_ms,
        result.hop_ms,
    )
    return analysis_samples, analysis_sample_rate, frames, start_times, end_times


def export_clip_features_to_csv(result: AnalysisResult, path: str) -> None:
    audio = result.audio_data
    clip = result.clip

    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "audio_path",
                "sample_rate_hz",
                "analysis_sample_rate_hz",
                "downsample_factor",
                "channels",
                "duration_s",
                "frame_ms",
                "hop_ms",
                "frame_count",
                "mean_volume",
                "vstd",
                "vdr",
                "vu",
                "lster",
                "energy_entropy",
                "zstd",
                "hzcrr",
                "silent_ratio",
                "mean_f0_autocorrelation_hz",
                "mean_f0_amdf_hz",
                "mean_f0_cepstrum_hz",
                "mean_dominant_frequency_fft_hz",
                "mean_spectral_centroid_hz",
                "mean_effective_bandwidth_hz",
                "mean_ersb1",
                "mean_ersb2",
                "mean_ersb3",
                "mean_spectral_flatness",
                "mean_spectral_crest",
                "overall_label",
            ]
        )
        writer.writerow(
            [
                audio.path,
                audio.sample_rate,
                result.analysis_sample_rate,
                result.downsample_factor,
                audio.channels,
                f"{audio.duration_seconds:.6f}",
                f"{result.frame_ms:.6f}",
                f"{result.hop_ms:.6f}",
                len(result.frames),
                f"{clip.mean_volume:.6f}",
                f"{clip.vstd:.6f}",
                f"{clip.vdr:.6f}",
                f"{clip.vu:.6f}",
                f"{clip.lster:.6f}",
                f"{clip.energy_entropy:.6f}",
                f"{clip.zstd:.6f}",
                f"{clip.hzcrr:.6f}",
                f"{clip.silent_ratio:.6f}",
                f"{clip.mean_f0_autocorrelation:.6f}",
                f"{clip.mean_f0_amdf:.6f}",
                f"{clip.mean_f0_cepstrum:.6f}",
                f"{clip.mean_dominant_frequency_fft:.6f}",
                f"{clip.mean_spectral_centroid:.6f}",
                f"{clip.mean_effective_bandwidth:.6f}",
                f"{clip.mean_ersb1:.6f}",
                f"{clip.mean_ersb2:.6f}",
                f"{clip.mean_ersb3:.6f}",
                f"{clip.mean_spectral_flatness:.6f}",
                f"{clip.mean_spectral_crest:.6f}",
                clip.overall_label,
            ]
        )


def export_segments_to_csv(
    segments: list[tuple[float, float, str]],
    path: str,
    segment_type: str,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "segment_type",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "label",
            ]
        )

        for start_time, end_time, label in segments:
            writer.writerow(
                [
                    segment_type,
                    f"{start_time:.6f}",
                    f"{end_time:.6f}",
                    f"{max(0.0, end_time - start_time):.6f}",
                    label,
                ]
            )


def export_snapshot_time_domain_to_csv(
    snapshot: SpectrumSnapshot,
    path: str,
    audio_path: str = "",
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "audio_path",
                "start_time_s",
                "duration_s",
                "sample_rate_hz",
                "window_name",
                "sample_index",
                "time_local_s",
                "raw_sample",
                "windowed_sample",
            ]
        )

        for index, (time_value, raw_value, windowed_value) in enumerate(
            zip(snapshot.time_axis, snapshot.raw_samples, snapshot.windowed_samples)
        ):
            writer.writerow(
                [
                    audio_path,
                    f"{snapshot.start_time:.6f}",
                    f"{snapshot.duration_seconds:.6f}",
                    snapshot.sample_rate,
                    snapshot.window_name,
                    index,
                    f"{float(time_value):.9f}",
                    f"{float(raw_value):.9f}",
                    f"{float(windowed_value):.9f}",
                ]
            )


def export_snapshot_spectrum_to_csv(
    snapshot: SpectrumSnapshot,
    path: str,
    audio_path: str = "",
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "audio_path",
                "start_time_s",
                "duration_s",
                "sample_rate_hz",
                "window_name",
                "frequency_hz",
                "raw_magnitude_db_rel",
                "windowed_magnitude_db_rel",
                "spectral_centroid_hz",
                "effective_bandwidth_hz",
                "spectral_flatness",
                "spectral_crest",
                "band_energy_1",
                "band_energy_2",
                "band_energy_3",
                "band_energy_4",
                "band_ratio_1",
                "band_ratio_2",
                "band_ratio_3",
                "band_ratio_4",
                "f0_cepstrum_hz",
            ]
        )

        for index, frequency in enumerate(snapshot.frequencies):
            writer.writerow(
                [
                    audio_path,
                    f"{snapshot.start_time:.6f}",
                    f"{snapshot.duration_seconds:.6f}",
                    snapshot.sample_rate,
                    snapshot.window_name,
                    f"{float(frequency):.6f}",
                    f"{float(snapshot.raw_magnitude_db[index]):.6f}",
                    f"{float(snapshot.windowed_magnitude_db[index]):.6f}",
                    f"{snapshot.spectral_centroid:.6f}",
                    f"{snapshot.effective_bandwidth:.6f}",
                    f"{snapshot.spectral_flatness:.6f}",
                    f"{snapshot.spectral_crest:.6f}",
                    f"{snapshot.band_energies[0]:.6f}",
                    f"{snapshot.band_energies[1]:.6f}",
                    f"{snapshot.band_energies[2]:.6f}",
                    f"{snapshot.band_energies[3]:.6f}",
                    f"{snapshot.band_ratios[0]:.6f}",
                    f"{snapshot.band_ratios[1]:.6f}",
                    f"{snapshot.band_ratios[2]:.6f}",
                    f"{snapshot.band_ratios[3]:.6f}",
                    f"{snapshot.f0_cepstrum:.6f}",
                ]
            )


def export_frame_spectra_to_csv(
    result: AnalysisResult,
    path: str,
    audio_path: str = "",
    window_name: str = "hamming",
) -> None:
    _analysis_samples, analysis_sample_rate, frames, start_times, end_times = get_analysis_frames_for_export(result)
    normalized_window_name = normalize_window_name(window_name)
    frame_count = min(len(result.frames), len(frames), len(start_times), len(end_times))

    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "audio_path",
                "original_sample_rate_hz",
                "analysis_sample_rate_hz",
                "frame_ms",
                "hop_ms",
                "fft_window_name",
                "frame_index",
                "frame_start_s",
                "frame_end_s",
                "frame_center_s",
                "frame_duration_s",
                "silent_flag",
                "voicing_label",
                "speech_music_label",
                "analysis_f0_autocorrelation_hz",
                "analysis_f0_amdf_hz",
                "analysis_f0_cepstrum_hz",
                "analysis_dominant_frequency_fft_hz",
                "analysis_spectral_centroid_hz",
                "analysis_effective_bandwidth_hz",
                "analysis_ersb1",
                "analysis_ersb2",
                "analysis_ersb3",
                "analysis_spectral_flatness",
                "analysis_spectral_crest",
                "frequency_bin_index",
                "frequency_hz",
                "raw_magnitude_linear",
                "raw_power_linear",
                "raw_magnitude_db_rel",
                "windowed_magnitude_linear",
                "windowed_power_linear",
                "windowed_magnitude_db_rel",
            ]
        )

        for index in range(frame_count):
            frame_features = result.frames[index]
            frame = frames[index]
            frame_start = start_times[index]
            frame_end = end_times[index]
            frame_center = (frame_start + frame_end) * 0.5
            _raw_windowed, frequencies, raw_magnitudes, raw_power = calculate_fft_spectrum(
                frame,
                analysis_sample_rate,
                "rectangular",
            )
            _windowed, _windowed_frequencies, windowed_magnitudes, windowed_power = calculate_fft_spectrum(
                frame,
                analysis_sample_rate,
                normalized_window_name,
            )
            raw_magnitude_db = build_relative_db(raw_magnitudes)
            windowed_magnitude_db = build_relative_db(windowed_magnitudes)

            for bin_index, frequency in enumerate(frequencies):
                writer.writerow(
                    [
                        audio_path,
                        result.audio_data.sample_rate,
                        analysis_sample_rate,
                        f"{result.frame_ms:.6f}",
                        f"{result.hop_ms:.6f}",
                        normalized_window_name,
                        frame_features.index,
                        f"{frame_start:.6f}",
                        f"{frame_end:.6f}",
                        f"{frame_center:.6f}",
                        f"{max(0.0, frame_end - frame_start):.6f}",
                        frame_features.silent_flag,
                        frame_features.voicing_label,
                        frame_features.speech_music_label,
                        f"{frame_features.f0_autocorrelation:.6f}",
                        f"{frame_features.f0_amdf:.6f}",
                        f"{frame_features.f0_cepstrum:.6f}",
                        f"{frame_features.dominant_frequency_fft:.6f}",
                        f"{frame_features.spectral_centroid:.6f}",
                        f"{frame_features.effective_bandwidth:.6f}",
                        f"{frame_features.ersb1:.6f}",
                        f"{frame_features.ersb2:.6f}",
                        f"{frame_features.ersb3:.6f}",
                        f"{frame_features.spectral_flatness:.6f}",
                        f"{frame_features.spectral_crest:.6f}",
                        bin_index,
                        f"{float(frequency):.6f}",
                        f"{float(raw_magnitudes[bin_index]):.9f}",
                        f"{float(raw_power[bin_index]):.9f}",
                        f"{float(raw_magnitude_db[bin_index]):.6f}",
                        f"{float(windowed_magnitudes[bin_index]):.9f}",
                        f"{float(windowed_power[bin_index]):.9f}",
                        f"{float(windowed_magnitude_db[bin_index]):.6f}",
                    ]
                )


def export_cepstrum_snapshot_to_csv(
    snapshot: SpectrumSnapshot,
    path: str,
    audio_path: str = "",
    min_frequency_hz: float = 50.0,
    max_frequency_hz: float = 400.0,
) -> None:
    min_quefrency_ms = (1.0 / max_frequency_hz) * 1000.0
    max_quefrency_ms = (1.0 / min_frequency_hz) * 1000.0

    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "audio_path",
                "start_time_s",
                "duration_s",
                "sample_rate_hz",
                "window_name",
                "quefrency_ms",
                "cepstrum_value",
                "in_f0_search_range",
                "detected_f0_cepstrum_hz",
            ]
        )

        for quefrency_ms, cepstrum_value in zip(snapshot.cepstrum_quefrencies_ms, snapshot.cepstrum_values):
            in_search_range = min_quefrency_ms <= float(quefrency_ms) <= max_quefrency_ms
            writer.writerow(
                [
                    audio_path,
                    f"{snapshot.start_time:.6f}",
                    f"{snapshot.duration_seconds:.6f}",
                    snapshot.sample_rate,
                    snapshot.window_name,
                    f"{float(quefrency_ms):.6f}",
                    f"{float(cepstrum_value):.9f}",
                    int(in_search_range),
                    f"{snapshot.f0_cepstrum:.6f}",
                ]
            )


def export_frame_cepstra_to_csv(
    result: AnalysisResult,
    path: str,
    audio_path: str = "",
    window_name: str = "hamming",
    segment_duration_ms: float | None = None,
    min_frequency_hz: float = 50.0,
    max_frequency_hz: float = 400.0,
) -> None:
    analysis_samples, analysis_sample_rate, frames, start_times, end_times = get_analysis_frames_for_export(result)
    normalized_window_name = normalize_window_name(window_name)
    frame_count = min(len(result.frames), len(frames), len(start_times), len(end_times))
    effective_duration_ms = segment_duration_ms if segment_duration_ms is not None else max(40.0, result.frame_ms)
    effective_duration_seconds = max(0.005, effective_duration_ms / 1000.0)
    min_quefrency_ms = (1.0 / max_frequency_hz) * 1000.0
    max_quefrency_ms = (1.0 / min_frequency_hz) * 1000.0

    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "audio_path",
                "original_sample_rate_hz",
                "analysis_sample_rate_hz",
                "cepstrum_window_name",
                "cepstrum_segment_duration_ms",
                "cepstrum_min_frequency_hz",
                "cepstrum_max_frequency_hz",
                "frame_index",
                "frame_start_s",
                "frame_end_s",
                "frame_center_s",
                "segment_start_s",
                "segment_end_s",
                "segment_duration_s",
                "silent_flag",
                "voicing_label",
                "speech_music_label",
                "analysis_f0_autocorrelation_hz",
                "analysis_f0_amdf_hz",
                "analysis_f0_cepstrum_hz",
                "raw_detected_f0_cepstrum_hz",
                "quefrency_bin_index",
                "quefrency_ms",
                "equivalent_frequency_hz",
                "cepstrum_value",
                "in_f0_search_range",
            ]
        )

        for index in range(frame_count):
            frame_features = result.frames[index]
            frame_start = start_times[index]
            frame_end = end_times[index]
            frame_center = (frame_start + frame_end) * 0.5
            segment = extract_centered_audio_segment(
                analysis_samples,
                analysis_sample_rate,
                frame_center,
                effective_duration_seconds,
            )

            reference_candidates = []
            if 70.0 <= frame_features.f0_autocorrelation <= 350.0:
                reference_candidates.append(frame_features.f0_autocorrelation)
            if 70.0 <= frame_features.f0_amdf <= 350.0:
                reference_candidates.append(frame_features.f0_amdf)
            reference_frequency = mean_value(reference_candidates) if reference_candidates else None

            quefrencies, cepstrum = calculate_real_cepstrum(
                segment,
                analysis_sample_rate,
                normalized_window_name,
            )
            raw_detected_f0 = detect_cepstrum_f0_from_curve(
                quefrencies,
                cepstrum,
                min_frequency=min_frequency_hz,
                max_frequency=max_frequency_hz,
                reference_frequency=reference_frequency,
            )

            if len(quefrencies) == 0 or len(cepstrum) == 0:
                continue

            half_length = (len(quefrencies) // 2) + 1
            quefrencies = quefrencies[:half_length]
            cepstrum = cepstrum[:half_length]

            segment_start = frame_center - (effective_duration_seconds * 0.5)
            segment_end = segment_start + effective_duration_seconds

            for bin_index, quefrency in enumerate(quefrencies):
                quefrency_ms = float(quefrency * 1000.0)
                equivalent_frequency = 0.0
                if quefrency > 1e-12:
                    equivalent_frequency = 1.0 / float(quefrency)
                in_search_range = min_quefrency_ms <= quefrency_ms <= max_quefrency_ms

                writer.writerow(
                    [
                        audio_path,
                        result.audio_data.sample_rate,
                        analysis_sample_rate,
                        normalized_window_name,
                        f"{effective_duration_ms:.6f}",
                        f"{min_frequency_hz:.6f}",
                        f"{max_frequency_hz:.6f}",
                        frame_features.index,
                        f"{frame_start:.6f}",
                        f"{frame_end:.6f}",
                        f"{frame_center:.6f}",
                        f"{segment_start:.6f}",
                        f"{segment_end:.6f}",
                        f"{effective_duration_seconds:.6f}",
                        frame_features.silent_flag,
                        frame_features.voicing_label,
                        frame_features.speech_music_label,
                        f"{frame_features.f0_autocorrelation:.6f}",
                        f"{frame_features.f0_amdf:.6f}",
                        f"{frame_features.f0_cepstrum:.6f}",
                        f"{raw_detected_f0:.6f}",
                        bin_index,
                        f"{quefrency_ms:.6f}",
                        f"{equivalent_frequency:.6f}",
                        f"{float(cepstrum[bin_index]):.9f}",
                        int(in_search_range),
                    ]
                )


def export_spectrogram_to_csv(
    spectrogram: SpectrogramData,
    path: str,
    audio_path: str = "",
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "audio_path",
                "window_name",
                "frame_ms",
                "overlap_percent",
                "sample_rate_hz",
                "time_s",
                "frequency_hz",
                "magnitude_db_rel",
            ]
        )

        for frequency_index, frequency in enumerate(spectrogram.frequencies):
            for time_index, time_value in enumerate(spectrogram.times):
                writer.writerow(
                    [
                        audio_path,
                        spectrogram.window_name,
                        f"{spectrogram.frame_ms:.6f}",
                        f"{spectrogram.overlap_percent:.6f}",
                        spectrogram.sample_rate,
                        f"{float(time_value):.6f}",
                        f"{float(frequency):.6f}",
                        f"{float(spectrogram.magnitude_db[frequency_index, time_index]):.6f}",
                    ]
                )


def export_summary_to_txt(result: AnalysisResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(build_summary_lines(result)))
