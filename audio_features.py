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
    overall_label: str


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


def frame_signal(samples: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[list[np.ndarray], list[float], list[float], int, int]:
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


def calculate_autocorrelation_f0(frame: np.ndarray, sample_rate: int, min_frequency: float = 50.0, max_frequency: float = 500.0) -> float:
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


def calculate_amdf_f0(frame: np.ndarray, sample_rate: int, min_frequency: float = 50.0, max_frequency: float = 500.0) -> float:
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


def create_hamming_window(size: int) -> np.ndarray:
    if size <= 1:
        return np.ones(size, dtype=np.float64)

    values = np.zeros(size, dtype=np.float64)
    for index in range(size):
        values[index] = 0.54 - 0.46 * math.cos((2.0 * math.pi * index) / (size - 1))
    return values


def calculate_dominant_frequency_fft(frame: np.ndarray, sample_rate: int) -> float:
    if len(frame) < 2 or sample_rate <= 0:
        return 0.0

    window = create_hamming_window(len(frame))
    windowed_frame = frame * window
    spectrum = np.fft.rfft(windowed_frame)
    magnitudes = np.abs(spectrum)

    if len(magnitudes) <= 1:
        return 0.0

    magnitudes[0] = 0.0
    best_index = int(np.argmax(magnitudes))
    frequencies = np.fft.rfftfreq(len(frame), d=1.0 / sample_rate)
    return float(frequencies[best_index])


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
    analysis_sample_rate = audio_data.sample_rate
    downsample_factor = 1
    analysis_samples = audio_data.samples

    if audio_data.sample_rate > 16000:
        downsample_factor = math.ceil(audio_data.sample_rate / 16000)
        analysis_sample_rate = int(round(audio_data.sample_rate / downsample_factor))
        analysis_samples = downsample_signal(audio_data.samples, downsample_factor)

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

    for frame in frames:
        volume = calculate_volume(frame)
        ste = calculate_ste(frame)
        zcr = calculate_zcr(frame)
        f0_autocorrelation = calculate_autocorrelation_f0(frame, analysis_sample_rate)
        f0_amdf = calculate_amdf_f0(frame, analysis_sample_rate)
        dominant_frequency = calculate_dominant_frequency_fft(frame, analysis_sample_rate)

        volumes.append(volume)
        ste_values.append(ste)
        zcr_values.append(zcr)
        f0_autocorrelation_values.append(f0_autocorrelation)
        f0_amdf_values.append(f0_amdf)
        dominant_frequencies.append(dominant_frequency)

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
        has_valid_f0 = valid_f0_autocorrelation or valid_f0_amdf
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

        if voicing_labels[index] != "voiced":
            f0_autocorrelation_values[index] = 0.0
            f0_amdf_values[index] = 0.0
            continue

        if not valid_f0_autocorrelation:
            f0_autocorrelation_values[index] = 0.0
        if not valid_f0_amdf:
            f0_amdf_values[index] = 0.0

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
                    frame.voicing_label,
                    frame.speech_music_label,
                ]
            )


def export_summary_to_txt(result: AnalysisResult, path: str) -> None:
    lines = []
    audio = result.audio_data
    clip = result.clip

    lines.append("Analiza sygnalu audio w dziedzinie czasu")
    lines.append("")
    lines.append(f"Plik: {audio.path}")
    lines.append(f"Czestotliwosc probkowania: {audio.sample_rate} Hz")
    lines.append(f"Czestotliwosc analizy: {result.analysis_sample_rate} Hz")
    lines.append(f"Downsample factor: {result.downsample_factor}")
    lines.append(f"Liczba kanalow: {audio.channels}")
    lines.append(f"Dlugosc sygnalu: {audio.duration_seconds:.3f} s")
    lines.append(f"Frame size: {result.frame_ms:.2f} ms ({result.frame_size_samples} probek)")
    lines.append(f"Hop size: {result.hop_ms:.2f} ms ({result.hop_size_samples} probek)")
    lines.append(f"Liczba ramek: {len(result.frames)}")
    lines.append("")
    lines.append("Progi heurystyczne:")
    lines.append(f"- prog ciszy dla volume: {result.silence_volume_threshold:.4f}")
    lines.append(f"- prog ciszy dla ZCR: {result.silence_zcr_threshold:.4f}")
    lines.append("")
    lines.append("Cechy clip-level:")
    lines.append(f"- Mean Volume: {clip.mean_volume:.6f}")
    lines.append(f"- VSTD: {clip.vstd:.6f}")
    lines.append(f"- VDR: {clip.vdr:.6f}")
    lines.append(f"- VU: {clip.vu:.6f}")
    lines.append(f"- LSTER: {clip.lster:.6f}")
    lines.append(f"- Energy Entropy: {clip.energy_entropy:.6f}")
    lines.append(f"- ZSTD: {clip.zstd:.6f}")
    lines.append(f"- HZCRR: {clip.hzcrr:.6f}")
    lines.append(f"- Silent Ratio: {clip.silent_ratio:.6f}")
    lines.append(f"- Mean F0 (autokorelacja): {clip.mean_f0_autocorrelation:.3f} Hz")
    lines.append(f"- Mean F0 (AMDF): {clip.mean_f0_amdf:.3f} Hz")
    lines.append(f"- Mean dominant frequency FFT: {clip.mean_dominant_frequency_fft:.3f} Hz")
    lines.append(f"- Ogolna etykieta klipu: {clip.overall_label}")
    lines.append("")
    lines.append("Segmenty voiced/unvoiced:")
    for start_time, end_time, label in result.voicing_segments:
        lines.append(f"- {start_time:.3f}s - {end_time:.3f}s: {label}")
    lines.append("")
    lines.append("Segmenty speech/music:")
    for start_time, end_time, label in result.speech_music_segments:
        lines.append(f"- {start_time:.3f}s - {end_time:.3f}s: {label}")

    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))
