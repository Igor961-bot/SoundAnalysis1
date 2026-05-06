import math
import os
import sys

import numpy as np


VENDOR_DIR = os.path.join(os.path.dirname(__file__), ".vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSlider,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )


from audio_features import (
    analyze_audio,
    build_summary_lines,
    compute_spectrum_snapshot,
    compute_spectrogram,
    export_clip_features_to_csv,
    export_cepstrum_snapshot_to_csv,
    export_frame_cepstra_to_csv,
    export_frame_spectra_to_csv,
    export_frames_to_csv,
    export_segments_to_csv,
    export_snapshot_spectrum_to_csv,
    export_snapshot_time_domain_to_csv,
    export_spectrogram_to_csv,
    export_summary_to_txt,
    load_wav_file,
)


class AnalysisThread(QThread):
    analysis_finished = pyqtSignal(object)
    analysis_failed = pyqtSignal(str)

    def __init__(self, audio_data, frame_ms: float, hop_ms: float) -> None:
        super().__init__()
        self.audio_data = audio_data
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms

    def run(self) -> None:
        try:
            result = analyze_audio(self.audio_data, frame_ms=self.frame_ms, hop_ms=self.hop_ms)
        except Exception as error:
            self.analysis_failed.emit(str(error))
            return

        self.analysis_finished.emit(result)


class LinePlotWidget(QWidget):
    def __init__(
        self,
        title: str,
        line_color: str,
        height: int = 190,
        parent: QWidget | None = None,
        x_unit: str = "s",
        x_decimals: int = 2,
        y_decimals: int = 3,
        x_axis_label: str = "Czas [s]",
        y_axis_label: str = "Wartosc",
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.line_color = QColor(line_color)
        self.times = []
        self.values = []
        self.overlay_segments = []
        self.playhead_time = None
        self.view_start_time = None
        self.view_end_time = None
        self.x_unit = x_unit
        self.x_decimals = x_decimals
        self.y_decimals = y_decimals
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.fixed_y_min = None
        self.fixed_y_max = None
        self.fixed_y_base_min = None
        self.fixed_y_base_max = None
        self.fixed_y_anchor = "center"
        self.fixed_y_zoom = 1.0
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_plot_data(
        self,
        times: list[float],
        values: list[float],
        overlay_segments: list[tuple[float, float, str]] | None = None,
    ) -> None:
        self.times = list(times)
        self.values = list(values)
        self.overlay_segments = overlay_segments or []
        self.update()

    def clear_plot(self) -> None:
        self.times = []
        self.values = []
        self.overlay_segments = []
        self.playhead_time = None
        self.update()

    def set_playhead_time(self, playhead_time: float | None) -> None:
        self.playhead_time = playhead_time
        self.update()

    def set_view_range(self, start_time: float | None, end_time: float | None) -> None:
        self.view_start_time = start_time
        self.view_end_time = end_time
        self.update()

    def set_axis_format(self, x_unit: str = "s", x_decimals: int = 2, y_decimals: int = 3) -> None:
        self.x_unit = x_unit
        self.x_decimals = x_decimals
        self.y_decimals = y_decimals
        self.update()

    def set_axis_labels(self, x_axis_label: str, y_axis_label: str) -> None:
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.update()

    def set_fixed_y_range(self, y_min: float | None, y_max: float | None, anchor: str = "center") -> None:
        if y_min is None or y_max is None:
            self.fixed_y_min = None
            self.fixed_y_max = None
            self.fixed_y_base_min = None
            self.fixed_y_base_max = None
        else:
            self.fixed_y_base_min = float(y_min)
            self.fixed_y_base_max = float(y_max)
            self.fixed_y_anchor = anchor
            self.apply_fixed_y_zoom()
        self.update()

    def set_fixed_y_zoom(self, zoom_factor: float) -> None:
        self.fixed_y_zoom = max(0.1, float(zoom_factor))
        self.apply_fixed_y_zoom()
        self.update()

    def apply_fixed_y_zoom(self) -> None:
        if self.fixed_y_base_min is None or self.fixed_y_base_max is None:
            self.fixed_y_min = None
            self.fixed_y_max = None
            return

        base_min = self.fixed_y_base_min
        base_max = self.fixed_y_base_max
        base_span = max(1e-12, base_max - base_min)

        if self.fixed_y_anchor == "min":
            self.fixed_y_min = base_min
            self.fixed_y_max = base_min + (base_span / self.fixed_y_zoom)
        elif self.fixed_y_anchor == "max":
            self.fixed_y_max = base_max
            self.fixed_y_min = base_max - (base_span / self.fixed_y_zoom)
        else:
            center = (base_min + base_max) * 0.5
            half_span = (base_span * 0.5) / self.fixed_y_zoom
            self.fixed_y_min = center - half_span
            self.fixed_y_max = center + half_span

        self.update()

    def format_axis_value(self, value: float, decimals: int, unit: str = "") -> str:
        suffix = f" {unit}" if unit else ""
        return f"{value:.{decimals}f}{suffix}"

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f6f1"))

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#222222"))
        painter.drawText(14, 20, self.title)

        plot_rect = QRectF(72, 34, max(60, self.width() - 92), max(60, self.height() - 86))
        painter.setPen(QColor("#404040"))
        painter.drawRect(plot_rect)

        if not self.times or not self.values:
            painter.drawText(int(plot_rect.left()) + 12, int(plot_rect.center().y()), "Brak")
            return

        x_min = self.times[0]
        x_max = self.times[-1]
        if x_max <= x_min:
            x_max = x_min + 1.0

        if self.view_start_time is not None and self.view_end_time is not None:
            x_min = max(self.times[0], self.view_start_time)
            x_max = min(self.times[-1], self.view_end_time)
            if x_max <= x_min:
                x_max = x_min + 1e-6

        for start_time, end_time, color_name in self.overlay_segments:
            if end_time < x_min or start_time > x_max:
                continue
            overlay_color = QColor(color_name)
            overlay_color.setAlpha(70)
            start_x = plot_rect.left() + ((start_time - x_min) / (x_max - x_min)) * plot_rect.width()
            end_x = plot_rect.left() + ((end_time - x_min) / (x_max - x_min)) * plot_rect.width()
            left_x = max(plot_rect.left(), min(start_x, end_x))
            right_x = min(plot_rect.right(), max(start_x, end_x))
            width = max(1.0, right_x - left_x)
            painter.fillRect(QRectF(left_x, plot_rect.top(), width, plot_rect.height()), overlay_color)

        visible_times, visible_values = self.reduce_series(max(2, int(plot_rect.width())), x_min, x_max)
        if not visible_times or not visible_values:
            painter.drawText(int(plot_rect.left()) + 12, int(plot_rect.center().y()), "Brak")
            return

        if self.fixed_y_min is not None and self.fixed_y_max is not None:
            y_min = self.fixed_y_min
            y_max = self.fixed_y_max
        else:
            y_min = min(visible_values)
            y_max = max(visible_values)
            if abs(y_max - y_min) < 1e-12:
                delta = 1.0 if abs(y_max) < 1e-12 else abs(y_max) * 0.2
                y_min -= delta
                y_max += delta

        if y_min < 0.0 < y_max:
            zero_y = plot_rect.bottom() - ((0.0 - y_min) / (y_max - y_min)) * plot_rect.height()
            painter.setPen(QPen(QColor("#bbbbbb"), 1, Qt.DashLine))
            painter.drawLine(QPointF(plot_rect.left(), zero_y), QPointF(plot_rect.right(), zero_y))

        painter.setPen(QPen(self.line_color, 1.8))
        previous_point = None
        for time_value, signal_value in zip(visible_times, visible_values):
            x_position = plot_rect.left() + ((time_value - x_min) / (x_max - x_min)) * plot_rect.width()
            y_position = plot_rect.bottom() - ((signal_value - y_min) / (y_max - y_min)) * plot_rect.height()
            y_position = max(plot_rect.top(), min(plot_rect.bottom(), y_position))
            current_point = QPointF(x_position, y_position)
            if previous_point is not None:
                painter.drawLine(previous_point, current_point)
            previous_point = current_point

        if self.playhead_time is not None and x_min <= self.playhead_time <= x_max:
            playhead_x = plot_rect.left() + ((self.playhead_time - x_min) / (x_max - x_min)) * plot_rect.width()
            painter.setPen(QPen(QColor("#d7263d"), 2))
            painter.drawLine(QPointF(playhead_x, plot_rect.top()), QPointF(playhead_x, plot_rect.bottom()))

        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        painter.setPen(QColor("#555555"))
        painter.drawText(8, int(plot_rect.top()) + 8, self.format_axis_value(y_max, self.y_decimals))
        painter.drawText(8, int(plot_rect.bottom()), self.format_axis_value(y_min, self.y_decimals))
        painter.drawText(
            int(plot_rect.left()),
            self.height() - 24,
            self.format_axis_value(x_min, self.x_decimals, self.x_unit),
        )
        painter.drawText(
            int(plot_rect.right()) - 74,
            self.height() - 24,
            self.format_axis_value(x_max, self.x_decimals, self.x_unit),
        )
        painter.drawText(
            QRectF(plot_rect.left(), self.height() - 18, plot_rect.width(), 14),
            Qt.AlignCenter,
            self.x_axis_label,
        )

        painter.save()
        painter.translate(18, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-80, -10, 160, 20), Qt.AlignCenter, self.y_axis_label)
        painter.restore()

    def reduce_series(self, target_points: int, x_min: float, x_max: float) -> tuple[list[float], list[float]]:
        visible_times = []
        visible_values = []
        for time_value, signal_value in zip(self.times, self.values):
            if x_min <= time_value <= x_max:
                visible_times.append(time_value)
                visible_values.append(signal_value)

        if not visible_times:
            return [], []

        if len(visible_values) <= target_points:
            return visible_times, visible_values

        step = max(1, math.ceil(len(visible_values) / target_points))
        reduced_times = []
        reduced_values = []

        for index in range(0, len(visible_values), step):
            reduced_times.append(visible_times[index])
            reduced_values.append(visible_values[index])

        if reduced_times[-1] != visible_times[-1]:
            reduced_times.append(visible_times[-1])
            reduced_values.append(visible_values[-1])

        return reduced_times, reduced_values


class TimelineWidget(QWidget):
    LABEL_COLORS = {
        "silence": "#b9b9b9",
        "voiced": "#7ab87f",
        "unvoiced": "#e1a84f",
        "speech": "#5f9e6e",
        "music": "#c98142",
        "mixed": "#5d89c6",
    }

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.segments = []
        self.total_duration = 1.0
        self.playhead_time = None
        self.view_start_time = None
        self.view_end_time = None
        self.x_axis_label = "Czas w pliku [s]"
        self.row_label = "Segment"
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_timeline_data(self, segments: list[tuple[float, float, str]], total_duration: float) -> None:
        self.segments = segments
        self.total_duration = max(total_duration, 1e-6)
        self.update()

    def clear_timeline(self) -> None:
        self.segments = []
        self.total_duration = 1.0
        self.playhead_time = None
        self.update()

    def set_playhead_time(self, playhead_time: float | None) -> None:
        self.playhead_time = playhead_time
        self.update()

    def set_view_range(self, start_time: float | None, end_time: float | None) -> None:
        self.view_start_time = start_time
        self.view_end_time = end_time
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f6f1"))

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#222222"))
        painter.drawText(14, 20, self.title)

        timeline_rect = QRectF(70, 30, max(80, self.width() - 84), 30)
        painter.setPen(QColor("#404040"))
        painter.drawRect(timeline_rect)

        if not self.segments:
            painter.drawText(18, 52, "Brak")
            return

        view_start = 0.0
        view_end = self.total_duration
        if self.view_start_time is not None and self.view_end_time is not None:
            view_start = max(0.0, self.view_start_time)
            view_end = min(self.total_duration, self.view_end_time)
            if view_end <= view_start:
                view_end = view_start + 1e-6

        for start_time, end_time, label in self.segments:
            if end_time < view_start or start_time > view_end:
                continue
            start_x = timeline_rect.left() + ((start_time - view_start) / (view_end - view_start)) * timeline_rect.width()
            end_x = timeline_rect.left() + ((end_time - view_start) / (view_end - view_start)) * timeline_rect.width()
            color = QColor(self.LABEL_COLORS.get(label, "#7d7d7d"))
            painter.fillRect(QRectF(start_x, timeline_rect.top(), max(1.0, end_x - start_x), timeline_rect.height()), color)

            if (end_x - start_x) >= 56.0:
                painter.setPen(QColor("#1f1f1f"))
                painter.drawText(QRectF(start_x, timeline_rect.top(), end_x - start_x, timeline_rect.height()), Qt.AlignCenter, label)

        if self.playhead_time is not None and view_start <= self.playhead_time <= view_end:
            playhead_x = timeline_rect.left() + ((self.playhead_time - view_start) / (view_end - view_start)) * timeline_rect.width()
            painter.setPen(QPen(QColor("#d7263d"), 2))
            painter.drawLine(QPointF(playhead_x, timeline_rect.top()), QPointF(playhead_x, timeline_rect.bottom()))

        painter.setPen(QColor("#555555"))
        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        painter.drawText(14, 48, self.row_label)
        painter.drawText(14, 76, "Legenda:")
        painter.drawText(int(timeline_rect.left()), 92, f"{view_start:.2f} s")
        painter.drawText(int(timeline_rect.right()) - 60, 92, f"{view_end:.2f} s")
        painter.drawText(QRectF(timeline_rect.left(), 82, timeline_rect.width(), 14), Qt.AlignCenter, self.x_axis_label)

        current_x = 70
        used_labels = []
        for _, _, label in self.segments:
            if label not in used_labels:
                used_labels.append(label)

        for label in used_labels:
            color = QColor(self.LABEL_COLORS.get(label, "#7d7d7d"))
            painter.fillRect(current_x, 67, 12, 12, color)
            painter.drawRect(current_x, 67, 12, 12)
            painter.drawText(current_x + 18, 78, label)
            current_x += 90


class SpectrogramWidget(QWidget):
    def __init__(self, title: str, height: int = 360, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.times = np.zeros(0, dtype=np.float64)
        self.frequencies = np.zeros(0, dtype=np.float64)
        self.magnitude_db = np.zeros((0, 0), dtype=np.float64)
        self.image = None
        self.playhead_time = None
        self.view_start_time = None
        self.view_end_time = None
        self.x_axis_label = "Czas w pliku [s]"
        self.y_axis_label = "Czestotliwosc [Hz]"
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_spectrogram_data(self, times: np.ndarray, frequencies: np.ndarray, magnitude_db: np.ndarray) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.magnitude_db = np.asarray(magnitude_db, dtype=np.float64)
        self.image = self.build_image()
        self.update()

    def clear_spectrogram(self) -> None:
        self.times = np.zeros(0, dtype=np.float64)
        self.frequencies = np.zeros(0, dtype=np.float64)
        self.magnitude_db = np.zeros((0, 0), dtype=np.float64)
        self.image = None
        self.playhead_time = None
        self.update()

    def set_playhead_time(self, playhead_time: float | None) -> None:
        self.playhead_time = playhead_time
        self.update()

    def set_view_range(self, start_time: float | None, end_time: float | None) -> None:
        self.view_start_time = start_time
        self.view_end_time = end_time
        self.update()

    def interpolate_channel(self, start: int, end: int, fraction: float) -> int:
        return int(round(start + ((end - start) * fraction)))

    def color_for_value(self, normalized_value: float) -> QColor:
        value = max(0.0, min(1.0, normalized_value))
        if value < 0.33:
            fraction = value / 0.33
            return QColor(
                self.interpolate_channel(10, 41, fraction),
                self.interpolate_channel(18, 98, fraction),
                self.interpolate_channel(40, 255, fraction),
            )
        if value < 0.66:
            fraction = (value - 0.33) / 0.33
            return QColor(
                self.interpolate_channel(41, 48, fraction),
                self.interpolate_channel(98, 196, fraction),
                self.interpolate_channel(255, 141, fraction),
            )
        fraction = (value - 0.66) / 0.34
        return QColor(
            self.interpolate_channel(48, 255, fraction),
            self.interpolate_channel(196, 238, fraction),
            self.interpolate_channel(141, 88, fraction),
        )

    def build_image(self) -> QImage | None:
        if self.magnitude_db.size == 0 or self.magnitude_db.shape[0] == 0 or self.magnitude_db.shape[1] == 0:
            return None

        min_db = float(np.min(self.magnitude_db))
        max_db = float(np.max(self.magnitude_db))
        if max_db <= min_db:
            max_db = min_db + 1.0

        image = QImage(self.magnitude_db.shape[1], self.magnitude_db.shape[0], QImage.Format_RGB32)
        for row_index in range(self.magnitude_db.shape[0]):
            image_row = self.magnitude_db.shape[0] - 1 - row_index
            for column_index in range(self.magnitude_db.shape[1]):
                value = float(self.magnitude_db[row_index, column_index])
                normalized = (value - min_db) / (max_db - min_db)
                image.setPixelColor(column_index, image_row, self.color_for_value(normalized))
        return image

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f6f1"))

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#222222"))
        painter.drawText(14, 20, self.title)

        plot_rect = QRectF(72, 34, max(60, self.width() - 92), max(60, self.height() - 86))
        painter.setPen(QColor("#404040"))
        painter.drawRect(plot_rect)

        if self.image is None or len(self.times) == 0:
            painter.drawText(int(plot_rect.left()) + 12, int(plot_rect.center().y()), "Brak")
            return

        start_index = 0
        end_index = len(self.times) - 1
        view_start = float(self.times[0])
        view_end = float(self.times[-1])

        if self.view_start_time is not None and self.view_end_time is not None:
            view_start = max(view_start, self.view_start_time)
            view_end = min(view_end, self.view_end_time)
            start_index = int(np.searchsorted(self.times, view_start, side="left"))
            end_index = int(np.searchsorted(self.times, view_end, side="right")) - 1
            start_index = max(0, min(start_index, len(self.times) - 1))
            end_index = max(start_index, min(end_index, len(self.times) - 1))
            view_start = float(self.times[start_index])
            view_end = float(self.times[end_index])

        source_rect = QRectF(
            float(start_index),
            0.0,
            float(max(1, end_index - start_index + 1)),
            float(self.image.height()),
        )
        painter.drawImage(plot_rect, self.image, source_rect)
        painter.setPen(QColor("#404040"))
        painter.drawRect(plot_rect)

        if self.playhead_time is not None and view_start <= self.playhead_time <= view_end and view_end > view_start:
            playhead_x = plot_rect.left() + ((self.playhead_time - view_start) / (view_end - view_start)) * plot_rect.width()
            painter.setPen(QPen(QColor("#d7263d"), 2))
            painter.drawLine(QPointF(playhead_x, plot_rect.top()), QPointF(playhead_x, plot_rect.bottom()))

        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        painter.setPen(QColor("#555555"))
        max_frequency = float(self.frequencies[-1]) if len(self.frequencies) else 0.0
        painter.drawText(8, int(plot_rect.top()) + 8, f"{max_frequency:.0f} Hz")
        painter.drawText(8, int(plot_rect.bottom()), "0 Hz")
        painter.drawText(int(plot_rect.left()), self.height() - 24, f"{view_start:.2f} s")
        painter.drawText(int(plot_rect.right()) - 60, self.height() - 24, f"{view_end:.2f} s")
        painter.drawText(
            QRectF(plot_rect.left(), self.height() - 18, plot_rect.width(), 14),
            Qt.AlignCenter,
            self.x_axis_label,
        )

        painter.save()
        painter.translate(18, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-90, -10, 180, 20), Qt.AlignCenter, self.y_axis_label)
        painter.restore()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.audio_data = None
        self.analysis_result = None
        self.spectrogram_data = None
        self.fft_snapshot = None
        self.cepstrum_snapshot = None
        self.analysis_thread = None
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(50)
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)
        self.player.stateChanged.connect(self.on_player_state_changed)
        self.position_slider_is_dragged = False
        self.current_view_start_seconds = 0.0
        self.analysis_controls_enabled = True
        self.live_update_interval_ms = 80
        self.last_fft_live_update_ms = -1_000_000
        self.last_cepstrum_live_update_ms = -1_000_000
        self.plot_scale_controls = {}

        self.setWindowTitle("Projekt 1 + Projekt 2")
        self.resize(1480, 980)
        self.build_ui()

    def create_scaled_plot_panel(self, plot_widget: QWidget, scale_key: str) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(4)
        panel_layout.addWidget(plot_widget)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(6, 0, 6, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(QLabel("Zoom Y:"))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(10, 60)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        slider.setValue(10)
        slider.valueChanged.connect(lambda value, key=scale_key: self.on_plot_scale_changed(key, value))
        controls_layout.addWidget(slider, 1)

        value_label = QLabel("1.0x")
        value_label.setMinimumWidth(48)
        controls_layout.addWidget(value_label)
        panel_layout.addLayout(controls_layout)

        self.plot_scale_controls[scale_key] = {
            "plot": plot_widget,
            "slider": slider,
            "label": value_label,
        }
        return panel

    def on_plot_scale_changed(self, scale_key: str, slider_value: int) -> None:
        if scale_key not in self.plot_scale_controls:
            return

        zoom_factor = slider_value / 10.0
        plot_widget = self.plot_scale_controls[scale_key]["plot"]
        value_label = self.plot_scale_controls[scale_key]["label"]
        plot_widget.set_fixed_y_zoom(zoom_factor)
        value_label.setText(f"{zoom_factor:.1f}x")

    def build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        controls_layout = QGridLayout()

        self.open_button = QPushButton("Wczytaj WAV")
        self.open_button.clicked.connect(self.open_wav_file)
        controls_layout.addWidget(self.open_button, 0, 0)

        self.analyze_button = QPushButton("Analiza")
        self.analyze_button.clicked.connect(self.analyze_current_audio)
        controls_layout.addWidget(self.analyze_button, 0, 1)

        self.export_csv_button = QPushButton("to CSV")
        self.export_csv_button.clicked.connect(self.export_csv)
        controls_layout.addWidget(self.export_csv_button, 0, 2)

        self.export_txt_button = QPushButton("to TXT")
        self.export_txt_button.clicked.connect(self.export_txt)
        controls_layout.addWidget(self.export_txt_button, 0, 3)

        self.export_project2_csv_button = QPushButton("CSV P2")
        self.export_project2_csv_button.clicked.connect(self.export_project2_csv_bundle)
        controls_layout.addWidget(self.export_project2_csv_button, 0, 4)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_audio)
        controls_layout.addWidget(self.play_button, 0, 5)

        self.pause_button = QPushButton("Pauza")
        self.pause_button.clicked.connect(self.pause_audio)
        controls_layout.addWidget(self.pause_button, 0, 6)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        controls_layout.addWidget(self.stop_button, 0, 7)

        controls_layout.addWidget(QLabel("Frame [ms]:"), 0, 8)
        self.frame_input = QLineEdit("20")
        self.frame_input.setMaximumWidth(80)
        controls_layout.addWidget(self.frame_input, 0, 9)

        controls_layout.addWidget(QLabel("Hop [ms]:"), 0, 10)
        self.hop_input = QLineEdit("10")
        self.hop_input.setMaximumWidth(80)
        controls_layout.addWidget(self.hop_input, 0, 11)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderPressed.connect(self.on_position_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_position_slider_released)
        self.position_slider.sliderMoved.connect(self.on_position_slider_moved)
        controls_layout.addWidget(self.position_slider, 1, 0, 1, 11)

        self.position_label = QLabel("00:00.0 / 00:00.0")
        controls_layout.addWidget(self.position_label, 1, 11)

        controls_layout.addWidget(QLabel("Zoom:"), 2, 0)
        self.zoom_selector = QComboBox()
        self.zoom_selector.addItem("Caly plik", None)
        self.zoom_selector.addItem("2 s", 2.0)
        self.zoom_selector.addItem("5 s", 5.0)
        self.zoom_selector.addItem("10 s", 10.0)
        self.zoom_selector.addItem("30 s", 30.0)
        self.zoom_selector.addItem("60 s", 60.0)
        self.zoom_selector.currentIndexChanged.connect(self.on_zoom_changed)
        controls_layout.addWidget(self.zoom_selector, 2, 1)

        self.view_slider = QSlider(Qt.Horizontal)
        self.view_slider.setEnabled(False)
        self.view_slider.valueChanged.connect(self.on_view_slider_changed)
        controls_layout.addWidget(self.view_slider, 2, 2, 1, 9)

        self.view_label = QLabel("Widok: caly plik")
        controls_layout.addWidget(self.view_label, 2, 11)

        self.info_label = QLabel("Wczytaj plik WAV, następnie wciśnij Analiza")
        self.info_label.setWordWrap(True)
        controls_layout.addWidget(self.info_label, 3, 0, 1, 12)

        main_layout.addLayout(controls_layout)

        self.tabs = QTabWidget()

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.waveform_plot = LinePlotWidget(
            "Przebieg czasowy z zaznaczona cisza",
            "#335c88",
            240,
            x_axis_label="Czas w pliku [s]",
            y_axis_label="Amplituda",
        )
        self.voicing_timeline = TimelineWidget("Fragmenty voiced / unvoiced / silence")
        self.speech_music_timeline = TimelineWidget("Fragmenty speech / music / silence")
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.waveform_plot)
        summary_layout.addWidget(self.voicing_timeline)
        summary_layout.addWidget(self.speech_music_timeline)
        summary_layout.addWidget(self.summary_text)
        self.tabs.addTab(summary_tab, "Podsumowanie")

        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        feature_controls = QHBoxLayout()
        feature_controls.addWidget(QLabel("Cecha:"))
        self.feature_selector = QComboBox()
        self.feature_selector.addItems(
            [
                "Volume",
                "STE",
                "ZCR",
                "F0 autokorelacja",
                "F0 AMDF",
                "Dominujaca czestotliwosc FFT",
                "Centroid widmowy",
                "Bandwidth efektywny",
                "ERSB1",
                "ERSB2",
                "ERSB3",
                "Spectral Flatness",
                "Spectral Crest",
                "F0 cepstrum",
            ]
        )
        self.feature_selector.currentIndexChanged.connect(self.update_feature_plot)
        feature_controls.addWidget(self.feature_selector)
        feature_controls.addStretch(1)
        features_layout.addLayout(feature_controls)
        self.feature_plot = LinePlotWidget(
            "Volume",
            "#5f9e6e",
            340,
            x_unit="s",
            x_decimals=2,
            y_decimals=3,
            x_axis_label="Czas w pliku [s]",
            y_axis_label="Wartosc cechy",
        )
        features_layout.addWidget(self.feature_plot)
        self.tabs.addTab(features_tab, "Cechy ramek")

        frames_tab = QWidget()
        frames_layout = QVBoxLayout(frames_tab)

        self.frames_table = QTableWidget()
        self.frames_table.setColumnCount(21)
        self.frames_table.setHorizontalHeaderLabels(
            [
                "Nr",
                "Start [s]",
                "Koniec [s]",
                "Volume",
                "Vol norm",
                "STE",
                "ZCR",
                "Cisza",
                "F0 auto",
                "F0 AMDF",
                "FFT dom",
                "Centroid",
                "Bandwidth",
                "ERSB1",
                "ERSB2",
                "ERSB3",
                "SFM",
                "SCF",
                "F0 cepstrum",
                "Voicing",
                "Speech/Music",
            ]
        )
        header = self.frames_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        frames_layout.addWidget(self.frames_table)
        self.tabs.addTab(frames_tab, "Ramki")

        self.fft_tab = QWidget()
        fft_layout = QVBoxLayout(self.fft_tab)
        fft_controls = QGridLayout()
        fft_controls.addWidget(QLabel("Zakres:"), 0, 0)
        self.fft_domain_selector = QComboBox()
        self.fft_domain_selector.addItem("Wybrana ramka", "frame")
        self.fft_domain_selector.addItem("Caly sygnal", "full")
        self.fft_domain_selector.currentIndexChanged.connect(self.on_fft_mode_changed)
        fft_controls.addWidget(self.fft_domain_selector, 0, 1)

        fft_controls.addWidget(QLabel("Start [s]:"), 0, 2)
        self.fft_start_input = QLineEdit("0.0")
        self.fft_start_input.setMaximumWidth(90)
        fft_controls.addWidget(self.fft_start_input, 0, 3)

        fft_controls.addWidget(QLabel("Dlugosc [ms]:"), 0, 4)
        self.fft_duration_input = QLineEdit("40")
        self.fft_duration_input.setMaximumWidth(90)
        fft_controls.addWidget(self.fft_duration_input, 0, 5)

        fft_controls.addWidget(QLabel("Okno:"), 0, 6)
        self.fft_window_selector = QComboBox()
        self.populate_window_selector(self.fft_window_selector)
        fft_controls.addWidget(self.fft_window_selector, 0, 7)

        self.fft_live_checkbox = QCheckBox("LIVE")
        self.fft_live_checkbox.toggled.connect(self.on_fft_live_toggled)
        fft_controls.addWidget(self.fft_live_checkbox, 0, 8)

        self.fft_use_playhead_button = QPushButton("Pozycja odtwarzania")
        self.fft_use_playhead_button.clicked.connect(self.use_playhead_for_fft)
        fft_controls.addWidget(self.fft_use_playhead_button, 0, 9)

        self.fft_refresh_button = QPushButton("Odswiez FFT")
        self.fft_refresh_button.clicked.connect(self.update_fft_tab)
        fft_controls.addWidget(self.fft_refresh_button, 0, 10)
        fft_layout.addLayout(fft_controls)

        fft_plots_layout = QGridLayout()
        self.fft_signal_plot = LinePlotWidget(
            "Fragment sygnalu",
            "#335c88",
            230,
            x_axis_label="Czas lokalny [s]",
            y_axis_label="Amplituda",
        )
        self.fft_windowed_signal_plot = LinePlotWidget(
            "Fragment po oknie",
            "#c98142",
            230,
            x_axis_label="Czas lokalny [s]",
            y_axis_label="Amplituda",
        )
        self.fft_raw_spectrum_plot = LinePlotWidget(
            "Widmo FFT bez okna",
            "#376fa0",
            230,
            x_unit="Hz",
            x_decimals=0,
            y_decimals=1,
            x_axis_label="Czestotliwosc [Hz]",
            y_axis_label="Magnituda [dB rel.]",
        )
        self.fft_windowed_spectrum_plot = LinePlotWidget(
            "Widmo FFT po oknie",
            "#2e8c93",
            230,
            x_unit="Hz",
            x_decimals=0,
            y_decimals=1,
            x_axis_label="Czestotliwosc [Hz]",
            y_axis_label="Magnituda [dB rel.]",
        )
        fft_plots_layout.addWidget(self.create_scaled_plot_panel(self.fft_signal_plot, "fft_signal"), 0, 0)
        fft_plots_layout.addWidget(self.create_scaled_plot_panel(self.fft_windowed_signal_plot, "fft_windowed_signal"), 0, 1)
        fft_plots_layout.addWidget(self.create_scaled_plot_panel(self.fft_raw_spectrum_plot, "fft_raw_spectrum"), 1, 0)
        fft_plots_layout.addWidget(self.create_scaled_plot_panel(self.fft_windowed_spectrum_plot, "fft_windowed_spectrum"), 1, 1)
        fft_layout.addLayout(fft_plots_layout)

        self.fft_details_text = QPlainTextEdit()
        self.fft_details_text.setReadOnly(True)
        self.fft_details_text.setMaximumHeight(150)
        fft_layout.addWidget(self.fft_details_text)
        self.tabs.addTab(self.fft_tab, "FFT i okna")

        spectrogram_tab = QWidget()
        spectrogram_layout = QVBoxLayout(spectrogram_tab)
        spectrogram_controls = QHBoxLayout()
        spectrogram_controls.addWidget(QLabel("Okno:"))
        self.spectrogram_window_selector = QComboBox()
        self.populate_window_selector(self.spectrogram_window_selector, default_name="Hann")
        spectrogram_controls.addWidget(self.spectrogram_window_selector)
        spectrogram_controls.addWidget(QLabel("Ramka [ms]:"))
        self.spectrogram_frame_input = QLineEdit("40")
        self.spectrogram_frame_input.setMaximumWidth(90)
        spectrogram_controls.addWidget(self.spectrogram_frame_input)
        spectrogram_controls.addWidget(QLabel("Overlap [%]:"))
        self.spectrogram_overlap_input = QLineEdit("50")
        self.spectrogram_overlap_input.setMaximumWidth(90)
        spectrogram_controls.addWidget(self.spectrogram_overlap_input)
        spectrogram_controls.addWidget(QLabel("Max Hz:"))
        self.spectrogram_max_frequency_input = QLineEdit("8000")
        self.spectrogram_max_frequency_input.setMaximumWidth(90)
        spectrogram_controls.addWidget(self.spectrogram_max_frequency_input)
        self.spectrogram_refresh_button = QPushButton("Generuj spektrogram")
        self.spectrogram_refresh_button.clicked.connect(self.update_spectrogram_tab)
        spectrogram_controls.addWidget(self.spectrogram_refresh_button)
        spectrogram_controls.addStretch(1)
        spectrogram_layout.addLayout(spectrogram_controls)

        self.spectrogram_widget = SpectrogramWidget("Spektrogram STFT")
        spectrogram_layout.addWidget(self.spectrogram_widget)
        self.spectrogram_info_label = QLabel("Kliknij Generuj spektrogram, aby obliczyc widok STFT.")
        self.spectrogram_info_label.setWordWrap(True)
        spectrogram_layout.addWidget(self.spectrogram_info_label)
        self.tabs.addTab(spectrogram_tab, "Spektrogram")

        self.cepstrum_tab = QWidget()
        cepstrum_layout = QVBoxLayout(self.cepstrum_tab)
        cepstrum_controls = QGridLayout()
        cepstrum_controls.addWidget(QLabel("Start [s]:"), 0, 0)
        self.cepstrum_start_input = QLineEdit("0.0")
        self.cepstrum_start_input.setMaximumWidth(90)
        cepstrum_controls.addWidget(self.cepstrum_start_input, 0, 1)

        cepstrum_controls.addWidget(QLabel("Dlugosc [ms]:"), 0, 2)
        self.cepstrum_duration_input = QLineEdit("40")
        self.cepstrum_duration_input.setMaximumWidth(90)
        cepstrum_controls.addWidget(self.cepstrum_duration_input, 0, 3)

        cepstrum_controls.addWidget(QLabel("Okno:"), 0, 4)
        self.cepstrum_window_selector = QComboBox()
        self.populate_window_selector(self.cepstrum_window_selector)
        cepstrum_controls.addWidget(self.cepstrum_window_selector, 0, 5)

        self.cepstrum_live_checkbox = QCheckBox("LIVE")
        self.cepstrum_live_checkbox.toggled.connect(self.on_cepstrum_live_toggled)
        cepstrum_controls.addWidget(self.cepstrum_live_checkbox, 0, 6)

        self.cepstrum_use_playhead_button = QPushButton("Pozycja odtwarzania")
        self.cepstrum_use_playhead_button.clicked.connect(self.use_playhead_for_cepstrum)
        cepstrum_controls.addWidget(self.cepstrum_use_playhead_button, 0, 7)

        self.cepstrum_refresh_button = QPushButton("Odswiez cepstrum")
        self.cepstrum_refresh_button.clicked.connect(self.update_cepstrum_tab)
        cepstrum_controls.addWidget(self.cepstrum_refresh_button, 0, 8)
        cepstrum_layout.addLayout(cepstrum_controls)

        cepstrum_plots_layout = QGridLayout()
        self.cepstrum_signal_plot = LinePlotWidget(
            "Analizowany fragment",
            "#335c88",
            220,
            x_axis_label="Czas lokalny fragmentu [s]",
            y_axis_label="Amplituda",
        )
        self.cepstrum_plot = LinePlotWidget(
            "Cepstrum rzeczywiste",
            "#7a5ba4",
            220,
            x_unit="ms",
            x_decimals=2,
            y_decimals=3,
            x_axis_label="Quefrency [ms]",
            y_axis_label="Amplituda cepstrum",
        )
        cepstrum_plots_layout.addWidget(self.create_scaled_plot_panel(self.cepstrum_signal_plot, "cepstrum_signal"), 0, 0)
        cepstrum_plots_layout.addWidget(self.create_scaled_plot_panel(self.cepstrum_plot, "cepstrum_quefrency"), 0, 1)
        cepstrum_layout.addLayout(cepstrum_plots_layout)

        self.cepstrum_f0_plot = LinePlotWidget(
            "F0 z cepstrum w czasie",
            "#a55454",
            260,
            x_unit="s",
            x_decimals=2,
            y_decimals=1,
            x_axis_label="Czas w pliku [s]",
            y_axis_label="F0 [Hz]",
        )
        cepstrum_layout.addWidget(self.cepstrum_f0_plot)

        self.cepstrum_details_text = QPlainTextEdit()
        self.cepstrum_details_text.setReadOnly(True)
        self.cepstrum_details_text.setMaximumHeight(120)
        cepstrum_layout.addWidget(self.cepstrum_details_text)
        self.tabs.addTab(self.cepstrum_tab, "Cepstrum")

        main_layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.on_fft_mode_changed()
        self.sync_cepstrum_controls_state()
        self.configure_fixed_live_plot_ranges()
        self.set_playback_controls_enabled(False)

    def configure_fixed_live_plot_ranges(self) -> None:
        self.fft_signal_plot.set_fixed_y_range(-0.30, 0.30, anchor="center")
        self.fft_windowed_signal_plot.set_fixed_y_range(-0.30, 0.30, anchor="center")
        self.fft_raw_spectrum_plot.set_fixed_y_range(-70.0, 0.0, anchor="max")
        self.fft_windowed_spectrum_plot.set_fixed_y_range(-70.0, 0.0, anchor="max")
        self.cepstrum_signal_plot.set_fixed_y_range(-0.30, 0.30, anchor="center")
        self.cepstrum_plot.set_fixed_y_range(0.0, 0.10, anchor="min")

    def populate_window_selector(self, combo_box: QComboBox, default_name: str = "Hamming") -> None:
        window_names = ["Prostokatne", "Trojkatne", "Hamming", "Hann", "Blackman"]
        for name in window_names:
            combo_box.addItem(name)

        default_index = combo_box.findText(default_name, Qt.MatchFixedString)
        combo_box.setCurrentIndex(max(0, default_index))

    def parse_float_input(
        self,
        input_widget: QLineEdit,
        field_name: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float | None:
        try:
            value = float(input_widget.text().replace(",", "."))
        except ValueError:
            self.show_error("Err", f"{field_name} musi byc liczba.")
            return None

        if minimum is not None and value < minimum:
            self.show_error("Err", f"{field_name} musi byc >= {minimum}.")
            return None

        if maximum is not None and value > maximum:
            self.show_error("Err", f"{field_name} musi byc <= {maximum}.")
            return None

        return value

    def build_xy_series(
        self,
        x_values,
        y_values,
        max_points: int = 12000,
    ) -> tuple[list[float], list[float]]:
        if len(x_values) == 0 or len(y_values) == 0:
            return [0.0], [0.0]

        step = max(1, math.ceil(len(y_values) / max_points))
        reduced_x = []
        reduced_y = []

        for index in range(0, len(y_values), step):
            reduced_x.append(float(x_values[index]))
            reduced_y.append(float(y_values[index]))

        if reduced_x[-1] != float(x_values[-1]):
            reduced_x.append(float(x_values[-1]))
            reduced_y.append(float(y_values[-1]))

        return reduced_x, reduced_y

    def clear_project2_views(self) -> None:
        self.fft_snapshot = None
        self.cepstrum_snapshot = None
        self.spectrogram_data = None
        self.fft_signal_plot.clear_plot()
        self.fft_windowed_signal_plot.clear_plot()
        self.fft_raw_spectrum_plot.clear_plot()
        self.fft_windowed_spectrum_plot.clear_plot()
        self.fft_details_text.clear()
        self.spectrogram_widget.clear_spectrogram()
        self.spectrogram_info_label.setText("Kliknij Generuj spektrogram, aby obliczyc widok STFT.")
        self.cepstrum_signal_plot.clear_plot()
        self.cepstrum_plot.clear_plot()
        self.cepstrum_f0_plot.clear_plot()
        self.cepstrum_details_text.clear()

    def sync_fft_controls_state(self) -> None:
        use_full_signal = self.fft_domain_selector.currentData() == "full"
        live_active = self.fft_live_checkbox.isChecked()
        self.fft_domain_selector.setEnabled(self.analysis_controls_enabled)
        self.fft_window_selector.setEnabled(self.analysis_controls_enabled)
        self.fft_duration_input.setEnabled(self.analysis_controls_enabled and not use_full_signal)
        self.fft_live_checkbox.setEnabled(self.analysis_controls_enabled and not use_full_signal)
        self.fft_start_input.setEnabled(self.analysis_controls_enabled and not use_full_signal and not live_active)
        self.fft_use_playhead_button.setEnabled(self.analysis_controls_enabled and not use_full_signal and not live_active)
        self.fft_refresh_button.setEnabled(self.analysis_controls_enabled)

    def sync_cepstrum_controls_state(self) -> None:
        live_active = self.cepstrum_live_checkbox.isChecked()
        self.cepstrum_window_selector.setEnabled(self.analysis_controls_enabled)
        self.cepstrum_duration_input.setEnabled(self.analysis_controls_enabled)
        self.cepstrum_live_checkbox.setEnabled(self.analysis_controls_enabled)
        self.cepstrum_start_input.setEnabled(self.analysis_controls_enabled and not live_active)
        self.cepstrum_use_playhead_button.setEnabled(self.analysis_controls_enabled and not live_active)
        self.cepstrum_refresh_button.setEnabled(self.analysis_controls_enabled)

    def on_fft_mode_changed(self, *_args) -> None:
        use_full_signal = self.fft_domain_selector.currentData() == "full"
        if use_full_signal and self.fft_live_checkbox.isChecked():
            self.fft_live_checkbox.blockSignals(True)
            self.fft_live_checkbox.setChecked(False)
            self.fft_live_checkbox.blockSignals(False)
        self.sync_fft_controls_state()

    def on_fft_live_toggled(self, checked: bool) -> None:
        self.last_fft_live_update_ms = -1_000_000
        self.sync_fft_controls_state()
        if checked:
            self.refresh_fft_live_view(self.player.position(), force=True)

    def use_playhead_for_fft(self) -> None:
        self.fft_start_input.setText(f"{self.player.position() / 1000.0:.3f}")
        if self.fft_domain_selector.currentData() != "full":
            self.update_fft_tab()

    def on_cepstrum_live_toggled(self, checked: bool) -> None:
        self.last_cepstrum_live_update_ms = -1_000_000
        self.sync_cepstrum_controls_state()
        if checked:
            self.refresh_cepstrum_live_view(self.player.position(), force=True)

    def use_playhead_for_cepstrum(self) -> None:
        self.cepstrum_start_input.setText(f"{self.player.position() / 1000.0:.3f}")
        self.update_cepstrum_tab()

    def refresh_fft_live_view(self, position_ms: int, force: bool = False) -> None:
        if (
            self.audio_data is None
            or not self.analysis_controls_enabled
            or not self.fft_live_checkbox.isChecked()
            or self.fft_domain_selector.currentData() == "full"
        ):
            return
        if not force and abs(position_ms - self.last_fft_live_update_ms) < self.live_update_interval_ms:
            return

        self.last_fft_live_update_ms = position_ms
        self.fft_start_input.setText(f"{position_ms / 1000.0:.3f}")
        self.update_fft_tab()

    def refresh_cepstrum_live_view(self, position_ms: int, force: bool = False) -> None:
        if self.audio_data is None or not self.analysis_controls_enabled or not self.cepstrum_live_checkbox.isChecked():
            return
        if not force and abs(position_ms - self.last_cepstrum_live_update_ms) < self.live_update_interval_ms:
            return

        self.last_cepstrum_live_update_ms = position_ms
        self.cepstrum_start_input.setText(f"{position_ms / 1000.0:.3f}")
        self.update_cepstrum_tab()

    def update_live_analysis_views(self, position_ms: int, force: bool = False) -> None:
        if self.audio_data is None or not self.analysis_controls_enabled:
            return
        if not force and self.player.state() != QMediaPlayer.PlayingState:
            return

        current_tab = self.tabs.currentWidget()
        if self.fft_live_checkbox.isChecked() and current_tab is self.fft_tab:
            self.refresh_fft_live_view(position_ms, force=force)
        if self.cepstrum_live_checkbox.isChecked() and current_tab is self.cepstrum_tab:
            self.refresh_cepstrum_live_view(position_ms, force=force)

    def on_tab_changed(self, *_args) -> None:
        self.update_live_analysis_views(self.player.position(), force=True)

    def update_fft_tab(self) -> None:
        self.fft_snapshot = None
        if self.audio_data is None:
            self.fft_signal_plot.clear_plot()
            self.fft_windowed_signal_plot.clear_plot()
            self.fft_raw_spectrum_plot.clear_plot()
            self.fft_windowed_spectrum_plot.clear_plot()
            self.fft_details_text.clear()
            return

        use_full_signal = self.fft_domain_selector.currentData() == "full"
        start_time = 0.0
        duration_seconds = None

        if not use_full_signal:
            start_value = self.parse_float_input(self.fft_start_input, "Start", minimum=0.0)
            duration_ms = self.parse_float_input(self.fft_duration_input, "Dlugosc", minimum=5.0)
            if start_value is None or duration_ms is None:
                return
            start_time = start_value
            duration_seconds = duration_ms / 1000.0

        snapshot = compute_spectrum_snapshot(
            self.audio_data,
            start_time=start_time,
            duration_seconds=duration_seconds,
            window_name=self.fft_window_selector.currentText(),
        )
        self.fft_snapshot = snapshot

        if use_full_signal:
            time_scale = 1.0
            time_unit = "s"
            time_decimals = 2
        else:
            time_scale = 1000.0
            time_unit = "ms"
            time_decimals = 2

        raw_times = snapshot.time_axis * time_scale
        raw_time_x, raw_time_y = self.build_xy_series(raw_times, snapshot.raw_samples, max_points=8000)
        windowed_time_x, windowed_time_y = self.build_xy_series(raw_times, snapshot.windowed_samples, max_points=8000)
        spectrum_x, raw_spectrum_y = self.build_xy_series(snapshot.frequencies, snapshot.raw_magnitude_db, max_points=6000)
        _window_spectrum_x, window_spectrum_y = self.build_xy_series(snapshot.frequencies, snapshot.windowed_magnitude_db, max_points=6000)

        self.fft_signal_plot.title = "Fragment sygnalu (czas lokalny)" if not use_full_signal else "Sygnal caly"
        self.fft_signal_plot.set_axis_format(time_unit, time_decimals, 3)
        self.fft_signal_plot.set_axis_labels(
            "Czas lokalny fragmentu [ms]" if not use_full_signal else "Czas w analizowanym sygnale [s]",
            "Amplituda",
        )
        self.fft_signal_plot.set_plot_data(raw_time_x, raw_time_y)

        self.fft_windowed_signal_plot.title = "Fragment po oknie (czas lokalny)" if not use_full_signal else "Sygnal po oknie"
        self.fft_windowed_signal_plot.set_axis_format(time_unit, time_decimals, 3)
        self.fft_windowed_signal_plot.set_axis_labels(
            "Czas lokalny fragmentu [ms]" if not use_full_signal else "Czas w analizowanym sygnale [s]",
            "Amplituda",
        )
        self.fft_windowed_signal_plot.set_plot_data(windowed_time_x, windowed_time_y)

        self.fft_raw_spectrum_plot.set_axis_format("Hz", 0, 1)
        self.fft_windowed_spectrum_plot.set_axis_format("Hz", 0, 1)
        self.fft_raw_spectrum_plot.set_axis_labels("Czestotliwosc [Hz]", "Magnituda [dB rel.]")
        self.fft_windowed_spectrum_plot.set_axis_labels("Czestotliwosc [Hz]", "Magnituda [dB rel.]")
        self.fft_raw_spectrum_plot.set_plot_data(spectrum_x, raw_spectrum_y)
        self.fft_windowed_spectrum_plot.set_plot_data(spectrum_x, window_spectrum_y)

        lines = [
            "Parametry analizy FFT",
            "",
            f"Zakres: {'caly sygnal' if use_full_signal else 'ramka'}",
            f"Start: {snapshot.start_time:.3f} s",
            f"Dlugosc: {snapshot.duration_seconds * 1000.0:.2f} ms",
            f"Okno: {self.fft_window_selector.currentText()}",
            "",
            f"Centroid widmowy: {snapshot.spectral_centroid:.2f} Hz",
            f"Bandwidth efektywny: {snapshot.effective_bandwidth:.2f} Hz",
            f"ERSB1: {snapshot.band_ratios[0]:.4f}",
            f"ERSB2: {snapshot.band_ratios[1]:.4f}",
            f"ERSB3: {snapshot.band_ratios[2]:.4f}",
            f"ERSB4: {snapshot.band_ratios[3]:.4f}",
            f"Spectral Flatness: {snapshot.spectral_flatness:.4f}",
            f"Spectral Crest: {snapshot.spectral_crest:.4f}",
            f"F0 z cepstrum: {snapshot.f0_cepstrum:.2f} Hz",
        ]
        self.fft_details_text.setPlainText("\n".join(lines))

    def update_spectrogram_tab(self) -> None:
        if self.audio_data is None:
            self.spectrogram_data = None
            self.spectrogram_widget.clear_spectrogram()
            self.spectrogram_info_label.setText("Brak pliku WAV do analizy.")
            return

        frame_ms = self.parse_float_input(self.spectrogram_frame_input, "Ramka spektrogramu", minimum=5.0)
        overlap_percent = self.parse_float_input(
            self.spectrogram_overlap_input,
            "Overlap spektrogramu",
            minimum=0.0,
            maximum=95.0,
        )
        if frame_ms is None or overlap_percent is None:
            return

        max_frequency = self.parse_float_input(
            self.spectrogram_max_frequency_input,
            "Max Hz",
            minimum=100.0,
            maximum=max(100.0, self.audio_data.sample_rate / 2.0),
        )
        if max_frequency is None:
            return

        self.spectrogram_data = compute_spectrogram(
            self.audio_data,
            frame_ms=frame_ms,
            overlap_percent=overlap_percent,
            window_name=self.spectrogram_window_selector.currentText(),
            max_frequency_hz=max_frequency,
        )
        self.spectrogram_widget.set_spectrogram_data(
            self.spectrogram_data.times,
            self.spectrogram_data.frequencies,
            self.spectrogram_data.magnitude_db,
        )
        self.spectrogram_widget.set_playhead_time(self.player.position() / 1000.0)
        self.apply_view_range_to_widgets()
        self.spectrogram_info_label.setText(
            f"Spektrogram: okno={self.spectrogram_window_selector.currentText()}, "
            f"ramka={frame_ms:.1f} ms, overlap={overlap_percent:.1f}%, "
            f"max={max_frequency:.0f} Hz, kolumn={self.spectrogram_data.magnitude_db.shape[1]}."
        )

    def update_cepstrum_tab(self) -> None:
        self.cepstrum_snapshot = None
        if self.audio_data is None:
            self.cepstrum_signal_plot.clear_plot()
            self.cepstrum_plot.clear_plot()
            self.cepstrum_f0_plot.clear_plot()
            self.cepstrum_details_text.clear()
            return

        start_value = self.parse_float_input(self.cepstrum_start_input, "Start cepstrum", minimum=0.0)
        duration_ms = self.parse_float_input(self.cepstrum_duration_input, "Dlugosc cepstrum", minimum=5.0)
        if start_value is None or duration_ms is None:
            return
        if start_value > self.audio_data.duration_seconds:
            self.show_error(
                "Err",
                f"Start cepstrum wykracza poza dlugosc pliku ({self.audio_data.duration_seconds:.3f} s).",
            )
            return

        cepstrum_reference_frequency = None
        if self.analysis_result is not None and self.analysis_result.frames:
            closest_frame = min(
                self.analysis_result.frames,
                key=lambda frame: abs(frame.start_time - start_value),
            )
            reference_values = [
                value
                for value in [closest_frame.f0_autocorrelation, closest_frame.f0_amdf]
                if value > 0.0
            ]
            if reference_values:
                cepstrum_reference_frequency = sum(reference_values) / len(reference_values)

        snapshot = compute_spectrum_snapshot(
            self.audio_data,
            start_time=start_value,
            duration_seconds=duration_ms / 1000.0,
            window_name=self.cepstrum_window_selector.currentText(),
            cepstrum_reference_frequency=cepstrum_reference_frequency,
            cepstrum_min_frequency=50.0,
            cepstrum_max_frequency=400.0,
        )
        self.cepstrum_snapshot = snapshot

        if snapshot.duration_seconds <= 0.12:
            signal_times = snapshot.time_axis * 1000.0
            signal_unit = "ms"
        else:
            signal_times = snapshot.time_axis
            signal_unit = "s"

        signal_x, signal_y = self.build_xy_series(signal_times, snapshot.raw_samples, max_points=8000)
        cepstrum_mask = (snapshot.cepstrum_quefrencies_ms >= 2.5) & (snapshot.cepstrum_quefrencies_ms <= 20.0)
        if np.any(cepstrum_mask):
            cepstrum_source_x = snapshot.cepstrum_quefrencies_ms[cepstrum_mask]
            cepstrum_source_y = snapshot.cepstrum_values[cepstrum_mask]
        else:
            cepstrum_source_x = snapshot.cepstrum_quefrencies_ms
            cepstrum_source_y = snapshot.cepstrum_values
        cepstrum_x, cepstrum_y = self.build_xy_series(
            cepstrum_source_x,
            cepstrum_source_y,
            max_points=4000,
        )

        self.cepstrum_signal_plot.set_axis_format(signal_unit, 2, 3)
        self.cepstrum_signal_plot.title = "Analizowany fragment (czas lokalny)"
        self.cepstrum_signal_plot.set_axis_labels(
            "Czas lokalny fragmentu [ms]" if signal_unit == "ms" else "Czas lokalny fragmentu [s]",
            "Amplituda",
        )
        self.cepstrum_signal_plot.set_plot_data(signal_x, signal_y)
        self.cepstrum_plot.set_axis_format("ms", 2, 3)
        self.cepstrum_plot.title = "Cepstrum rzeczywiste (zakres F0)"
        self.cepstrum_plot.set_axis_labels("Quefrency [ms]", "Amplituda cepstrum")
        self.cepstrum_plot.set_plot_data(cepstrum_x, cepstrum_y)

        if snapshot.f0_cepstrum > 0.0:
            f0_text = f"{snapshot.f0_cepstrum:.2f} Hz"
        else:
            f0_text = "brak stabilnego maksimum"

        details_lines = [
            "Analiza cepstralna",
            "",
            f"Start: {snapshot.start_time:.3f} s",
            f"Dlugosc: {snapshot.duration_seconds * 1000.0:.2f} ms",
            f"Okno: {self.cepstrum_window_selector.currentText()}",
            f"F0 z cepstrum: {f0_text}",
            f"Zakres szukania maksimum: 50-400 Hz (na wykresie pokazany zakres 2.5-20 ms)",
        ]
        self.cepstrum_details_text.setPlainText("\n".join(details_lines))

        if self.analysis_result is not None:
            frame_times = [frame.start_time for frame in self.analysis_result.frames]
            f0_values = [frame.f0_cepstrum for frame in self.analysis_result.frames]
            self.cepstrum_f0_plot.set_axis_format("s", 2, 1)
            self.cepstrum_f0_plot.set_axis_labels("Czas w pliku [s]", "F0 [Hz]")
            self.cepstrum_f0_plot.set_plot_data(frame_times, f0_values)
            self.cepstrum_f0_plot.set_playhead_time(self.player.position() / 1000.0)
            self.apply_view_range_to_widgets()
        else:
            self.cepstrum_f0_plot.clear_plot()

    def open_wav_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik WAV", "", "Pliki WAV")
        if not file_path:
            return

        self.load_audio_file(file_path)

    def load_audio_file(self, file_path: str) -> None:
        try:
            self.audio_data = load_wav_file(file_path)
        except Exception as error:
            self.show_error("Nie udalo sie wczytac pliku", str(error))
            return

        self.analysis_result = None
        self.spectrogram_data = None
        self.frames_table.setRowCount(0)
        self.summary_text.setPlainText("")
        self.feature_plot.clear_plot()
        self.voicing_timeline.clear_timeline()
        self.speech_music_timeline.clear_timeline()
        self.clear_project2_views()

        waveform_times, waveform_values = self.build_waveform_series(self.audio_data.samples, self.audio_data.sample_rate)
        self.waveform_plot.set_plot_data(waveform_times, waveform_values, [])

        media_url = QUrl.fromLocalFile(file_path)
        self.player.setMedia(QMediaContent(media_url))
        self.player.stop()
        self.position_slider.setRange(0, int(self.audio_data.duration_seconds * 1000))
        self.position_slider.setValue(0)
        self.position_label.setText(f"00:00.0 / {self.format_milliseconds(int(self.audio_data.duration_seconds * 1000))}")
        self.reset_view_range()
        self.update_playhead_visuals(0.0)
        self.set_playback_controls_enabled(True)
        self.fft_start_input.setText("0.0")
        self.cepstrum_start_input.setText("0.0")
        default_max_frequency = min(8000.0, self.audio_data.sample_rate / 2.0)
        self.spectrogram_max_frequency_input.setText(f"{default_max_frequency:.0f}")
        self.update_fft_tab()
        self.update_cepstrum_tab()

        self.info_label.setText(
            f"Plik: {file_path} | fs={self.audio_data.sample_rate} Hz | "
            f"kanaly={self.audio_data.channels} | czas={self.audio_data.duration_seconds:.3f} s | "
            "kliknij Analizuj"
        )

    def analyze_current_audio(self) -> None:
        if self.audio_data is None:
            self.show_error("Brak pliku", "Wczytaj plik WAV.")
            return

        try:
            frame_ms = float(self.frame_input.text().replace(",", "."))
            hop_ms = float(self.hop_input.text().replace(",", "."))
        except ValueError:
            self.show_error("Err", "Frame i hop musza byc liczbami.")
            return

        if frame_ms <= 0.0 or hop_ms <= 0.0:
            self.show_error("Err", "Frame i hop musza byc dodatnie.")
            return

        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            return

        self.set_controls_enabled(False)
        self.info_label.setText("Trwa analiza")
        self.analysis_thread = AnalysisThread(self.audio_data, frame_ms, hop_ms)
        self.analysis_thread.analysis_finished.connect(self.on_analysis_finished)
        self.analysis_thread.analysis_failed.connect(self.on_analysis_failed)
        self.analysis_thread.finished.connect(self.on_analysis_thread_stopped)
        self.analysis_thread.start()

    def on_analysis_finished(self, result) -> None:
        self.analysis_result = result
        self.info_label.setText(
            f"Analiza zakonczona: plik: {self.audio_data.path} | "
            f"fs={self.audio_data.sample_rate} Hz, fs analizy={self.analysis_result.analysis_sample_rate} Hz"
        )
        self.update_plots()
        self.update_summary()
        self.update_table()
        self.update_cepstrum_tab()

    def on_analysis_failed(self, message: str) -> None:
        self.show_error("Blad analizy", message)

    def on_analysis_thread_stopped(self) -> None:
        self.set_controls_enabled(True)
        if self.analysis_thread is not None:
            self.analysis_thread.deleteLater()
            self.analysis_thread = None

    def update_plots(self) -> None:
        if self.analysis_result is None:
            return

        audio = self.analysis_result.audio_data
        waveform_times, waveform_values = self.build_waveform_series(audio.samples, audio.sample_rate)

        silence_segments = []
        for start_time, end_time, label in self.analysis_result.voicing_segments:
            if label == "silence":
                silence_segments.append((start_time, end_time, "#b9b9b9"))

        self.waveform_plot.set_plot_data(waveform_times, waveform_values, silence_segments)
        self.voicing_timeline.set_timeline_data(self.analysis_result.voicing_segments, audio.duration_seconds)
        self.speech_music_timeline.set_timeline_data(self.analysis_result.speech_music_segments, audio.duration_seconds)
        self.update_feature_plot()
        self.apply_view_range_to_widgets()
        self.update_playhead_visuals(self.player.position() / 1000.0)

    def update_feature_plot(self, *_args) -> None:
        if self.analysis_result is None:
            self.feature_plot.clear_plot()
            return

        frame_times = [frame.start_time for frame in self.analysis_result.frames]
        selected_name = self.feature_selector.currentText()

        if selected_name == "Volume":
            values = [frame.normalized_volume for frame in self.analysis_result.frames]
            color = "#5f9e6e"
        elif selected_name == "STE":
            values = [frame.ste for frame in self.analysis_result.frames]
            color = "#c98142"
        elif selected_name == "ZCR":
            values = [frame.zcr for frame in self.analysis_result.frames]
            color = "#a55454"
        elif selected_name == "F0 autokorelacja":
            values = [frame.f0_autocorrelation for frame in self.analysis_result.frames]
            color = "#376fa0"
        elif selected_name == "F0 AMDF":
            values = [frame.f0_amdf for frame in self.analysis_result.frames]
            color = "#7a5ba4"
            y_axis_label = "F0 [Hz]"
        elif selected_name == "Dominujaca czestotliwosc FFT":
            values = [frame.dominant_frequency_fft for frame in self.analysis_result.frames]
            color = "#2e8c93"
            y_axis_label = "Czestotliwosc [Hz]"
        elif selected_name == "Centroid widmowy":
            values = [frame.spectral_centroid for frame in self.analysis_result.frames]
            color = "#376fa0"
            y_axis_label = "Czestotliwosc [Hz]"
        elif selected_name == "Bandwidth efektywny":
            values = [frame.effective_bandwidth for frame in self.analysis_result.frames]
            color = "#c98142"
            y_axis_label = "Czestotliwosc [Hz]"
        elif selected_name == "ERSB1":
            values = [frame.ersb1 for frame in self.analysis_result.frames]
            color = "#698f3f"
            y_axis_label = "Udzial energii [-]"
        elif selected_name == "ERSB2":
            values = [frame.ersb2 for frame in self.analysis_result.frames]
            color = "#4c7e95"
            y_axis_label = "Udzial energii [-]"
        elif selected_name == "ERSB3":
            values = [frame.ersb3 for frame in self.analysis_result.frames]
            color = "#9a6a3a"
            y_axis_label = "Udzial energii [-]"
        elif selected_name == "Spectral Flatness":
            values = [frame.spectral_flatness for frame in self.analysis_result.frames]
            color = "#9350a4"
            y_axis_label = "Miara plaskosci [-]"
        elif selected_name == "Spectral Crest":
            values = [frame.spectral_crest for frame in self.analysis_result.frames]
            color = "#b35757"
            y_axis_label = "Wspolczynnik grzebietu [-]"
        else:
            values = [frame.f0_cepstrum for frame in self.analysis_result.frames]
            color = "#c95d63"
            y_axis_label = "F0 [Hz]"

        if selected_name == "Volume":
            y_axis_label = "Glosnosc wzgledna [-]"
        elif selected_name == "STE":
            y_axis_label = "Energia krotkoczasowa"
        elif selected_name == "ZCR":
            y_axis_label = "ZCR [-]"
        elif selected_name == "F0 autokorelacja":
            y_axis_label = "F0 [Hz]"

        self.feature_plot.title = selected_name
        self.feature_plot.line_color = QColor(color)
        self.feature_plot.set_axis_format("s", 2, 3 if "F0" not in selected_name and "FFT" not in selected_name and "Centroid" not in selected_name and "Bandwidth" not in selected_name else 1)
        self.feature_plot.set_axis_labels("Czas w pliku [s]", y_axis_label)
        self.feature_plot.set_plot_data(frame_times, values)
        self.apply_view_range_to_widgets()
        self.feature_plot.set_playhead_time(self.player.position() / 1000.0)

    def update_summary(self) -> None:
        if self.analysis_result is None:
            self.summary_text.clear()
            return

        self.summary_text.setPlainText("\n".join(build_summary_lines(self.analysis_result)))

    def update_table(self) -> None:
        if self.analysis_result is None:
            self.frames_table.setRowCount(0)
            return

        frames = self.analysis_result.frames
        self.frames_table.setRowCount(len(frames))

        for row_index, frame in enumerate(frames):
            row_values = [
                str(frame.index),
                f"{frame.start_time:.3f}",
                f"{frame.end_time:.3f}",
                f"{frame.volume:.4f}",
                f"{frame.normalized_volume:.4f}",
                f"{frame.ste:.4f}",
                f"{frame.zcr:.4f}",
                str(frame.silent_flag),
                f"{frame.f0_autocorrelation:.2f}",
                f"{frame.f0_amdf:.2f}",
                f"{frame.dominant_frequency_fft:.2f}",
                f"{frame.spectral_centroid:.2f}",
                f"{frame.effective_bandwidth:.2f}",
                f"{frame.ersb1:.4f}",
                f"{frame.ersb2:.4f}",
                f"{frame.ersb3:.4f}",
                f"{frame.spectral_flatness:.4f}",
                f"{frame.spectral_crest:.4f}",
                f"{frame.f0_cepstrum:.2f}",
                frame.voicing_label,
                frame.speech_music_label,
            ]

            for column_index, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                if column_index == 19:
                    self.apply_label_color(item, frame.voicing_label)
                if column_index == 20:
                    self.apply_label_color(item, frame.speech_music_label)
                self.frames_table.setItem(row_index, column_index, item)

    def build_export_basename(self) -> str:
        if self.audio_data is None:
            return "audio"

        stem = os.path.splitext(os.path.basename(self.audio_data.path))[0]
        safe_characters = []
        for character in stem:
            if character.isalnum() or character in ("-", "_"):
                safe_characters.append(character)
            else:
                safe_characters.append("_")

        safe_name = "".join(safe_characters).strip("_")
        return safe_name or "audio"

    def refresh_project2_export_data(self) -> bool:
        self.update_fft_tab()
        if self.fft_snapshot is None:
            return False

        self.update_cepstrum_tab()
        if self.cepstrum_snapshot is None:
            return False

        self.spectrogram_data = None
        self.update_spectrogram_tab()
        if self.spectrogram_data is None:
            return False

        return True

    def export_project2_csv_bundle(self) -> None:
        if self.audio_data is None:
            self.show_error("Brak pliku", "Najpierw wczytaj plik WAV.")
            return
        if self.analysis_result is None:
            self.show_error("Brak analizy", "Najpierw uruchom Analiza, aby wyeksportowac pakiet P2.")
            return

        target_directory = QFileDialog.getExistingDirectory(self, "Wybierz folder pakietu CSV P2")
        if not target_directory:
            return

        if not self.refresh_project2_export_data():
            return

        export_basename = self.build_export_basename()
        bundle_directory = os.path.join(target_directory, f"{export_basename}_csv_p2")
        os.makedirs(bundle_directory, exist_ok=True)

        try:
            export_frames_to_csv(
                self.analysis_result,
                os.path.join(bundle_directory, f"{export_basename}_frame_features.csv"),
            )
            export_clip_features_to_csv(
                self.analysis_result,
                os.path.join(bundle_directory, f"{export_basename}_clip_features.csv"),
            )
            export_segments_to_csv(
                self.analysis_result.voicing_segments,
                os.path.join(bundle_directory, f"{export_basename}_voicing_segments.csv"),
                "voicing",
            )
            export_segments_to_csv(
                self.analysis_result.speech_music_segments,
                os.path.join(bundle_directory, f"{export_basename}_speech_music_segments.csv"),
                "speech_music",
            )
            export_snapshot_time_domain_to_csv(
                self.fft_snapshot,
                os.path.join(bundle_directory, f"{export_basename}_fft_snapshot_time.csv"),
                audio_path=self.audio_data.path,
            )
            export_snapshot_spectrum_to_csv(
                self.fft_snapshot,
                os.path.join(bundle_directory, f"{export_basename}_fft_snapshot_spectrum.csv"),
                audio_path=self.audio_data.path,
            )
            export_cepstrum_snapshot_to_csv(
                self.cepstrum_snapshot,
                os.path.join(bundle_directory, f"{export_basename}_cepstrum_snapshot.csv"),
                audio_path=self.audio_data.path,
            )
            export_frame_spectra_to_csv(
                self.analysis_result,
                os.path.join(bundle_directory, f"{export_basename}_frame_spectrum_long.csv"),
                audio_path=self.audio_data.path,
                window_name=self.fft_window_selector.currentText(),
            )
            export_frame_cepstra_to_csv(
                self.analysis_result,
                os.path.join(bundle_directory, f"{export_basename}_frame_cepstrum_long.csv"),
                audio_path=self.audio_data.path,
                window_name=self.cepstrum_window_selector.currentText(),
                segment_duration_ms=self.cepstrum_snapshot.duration_seconds * 1000.0,
                min_frequency_hz=50.0,
                max_frequency_hz=400.0,
            )
            export_spectrogram_to_csv(
                self.spectrogram_data,
                os.path.join(bundle_directory, f"{export_basename}_spectrogram_long.csv"),
                audio_path=self.audio_data.path,
            )
        except Exception as error:
            self.show_error("Blad zapisu", str(error))
            return

        self.info_label.setText(f"Zapisano pakiet CSV P2: {bundle_directory}")

    def export_csv(self) -> None:
        if self.analysis_result is None:
            self.show_error("Brak analizy", "Najpierw wczytaj plik WAV")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Zapisz CSV", "wyniki_ramek.csv", "Pliki CSV")
        if not file_path:
            return

        try:
            export_frames_to_csv(self.analysis_result, file_path)
        except Exception as error:
            self.show_error("Blad zapisu", str(error))
            return

        self.info_label.setText(f"Zapisano CSV: {file_path}")

    def export_txt(self) -> None:
        if self.analysis_result is None:
            self.show_error("Brak analizy", "Najpierw wczytaj plik WAV")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Zapisz TXT", "podsumowanie_analizy.txt", "Pliki TXT")
        if not file_path:
            return

        try:
            export_summary_to_txt(self.analysis_result, file_path)
        except Exception as error:
            self.show_error("Blad zapisu", str(error))
            return

        self.info_label.setText(f"Zapisano TXT: {file_path}")

    def build_waveform_series(self, samples, sample_rate: int, max_points: int = 12000) -> tuple[list[float], list[float]]:
        if sample_rate <= 0 or len(samples) == 0:
            return [0.0], [0.0]

        step = max(1, math.ceil(len(samples) / max_points))
        times = []
        values = []

        for index in range(0, len(samples), step):
            times.append(index / sample_rate)
            values.append(float(samples[index]))

        if times[-1] != (len(samples) - 1) / sample_rate:
            times.append((len(samples) - 1) / sample_rate)
            values.append(float(samples[-1]))

        return times, values

    def play_audio(self) -> None:
        if self.audio_data is None:
            return
        self.player.play()

    def pause_audio(self) -> None:
        if self.audio_data is None:
            return
        self.player.pause()

    def stop_audio(self) -> None:
        if self.audio_data is None:
            return
        self.player.stop()
        self.update_playhead_visuals(0.0)
        self.update_live_analysis_views(0, force=True)

    def on_player_position_changed(self, position_ms: int) -> None:
        if not self.position_slider_is_dragged:
            self.position_slider.setValue(position_ms)

        duration_ms = self.position_slider.maximum()
        self.position_label.setText(
            f"{self.format_milliseconds(position_ms)} / {self.format_milliseconds(duration_ms)}"
        )

        current_position_seconds = position_ms / 1000.0
        view_duration = self.get_view_duration_seconds()
        if (
            self.audio_data is not None
            and view_duration is not None
            and self.player.state() == QMediaPlayer.PlayingState
        ):
            current_end = self.current_view_start_seconds + view_duration
            if current_position_seconds < self.current_view_start_seconds or current_position_seconds > current_end:
                new_start = current_position_seconds - (view_duration * 0.5)
                self.set_view_start_seconds(new_start)

        self.update_playhead_visuals(current_position_seconds)
        self.update_live_analysis_views(position_ms)

    def on_player_duration_changed(self, duration_ms: int) -> None:
        if duration_ms <= 0:
            return
        self.position_slider.setRange(0, duration_ms)
        self.position_label.setText(
            f"{self.format_milliseconds(self.player.position())} / {self.format_milliseconds(duration_ms)}"
        )

    def on_player_state_changed(self, state: int) -> None:
        self.play_button.setEnabled(self.audio_data is not None and state != QMediaPlayer.PlayingState)
        self.pause_button.setEnabled(self.audio_data is not None and state == QMediaPlayer.PlayingState)
        self.stop_button.setEnabled(self.audio_data is not None)

    def on_position_slider_pressed(self) -> None:
        self.position_slider_is_dragged = True

    def on_position_slider_released(self) -> None:
        self.position_slider_is_dragged = False
        self.player.setPosition(self.position_slider.value())
        self.update_live_analysis_views(self.position_slider.value(), force=True)

    def on_position_slider_moved(self, value: int) -> None:
        duration_ms = self.position_slider.maximum()
        self.position_label.setText(
            f"{self.format_milliseconds(value)} / {self.format_milliseconds(duration_ms)}"
        )
        self.update_playhead_visuals(value / 1000.0)
        self.update_live_analysis_views(value, force=True)

    def update_playhead_visuals(self, position_seconds: float) -> None:
        self.waveform_plot.set_playhead_time(position_seconds)
        self.feature_plot.set_playhead_time(position_seconds)
        self.voicing_timeline.set_playhead_time(position_seconds)
        self.speech_music_timeline.set_playhead_time(position_seconds)
        self.spectrogram_widget.set_playhead_time(position_seconds)
        self.cepstrum_f0_plot.set_playhead_time(position_seconds)

    def on_zoom_changed(self, *_args) -> None:
        if self.audio_data is None:
            return

        view_duration = self.get_view_duration_seconds()
        if view_duration is None:
            self.current_view_start_seconds = 0.0
        else:
            suggested_start = (self.player.position() / 1000.0) - (view_duration * 0.5)
            self.current_view_start_seconds = suggested_start

        self.refresh_view_slider()
        self.set_view_start_seconds(self.current_view_start_seconds)

    def on_view_slider_changed(self, value: int) -> None:
        if self.audio_data is None:
            return
        self.set_view_start_seconds(value / 1000.0, update_slider=False)

    def reset_view_range(self) -> None:
        self.current_view_start_seconds = 0.0
        self.zoom_selector.setCurrentIndex(0)
        self.refresh_view_slider()
        self.apply_view_range_to_widgets()

    def get_view_duration_seconds(self) -> float | None:
        value = self.zoom_selector.currentData()
        if value is None:
            return None
        return float(value)

    def refresh_view_slider(self) -> None:
        if self.audio_data is None:
            self.view_slider.setEnabled(False)
            self.view_slider.setRange(0, 0)
            self.view_label.setText("Widok: caly plik")
            return

        view_duration = self.get_view_duration_seconds()
        total_duration = self.audio_data.duration_seconds

        if view_duration is None or view_duration >= total_duration:
            self.view_slider.setEnabled(False)
            self.view_slider.setRange(0, 0)
            self.view_slider.setValue(0)
            self.view_label.setText("Widok: caly plik")
            return

        max_start_ms = int(max(0.0, (total_duration - view_duration) * 1000.0))
        self.view_slider.setEnabled(True)
        self.view_slider.setRange(0, max_start_ms)
        self.view_slider.setValue(int(max(0.0, self.current_view_start_seconds) * 1000.0))

    def set_view_start_seconds(self, start_seconds: float, update_slider: bool = True) -> None:
        if self.audio_data is None:
            return

        total_duration = self.audio_data.duration_seconds
        view_duration = self.get_view_duration_seconds()

        if view_duration is None or view_duration >= total_duration:
            self.current_view_start_seconds = 0.0
            self.view_label.setText("Widok: caly plik")
            self.apply_view_range_to_widgets()
            return

        max_start = max(0.0, total_duration - view_duration)
        self.current_view_start_seconds = max(0.0, min(start_seconds, max_start))

        if update_slider:
            self.view_slider.blockSignals(True)
            self.view_slider.setValue(int(self.current_view_start_seconds * 1000.0))
            self.view_slider.blockSignals(False)

        view_end = self.current_view_start_seconds + view_duration
        self.view_label.setText(f"Widok: {self.current_view_start_seconds:.2f}s - {view_end:.2f}s")
        self.apply_view_range_to_widgets()

    def apply_view_range_to_widgets(self) -> None:
        start_time = None
        end_time = None

        if self.audio_data is not None:
            view_duration = self.get_view_duration_seconds()
            if view_duration is not None and view_duration < self.audio_data.duration_seconds:
                start_time = self.current_view_start_seconds
                end_time = self.current_view_start_seconds + view_duration

        self.waveform_plot.set_view_range(start_time, end_time)
        self.feature_plot.set_view_range(start_time, end_time)
        self.voicing_timeline.set_view_range(start_time, end_time)
        self.speech_music_timeline.set_view_range(start_time, end_time)
        self.spectrogram_widget.set_view_range(start_time, end_time)
        self.cepstrum_f0_plot.set_view_range(start_time, end_time)

    def format_milliseconds(self, value_ms: int) -> str:
        total_seconds = max(0.0, value_ms / 1000.0)
        minutes = int(total_seconds // 60)
        seconds = total_seconds - (minutes * 60)
        return f"{minutes:02d}:{seconds:04.1f}"

    def apply_label_color(self, item: QTableWidgetItem, label: str) -> None:
        colors = {
            "silence": "#d9d9d9",
            "voiced": "#cae4cd",
            "unvoiced": "#f1d7ac",
            "speech": "#cde4d3",
            "music": "#efcfb7",
            "mixed": "#cad8ec",
        }
        if label in colors:
            item.setBackground(QColor(colors[label]))

    def set_controls_enabled(self, enabled: bool) -> None:
        self.analysis_controls_enabled = enabled
        self.open_button.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        self.export_csv_button.setEnabled(enabled)
        self.export_txt_button.setEnabled(enabled)
        self.export_project2_csv_button.setEnabled(enabled)
        self.frame_input.setEnabled(enabled)
        self.hop_input.setEnabled(enabled)
        self.spectrogram_window_selector.setEnabled(enabled)
        self.spectrogram_frame_input.setEnabled(enabled)
        self.spectrogram_overlap_input.setEnabled(enabled)
        self.spectrogram_max_frequency_input.setEnabled(enabled)
        self.spectrogram_refresh_button.setEnabled(enabled)
        self.sync_fft_controls_state()
        self.sync_cepstrum_controls_state()

    def set_playback_controls_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(enabled)
        self.position_slider.setEnabled(enabled)
        if not enabled:
            self.view_slider.setEnabled(False)
        else:
            self.refresh_view_slider()

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.load_audio_file(sys.argv[1])

    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
