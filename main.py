import math
import os
import sys


VENDOR_DIR = os.path.join(os.path.dirname(__file__), ".vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
        QApplication,
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


from audio_features import analyze_audio, export_frames_to_csv, export_summary_to_txt, load_wav_file


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
    def __init__(self, title: str, line_color: str, height: int = 190, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.line_color = QColor(line_color)
        self.times = []
        self.values = []
        self.overlay_segments = []
        self.playhead_time = None
        self.view_start_time = None
        self.view_end_time = None
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_plot_data(self, times: list[float], values: list[float], overlay_segments: list[tuple[float, float, str]] | None = None) -> None:
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

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f6f1"))

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#222222"))
        painter.drawText(14, 20, self.title)

        plot_rect = QRectF(54, 32, max(60, self.width() - 68), max(60, self.height() - 56))
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
        painter.drawText(8, int(plot_rect.top()) + 8, f"{y_max:.3f}")
        painter.drawText(8, int(plot_rect.bottom()), f"{y_min:.3f}")
        painter.drawText(int(plot_rect.left()), self.height() - 8, f"{x_min:.2f} s")
        painter.drawText(int(plot_rect.right()) - 60, self.height() - 8, f"{x_max:.2f} s")

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

        timeline_rect = QRectF(14, 30, max(80, self.width() - 28), 30)
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
        painter.drawText(14, 76, "Legenda:")
        painter.drawText(int(timeline_rect.left()), 92, f"{view_start:.2f} s")
        painter.drawText(int(timeline_rect.right()) - 60, 92, f"{view_end:.2f} s")

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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.audio_data = None
        self.analysis_result = None
        self.analysis_thread = None
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(50)
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)
        self.player.stateChanged.connect(self.on_player_state_changed)
        self.position_slider_is_dragged = False
        self.current_view_start_seconds = 0.0

        self.setWindowTitle("Projekt 1")
        self.resize(1280, 860)
        self.build_ui()

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

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_audio)
        controls_layout.addWidget(self.play_button, 0, 4)

        self.pause_button = QPushButton("Pauza")
        self.pause_button.clicked.connect(self.pause_audio)
        controls_layout.addWidget(self.pause_button, 0, 5)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        controls_layout.addWidget(self.stop_button, 0, 6)

        controls_layout.addWidget(QLabel("Frame [ms]:"), 0, 7)
        self.frame_input = QLineEdit("20")
        self.frame_input.setMaximumWidth(80)
        controls_layout.addWidget(self.frame_input, 0, 8)

        controls_layout.addWidget(QLabel("Hop [ms]:"), 0, 9)
        self.hop_input = QLineEdit("10")
        self.hop_input.setMaximumWidth(80)
        controls_layout.addWidget(self.hop_input, 0, 10)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderPressed.connect(self.on_position_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_position_slider_released)
        self.position_slider.sliderMoved.connect(self.on_position_slider_moved)
        controls_layout.addWidget(self.position_slider, 1, 0, 1, 10)

        self.position_label = QLabel("00:00.0 / 00:00.0")
        controls_layout.addWidget(self.position_label, 1, 10)

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
        controls_layout.addWidget(self.view_slider, 2, 2, 1, 8)

        self.view_label = QLabel("Widok: caly plik")
        controls_layout.addWidget(self.view_label, 2, 10)

        self.info_label = QLabel("Wczytaj plik WAV, następnie wciśnij Analiza")
        self.info_label.setWordWrap(True)
        controls_layout.addWidget(self.info_label, 3, 0, 1, 11)

        main_layout.addLayout(controls_layout)

        self.tabs = QTabWidget()

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.waveform_plot = LinePlotWidget("Przebieg czasowy z zaznaczona cisza", "#335c88", 240)
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
            ]
        )
        self.feature_selector.currentIndexChanged.connect(self.update_feature_plot)
        feature_controls.addWidget(self.feature_selector)
        feature_controls.addStretch(1)
        features_layout.addLayout(feature_controls)
        self.feature_plot = LinePlotWidget("Volume", "#5f9e6e", 340)
        features_layout.addWidget(self.feature_plot)
        self.tabs.addTab(features_tab, "Cechy")

        frames_tab = QWidget()
        frames_layout = QVBoxLayout(frames_tab)

        self.frames_table = QTableWidget()
        self.frames_table.setColumnCount(13)
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
                "Voicing",
                "Speech/Music",
            ]
        )
        header = self.frames_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        frames_layout.addWidget(self.frames_table)
        self.tabs.addTab(frames_tab, "Ramki")

        main_layout.addWidget(self.tabs)
        self.set_playback_controls_enabled(False)

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
        self.frames_table.setRowCount(0)
        self.summary_text.setPlainText("")
        self.feature_plot.clear_plot()
        self.voicing_timeline.clear_timeline()
        self.speech_music_timeline.clear_timeline()

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
        else:
            values = [frame.dominant_frequency_fft for frame in self.analysis_result.frames]
            color = "#2e8c93"

        self.feature_plot.title = selected_name
        self.feature_plot.line_color = QColor(color)
        self.feature_plot.set_plot_data(frame_times, values)
        self.apply_view_range_to_widgets()
        self.feature_plot.set_playhead_time(self.player.position() / 1000.0)

    def update_summary(self) -> None:
        if self.analysis_result is None:
            self.summary_text.clear()
            return

        audio = self.analysis_result.audio_data
        clip = self.analysis_result.clip
        frames = self.analysis_result.frames

        voiced_frames = sum(1 for frame in frames if frame.voicing_label == "voiced")
        unvoiced_frames = sum(1 for frame in frames if frame.voicing_label == "unvoiced")
        silent_frames = sum(1 for frame in frames if frame.voicing_label == "silence")
        speech_frames = sum(1 for frame in frames if frame.speech_music_label == "speech")
        music_frames = sum(1 for frame in frames if frame.speech_music_label == "music")

        lines = [
            "Podsumowanie klipu",
            "",
            f"Plik: {audio.path}",
            f"Czestotliwosc probkowania: {audio.sample_rate} Hz",
            f"Czestotliwosc analizy: {self.analysis_result.analysis_sample_rate} Hz",
            f"Downsample factor: {self.analysis_result.downsample_factor}",
            f"Liczba kanalow: {audio.channels}",
            f"Dlugosc: {audio.duration_seconds:.3f} s",
            f"Frame/Hop: {self.analysis_result.frame_ms:.2f} ms / {self.analysis_result.hop_ms:.2f} ms",
            f"Liczba ramek: {len(frames)}",
            "",
            "Progi ciszy:",
            f"- volume_norm < {self.analysis_result.silence_volume_threshold:.4f}",
            f"- zcr < {self.analysis_result.silence_zcr_threshold:.4f}",
            "",
            "Cechy clip-level:",
            f"- Mean Volume: {clip.mean_volume:.6f}",
            f"- VSTD: {clip.vstd:.6f}",
            f"- VDR: {clip.vdr:.6f}",
            f"- VU: {clip.vu:.6f}",
            f"- LSTER: {clip.lster:.6f}",
            f"- Energy Entropy: {clip.energy_entropy:.6f}",
            f"- ZSTD: {clip.zstd:.6f}",
            f"- HZCRR: {clip.hzcrr:.6f}",
            f"- Silent Ratio: {clip.silent_ratio:.6f}",
            f"- Mean F0 (autokorelacja): {clip.mean_f0_autocorrelation:.3f} Hz",
            f"- Mean F0 (AMDF): {clip.mean_f0_amdf:.3f} Hz",
            f"- Mean dominant FFT frequency: {clip.mean_dominant_frequency_fft:.3f} Hz",
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

        for start_time, end_time, label in self.analysis_result.voicing_segments:
            lines.append(f"- {start_time:.3f}s - {end_time:.3f}s: {label}")

        lines.append("")
        lines.append("Segmenty speech/music:")
        for start_time, end_time, label in self.analysis_result.speech_music_segments:
            lines.append(f"- {start_time:.3f}s - {end_time:.3f}s: {label}")

        self.summary_text.setPlainText("\n".join(lines))

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
                frame.voicing_label,
                frame.speech_music_label,
            ]

            for column_index, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                if column_index == 11:
                    self.apply_label_color(item, frame.voicing_label)
                if column_index == 12:
                    self.apply_label_color(item, frame.speech_music_label)
                self.frames_table.setItem(row_index, column_index, item)

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

    def on_position_slider_moved(self, value: int) -> None:
        duration_ms = self.position_slider.maximum()
        self.position_label.setText(
            f"{self.format_milliseconds(value)} / {self.format_milliseconds(duration_ms)}"
        )
        self.update_playhead_visuals(value / 1000.0)

    def update_playhead_visuals(self, position_seconds: float) -> None:
        self.waveform_plot.set_playhead_time(position_seconds)
        self.feature_plot.set_playhead_time(position_seconds)
        self.voicing_timeline.set_playhead_time(position_seconds)
        self.speech_music_timeline.set_playhead_time(position_seconds)

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
        self.open_button.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        self.export_csv_button.setEnabled(enabled)
        self.export_txt_button.setEnabled(enabled)
        self.frame_input.setEnabled(enabled)
        self.hop_input.setEnabled(enabled)

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
