import sys
import threading
import numba
import numpy as np
import sounddevice as sd
from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

SAMPLE_RATE = 44100
BLOCKSIZE = 1024
N_HARMONICS = 24  # number of overtones/harmonics

# Utility: convert dB to linear
def db_to_linear(db):
    return 10.0 ** (db / 20.0)

def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-(x - b) / a))

@numba.njit
def synth_block(phases, freqs, amps, frames, sr):
    # Numba-optimized block synthesis
    block = np.zeros(frames, dtype=np.float32)
    two_pi = 2.0 * np.pi
    for i in range(len(freqs)):
        freq = freqs[i]
        amp = amps[i]
        phase = phases[i]
        for n in range(frames):
            block[n] += amp * np.sin(phase)
            phase += two_pi * freq / sr
        phases[i] = phase % two_pi
    return block

class AudioEngine:
    def __init__(self, shared):
        self.shared = shared
        self.sr = SAMPLE_RATE
        self.blocksize = BLOCKSIZE
        self.stream = None
        self.phases = np.zeros(N_HARMONICS, dtype=np.float32)  # phase per harmonic
        self.lock = threading.Lock()
        self.sm_vol = shared['volume']
        self.sm_f0 = shared['f0']
        self.sm_F1 = shared['F1']
        self.sm_F2 = shared['F2']
        self.sm_bw1 = shared['bw1']
        self.sm_bw2 = shared['bw2']
        self.sm_g1 = shared['g1']
        self.sm_g2 = shared['g2']

    def start(self):
        if self.stream is not None:
            return
        self.stream = sd.OutputStream(channels=1, callback=self.callback,
                                      samplerate=self.sr, blocksize=self.blocksize,
                                      dtype='float32')
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def callback(self, outdata, frames, time, status):
        # Real-time audio callback. Keep everything efficient.
        if status:
            print('Stream status:', status)

        with self.lock:
            # target parameters
            vol = float(self.shared['volume'])
            t_f0 = float(self.shared['f0'])
            t_F1 = float(self.shared['F1'])
            t_F2 = float(self.shared['F2'])
            t_bw1 = float(self.shared['bw1'])
            t_bw2 = float(self.shared['bw2'])
            t_g1 = float(self.shared['g1'])
            t_g2 = float(self.shared['g2'])

        # smoothing coefficient
        smooth = 0.8

        # apply smoothing outside lock
        self.sm_vol = smooth * self.sm_vol + (1.0 - smooth) * vol
        self.sm_f0 = smooth * self.sm_f0 + (1.0 - smooth) * t_f0
        self.sm_F1 = smooth * self.sm_F1 + (1.0 - smooth) * t_F1
        self.sm_F2 = smooth * self.sm_F2 + (1.0 - smooth) * t_F2
        self.sm_bw1 = smooth * self.sm_bw1 + (1.0 - smooth) * t_bw1
        self.sm_bw2 = smooth * self.sm_bw2 + (1.0 - smooth) * t_bw2
        self.sm_g1 = smooth * self.sm_g1 + (1.0 - smooth) * t_g1
        self.sm_g2 = smooth * self.sm_g2 + (1.0 - smooth) * t_g2

        # final parameters
        vol = self.sm_vol
        f0 = self.sm_f0 + np.cos(time.outputBufferDacTime * 2.0 * np.pi * 5.0) * self.sm_f0 * 0.01
        F1 = self.sm_F1
        F2 = self.sm_F2
        bw1 = self.sm_bw1 * F1
        bw2 = self.sm_bw2 * F2
        g1 = self.sm_g1
        g2 = self.sm_g2

        # harmonic indexes 1..N_HARMONICS
        ks = np.arange(1, N_HARMONICS + 1)
        freqs = ks * f0

        # base overtone amplitude with -3 dB per octave relative to previous
        # amplitude for harmonic k: a_k = (10^(-3/20))^(k-1)
        base_drop_linear = db_to_linear(-6.0)
        base_amplitudes = base_drop_linear ** np.log2(ks)

        # formant gains: two Gaussians centered at F1,F2 with bandwidths (sigma)
        # convert bandwidth (Hz) to gaussian sigma (we treat bw as standard deviation here)
        formant1 = 1.0 + g1 * np.exp(-0.5 * ((freqs - F1) / bw1) ** 2)
        formant2 = 1.0 + g2 * np.exp(-0.5 * ((freqs - F2) / bw2) ** 2)
        low_pass = sigmoid(freqs, -410, 4900)

        amplitudes = base_amplitudes * formant1 * formant2 * low_pass

        # normalize to avoid clipping; keep a safety headroom
        max_possible = np.sum(amplitudes)  # rough estimate
        if max_possible > 0:
            norm = 0.9 / max_possible
        else:
            norm = 1.0
        amplitudes *= norm * vol

        # compute block
        block = synth_block(self.phases, freqs, amplitudes, frames, self.sr)

        # convert to float32 and write to outdata
        outdata[:] = block.reshape(-1, 1)


class FormantPad(QtWidgets.QWidget):
    # 2D pad where x -> F1, y -> F2
    changed = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.margin = 10
        # default F1,F2
        self.F1 = 500.0
        self.F2 = 1500.0
        self._dragging = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200)))
        painter.drawRect(rect)

        # draw axes labels
        painter.drawText(rect.left() + 5, rect.bottom() - 5, 'F1')
        painter.drawText(rect.right() - 30, rect.top() + 15, 'F2')

        painter.drawText(rect.left(), rect.top()-5, "F2:600")
        painter.drawText(rect.left(), rect.bottom()+15, "F1:200")
        painter.drawText(rect.right()-40, rect.top()-5, "3500")
        painter.drawText(rect.right()-40, rect.bottom()+15, "1200")

        # draw marker
        x = self.freq_to_x(self.F1)
        y = self.freq_to_y(self.F2)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 100, 100)))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.drawEllipse(QtCore.QPointF(x, y), 6, 6)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.drawText(x + 10, y - 10, f"F1={int(self.F1)}Hz")
        painter.drawText(x + 10, y + 10, f"F2={int(self.F2)}Hz")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.set_from_pos(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.set_from_pos(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False

    def set_from_pos(self, pos):
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        x = np.clip(pos.x(), rect.left(), rect.right())
        y = np.clip(pos.y(), rect.top(), rect.bottom())
        self.F1 = self.x_to_freq(x)
        self.F2 = self.y_to_freq(y)
        self.changed.emit(self.F1, self.F2)
        self.update()

    def freq_to_x(self, f):
        # map F1 range 200 - 1200 Hz to x in rect
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        x = rect.left() + (f - 200.0) / (1200.0 - 200.0) * rect.width()
        return int(x)

    def freq_to_y(self, f):
        # map F2 range 600 - 3500 Hz to y in rect (invert y axis)
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        y = rect.bottom() - (f - 600.0) / (3500.0 - 600.0) * rect.height()
        return int(y)

    def x_to_freq(self, x):
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        return 200.0 + (x - rect.left()) / rect.width() * (1200.0 - 200.0)

    def y_to_freq(self, y):
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        return 600.0 + (rect.bottom() - y) / rect.height() * (3500.0 - 600.0)


class AmplitudePlot(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 2))
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        fig.tight_layout()
        self.bars = None

    def update_plot(self, freqs, amplitudes):
        self.ax.cla()
        for f, a in zip(freqs, amplitudes):
            self.ax.plot([f, f], [0, a], linewidth=1)
        self.ax.set_xlim(60, 8000)
        self.ax.set_ylim(0, np.max(amplitudes) * 1.1)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Amplitude (rel)')
        self.ax.set_title('Harmonics in Frequency Domain')
        self.draw()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Basic Formant Synthesizer')
        self.shared = {
            'volume': 0.3,
            'f0': 210.0,
            'F1': 850.0,
            'F2': 1610.0,
            'bw1': 0.3,
            'bw2': 0.2,
            'g1': 12.0,  # gain multiplier for formant (how strong the amplification is)
            'g2': 15.0,
        }

        self.audio = AudioEngine(self.shared)
        self.init_ui()
        #self.audio.start()

        # small timer to refresh GUI plot
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start()

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # controls: volume and fundamental frequency sliders
        controls = QtWidgets.QHBoxLayout()

        vol_layout = QtWidgets.QVBoxLayout()
        vol_label = QtWidgets.QLabel('Volume')
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(self.shared['volume'] * 100))
        self.vol_slider.valueChanged.connect(self.on_volume_change)
        vol_layout.addWidget(vol_label)
        vol_layout.addWidget(self.vol_slider)
        controls.addLayout(vol_layout)

        f0_layout = QtWidgets.QVBoxLayout()
        f0_label = QtWidgets.QLabel('Fundamental (Hz)')
        self.f0_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.f0_slider.setRange(int(np.log2(90) * 100), int(np.log2(800) * 100))
        self.f0_slider.setValue(int(np.log2(self.shared['f0']) * 100))
        self.f0_slider.valueChanged.connect(self.on_f0_change)
        f0_layout.addWidget(f0_label)
        f0_layout.addWidget(self.f0_slider)
        controls.addLayout(f0_layout)

        # formant pad
        self.formant_pad = FormantPad()
        self.formant_pad.F1 = self.shared['F1']
        self.formant_pad.F2 = self.shared['F2']
        self.formant_pad.changed.connect(self.on_formant_change)
        controls.addWidget(self.formant_pad)

        # additional formant parameter controls
        params_layout = QtWidgets.QFormLayout()
        self.bw1_spin = QtWidgets.QDoubleSpinBox()
        self.bw1_spin.setRange(0.1, 0.8)
        self.bw1_spin.setValue(self.shared['bw1'])
        self.bw1_spin.valueChanged.connect(self.on_bw1_change)
        params_layout.addRow('BW1 (Hz)', self.bw1_spin)

        self.bw2_spin = QtWidgets.QDoubleSpinBox()
        self.bw2_spin.setRange(0.1, 0.8)
        self.bw2_spin.setValue(self.shared['bw2'])
        self.bw2_spin.valueChanged.connect(self.on_bw2_change)
        params_layout.addRow('BW2 (Hz)', self.bw2_spin)

        self.g1_spin = QtWidgets.QDoubleSpinBox()
        self.g1_spin.setRange(0.0, 20.0)
        self.g1_spin.setValue(self.shared['g1'])
        self.g1_spin.valueChanged.connect(self.on_g1_change)
        params_layout.addRow('Formant1 Gain', self.g1_spin)

        self.g2_spin = QtWidgets.QDoubleSpinBox()
        self.g2_spin.setRange(0.0, 20.0)
        self.g2_spin.setValue(self.shared['g2'])
        self.g2_spin.valueChanged.connect(self.on_g2_change)
        params_layout.addRow('Formant2 Gain', self.g2_spin)

        controls.addLayout(params_layout)

        self.info_label = QtWidgets.QLabel()
        layout.addWidget(self.info_label)

        layout.addLayout(controls)

        # amplitude plot
        self.plot = AmplitudePlot(self)
        layout.addWidget(self.plot)

        # start/stop and quit
        button_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Start')
        self.start_btn.setCheckable(True)
        self.start_btn.setChecked(True)
        self.start_btn.toggled.connect(self.on_start_stop)
        button_layout.addWidget(self.start_btn)
        quit_btn = QtWidgets.QPushButton('Quit')
        quit_btn.clicked.connect(self.close)
        button_layout.addWidget(quit_btn)
        layout.addLayout(button_layout)

    def on_volume_change(self, val):
        self.shared['volume'] = val / 100.0

    def on_f0_change(self, val):
        self.shared['f0'] = 2.0 ** (val / 100.0)

    def on_formant_change(self, F1, F2):
        self.shared['F1'] = F1
        self.shared['F2'] = F2

    def on_bw1_change(self, val):
        self.shared['bw1'] = float(val)

    def on_bw2_change(self, val):
        self.shared['bw2'] = float(val)

    def on_g1_change(self, val):
        self.shared['g1'] = float(val)

    def on_g2_change(self, val):
        self.shared['g2'] = float(val)

    def on_start_stop(self, checked):
        if checked:
            # was running; stop
            self.audio.stop()
            self.start_btn.setText('Start')
        else:
            self.audio.start()
            self.start_btn.setText('Stop')

    def refresh_plot(self):
        # compute current amplitudes (same formula as audio callback but without phase)
        f0 = float(self.shared['f0'])
        ks = np.arange(1, N_HARMONICS + 1)
        freqs = ks * f0
        base_drop_linear = db_to_linear(-6.0)
        base_amplitudes = base_drop_linear ** np.log2(ks)
        F1 = float(self.shared['F1'])
        F2 = float(self.shared['F2'])
        bw1 = float(self.shared['bw1']) * F1
        bw2 = float(self.shared['bw2']) * F2
        g1 = float(self.shared['g1'])
        g2 = float(self.shared['g2'])
        formant1 = 1.0 + g1 * np.exp(-0.5 * ((freqs - F1) / bw1) ** 2)
        formant2 = 1.0 + g2 * np.exp(-0.5 * ((freqs - F2) / bw2) ** 2)
        low_pass = sigmoid(freqs, -410, 4900)
        amplitudes = base_amplitudes * formant1 * formant2 * low_pass
        # scale for display
        amplitudes_display = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes

        self.info_label.setText(f"f0={int(f0)} Hz   F1={int(F1)} Hz   F2={int(F2)} Hz")

        self.plot.update_plot(freqs, amplitudes_display)

    def closeEvent(self, event):
        self.audio.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
