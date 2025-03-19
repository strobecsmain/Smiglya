#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Объединённый финальный код, который содержит функционал open source (центральный виджет "Начать")
и дополнительные возможности: "Внести данные", "Запустить модель" и т.д.
Логи сохраняются в log_ai.txt, модели – в model_ai.h5.
"""

DEFAULT_LOG_FILENAME = "log_ai.txt"
DEFAULT_MODEL_FILENAME = "model_ai.h5"

import sys, os, types, re, numpy as np, matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QToolBar, QAction, QFileDialog, 
                             QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
                             QPushButton, QWidget, QDockWidget, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QTextCursor
from tensorflow.keras.models import load_model
from logic_weather import train_or_load, load_data, normalize_data, MONTHS

# Перехватчик stdout – чтобы все print летали в наше окно логов
class EmittingStream(QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        if text:
            self.text_written.emit(text)
    def flush(self): pass

# Холст для графиков
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

# Диалог настроек обучения – без излишеств
class ParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры обучения")
        l = QVBoxLayout(self)
        self.epochsSpin = self.add_spin(l, "Количество эпох:", 300)
        self.batchSpin = self.add_spin(l, "Размер батча:", 8)
        btnBox = QHBoxLayout()
        bOk, bCancel = QPushButton("Продолжить"), QPushButton("Отмена")
        bOk.clicked.connect(self.accept)
        bCancel.clicked.connect(self.reject)
        btnBox.addWidget(bOk)
        btnBox.addWidget(bCancel)
        l.addLayout(btnBox)
    def add_spin(self, layout, label, default):
        h = QHBoxLayout()
        h.addWidget(QLabel(label))
        s = QSpinBox()
        s.setRange(1, 1000000)
        s.setValue(default)
        h.addWidget(s)
        layout.addLayout(h)
        return s

# Центральный виджет с кнопкой "Начать" – как в open source версии
class CentralWidget(QWidget):
    startRequested = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addStretch()
        btn = QPushButton("Начать")
        btn.setFont(QFont("", 16, QFont.Bold))
        btn.clicked.connect(self.startRequested.emit)
        lay.addWidget(btn, alignment=Qt.AlignCenter)
        lay.addStretch()
        self.setLayout(lay)

# Фоновый поток для обучения, чтобы UI не зависал
class TrainThread(QThread):
    finished_signal = pyqtSignal(object, object, object, object)  # (history, model, prediction, scaler)
    log_signal = pyqtSignal(str)
    def __init__(self, args):
        super().__init__()
        self.args = args
    def run(self):
        import sys
        old_stdout = sys.stdout
        stream = EmittingStream()
        stream.text_written.connect(self.log_signal)
        sys.stdout = stream
        try:
            hist, mdl, pred, scl = train_or_load(
                data_file=self.args.data,
                output_file=self.args.output,
                model_file=DEFAULT_MODEL_FILENAME,
                load_existing=self.args.load_model,
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                logs_dir="logs"
            )
            self.finished_signal.emit(hist, mdl, pred, scl)
        finally:
            sys.stdout = old_stdout

# Главное окно приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI - прогноз погоды")
        # Общие переменные
        self.args = self.history = self.model = self.prediction = self.scaler = self.model_saved_path = None
        self.console_text = ""      # Содержит текст логов, считанных из log_ai.txt
        self.train_losses = []      # Распарсенные значения loss
        self.val_losses = []        # Распарсенные значения val_loss
        self.training_logs = []     # Буфер новых логов при обучении

        self.canvas_loss = MplCanvas(self, 5, 4, 100)
        self.canvas_pred = MplCanvas(self, 5, 4, 100)
        self.current_canvas = None

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        self.logDock = QDockWidget("Процесс обучения", self)
        self.logDock.setWidget(self.logText)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.logDock)
        self.logDock.hide()

        self._setup_toolbar()
        # В данном варианте используется центральный виджет с кнопкой "Начать"
        self.centralWidget = CentralWidget()
        self.centralWidget.startRequested.connect(self.startProcess)
        self.setCentralWidget(self.centralWidget)
        self.load_console_file(DEFAULT_LOG_FILENAME)
        self.logText.setPlainText(self.console_text)
    def load_console_file(self, file_path):
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.console_text = "".join(lines)
        self.train_losses = []
        self.val_losses = []
        for line in lines:
            match = re.search(r"loss:\s*([0-9.]+).*val_loss:\s*([0-9.]+)", line)
            if match:
                self.train_losses.append(float(match.group(1)))
                self.val_losses.append(float(match.group(2)))
    def _setup_toolbar(self):
        tb = QToolBar("Управление", self)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.setMinimumSize(1200, 600)
        tb.setMovable(False)
        tb.setFloatable(False)
        font = QFont()
        font.setBold(True)

        actions = [
            ("Процесс обучения", self.toggleLog),
            ("Обучить модель", self.initiateTraining),
            ("Сохранить модель", self.saveModel),
            ("Качество обучения", self.showLossCanvas),
            ("График прогноз", self.showPredCanvas),
            ("Сохранить график", self.saveFigure),
            ("Сохранить прогноз", self.saveForecast),
            ("Сохранить прогноз .txt", self.saveForecast),  # <-- Добавлена кнопка
            ("Внести данные", self.enterData),
            ("Запустить модель", self.launchModel)
        ]

        for text, slot in actions:
            act = QAction(text, self)
            act.setFont(font)
            act.triggered.connect(slot)
            tb.addAction(act)
    def toggleLog(self):
        self.logDock.setVisible(not self.logDock.isVisible())
    def appendLog(self, text):
        self.logText.moveCursor(QTextCursor.End)
        self.logText.insertPlainText(text)
        self.logText.moveCursor(QTextCursor.End)
        self.training_logs.append(text)
    # "Запустить модель": выбор модели и файла логов для инференса
    def launchModel(self):
        mFile, _ = QFileDialog.getOpenFileName(self, "Выберите файл модели", "", "H5 Files (*.h5);;Все файлы (*)")
        if not mFile:
            return
        logFile, _ = QFileDialog.getOpenFileName(self, "Выберите файл логов", "", "Text Files (*.txt);;Все файлы (*)")
        if not logFile:
            return
        self.model = load_model(mFile)
        self.statusBar().showMessage(f"Модель загружена из {mFile}.")
        if os.path.exists(logFile):
            with open(logFile, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.console_text = "".join(lines)
            self.logText.setPlainText(self.console_text)
            self.train_losses = []
            self.val_losses = []
            for line in lines:
                match = re.search(r"loss:\s*([0-9.]+).*val_loss:\s*([0-9.]+)", line)
                if match:
                    self.train_losses.append(float(match.group(1)))
                    self.val_losses.append(float(match.group(2)))
            self.showLossPlot()
        else:
            QMessageBox.warning(self, "Ошибка", f"Файл логов не найден: {logFile}")
    # "Начать": выбор модели и данных для инференса/обучения
    def startProcess(self):
        if QMessageBox.question(self, "Открыть модель?",
                                  "Открыть уже обученную модель?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            mFile, _ = QFileDialog.getOpenFileName(self, "Выберите файл модели", "", "H5 Files (*.h5);;Все файлы (*)")
            if mFile:
                dFile, _ = QFileDialog.getOpenFileName(self, "Выберите файл с данными", "", "Text Files (*.txt);;Все файлы (*)")
                if dFile:
                    self.loadModelAndPredict(mFile, dFile)
        else:
            self.initiateTraining()
    def loadModelAndPredict(self, model_file, data_file):
        self.model = load_model(model_file)
        data = load_data(data_file)
        if len(data) < 2:
            QMessageBox.warning(self, "Ошибка", "Недостаточно данных (минимум 2 года).")
            return
        data, scl = normalize_data(data)
        raw = self.model.predict(data[-1].reshape(1,12))[0]
        self.history = None
        self.prediction = scl.inverse_transform(raw.reshape(1,-1))[0]
        self.scaler = scl
        self.model_saved_path = None
        self.statusBar().showMessage("Модель загружена, прогноз готов.")
        self.showLossPlot()
        self.showPredPlot()
        self.showPredCanvas()
    # "Внести данные": инференс без обучения
    def enterData(self):
        if not self.model:
            QMessageBox.warning(self, "Внимание", "Модель не загружена.")
            return
        dFile, _ = QFileDialog.getOpenFileName(self, "Выберите файл с данными", "", "Text Files (*.txt);;Все файлы (*)")
        if dFile:
            data = load_data(dFile)
            if len(data) < 2:
                QMessageBox.warning(self, "Ошибка", "Недостаточно данных (минимум 2 года).")
                return
            data, scl = normalize_data(data)
            raw = self.model.predict(data[-1].reshape(1,12))[0]
            self.prediction = scl.inverse_transform(raw.reshape(1,-1))[0]
            self.scaler = scl
            self.statusBar().showMessage("Новые данные обработаны, прогноз обновлён.")
            self.showPredPlot()
            self.showPredCanvas()
    def initiateTraining(self):
        dFile, _ = QFileDialog.getOpenFileName(self, "Выберите файл с данными", "", "Text Files (*.txt);;Все файлы (*)")
        if not dFile:
            return
        dlg = ParamDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            outFile = os.path.join(os.path.dirname(dFile), "forecast.txt")
            self.args = types.SimpleNamespace(data=dFile, output=outFile, model=DEFAULT_MODEL_FILENAME,
                                               load_model=False, epochs=dlg.epochsSpin.value(),
                                               batch_size=dlg.batchSpin.value())
            self.logText.clear()
            self.training_logs = []
            self.trainThread = TrainThread(self.args)
            self.trainThread.log_signal.connect(self.appendLog)
            self.trainThread.finished_signal.connect(self.onTrainingFinished)
            self.trainThread.start()
            self.statusBar().showMessage("Идёт обучение...")
            self.logDock.show()
    def onTrainingFinished(self, hist, mdl, pred, scl):
        self.history, self.model, self.prediction, self.scaler = hist, mdl, pred, scl
        self.model_saved_path = None
        self.statusBar().showMessage("Обучение завершено.")
        self.showLossPlot()
        self.showPredPlot()
        self.showLossCanvas()
    def saveModel(self):
        if not self.model:
            QMessageBox.warning(self, "Внимание", "Нет модели для сохранения.")
            return
        fName, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", DEFAULT_MODEL_FILENAME,
                                               "H5 Files (*.h5);;Все файлы (*)")
        if fName:
            self.model.save(fName)
            self.model_saved_path = fName
            QMessageBox.information(self, "Сохранение", f"Модель сохранена: {fName}")
            with open(DEFAULT_LOG_FILENAME, "w", encoding="utf-8") as f:
                f.write("".join(self.training_logs))
            QMessageBox.information(self, "Сохранение логов",
                                    f"Логи обучения сохранены в {DEFAULT_LOG_FILENAME}")
    def closeEvent(self, event):
        if self.model and QMessageBox.question(self, "Сохранить модель?",
                                               "Сохранить модель перед выходом?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
            self.model.save(self.model_saved_path if self.model_saved_path else DEFAULT_MODEL_FILENAME)
        event.accept()
    def showLossPlot(self):
        ax = self.canvas_loss.axes
        ax.clear()
        if self.history:
            ax.set_title("График потерь (MSE)")
            ax.set_xlabel("Эпоха")
            ax.set_ylabel("Потери")
            ax.grid(True)
            ax.set_yticks(np.arange(0.0, 3.0, 0.1))
            ax.plot(self.history.history['loss'], label="Обучение")
            if "val_loss" in self.history.history:
                ax.plot(self.history.history['val_loss'], label="Валидация")
            ax.legend()
        else:
            if self.train_losses and self.val_losses:
                ax.set_title("График потерь (MSE) (из log_ai.txt)")
                ax.set_xlabel("Эпоха")
                ax.set_ylabel("Потери")
                ax.grid(True)
                epochs = range(1, len(self.train_losses) + 1)
                ax.plot(epochs, self.train_losses, label="Обучение")
                ax.plot(epochs, self.val_losses, label="Валидация")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Нет данных для графика", ha="center", va="center", transform=ax.transAxes)
        self.canvas_loss.draw()
    def showPredPlot(self):
        ax = self.canvas_pred.axes
        ax.clear()
        ax.set_title("Прогноз средних температур на следующий год")
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Температура (C°)")
        ax.grid(True)
        if self.prediction is not None:
            x = np.arange(1, 13)
            ax.set_xticks(x)
            ax.plot(x, self.prediction, marker="o")
        else:
            ax.text(0.5, 0.5, "Нет предсказания", ha="center", va="center", transform=ax.transAxes)
        self.canvas_pred.draw()
    def showLossCanvas(self):
        if self.current_canvas:
            self.current_canvas.setParent(None)
        self.current_canvas = self.canvas_loss
        self.setCentralWidget(self.current_canvas)
    def showPredCanvas(self):
        if self.current_canvas:
            self.current_canvas.setParent(None)
        self.current_canvas = self.canvas_pred
        self.setCentralWidget(self.current_canvas)
    def saveFigure(self):
        if not self.current_canvas:
            return
        fName, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "", "PNG Files (*.png);;Все файлы (*)")
        if fName:
            self.current_canvas.figure.savefig(fName)
            QMessageBox.information(self, "Сохранение", f"График сохранён: {fName}")
    def saveForecast(self):
        if self.prediction is None:
            QMessageBox.warning(self, "Внимание", "Нет прогноза для сохранения.")
            return
        fName, _ = QFileDialog.getSaveFileName(self, "Сохранить прогноз", "", "Text Files (*.txt);;Все файлы (*)")
        if fName:
            with open(fName, "w", encoding="utf-8") as f:
                for i, temp in enumerate(self.prediction):
                    m = MONTHS[i] if i < len(MONTHS) else f"Месяц {i+1}"
                    f.write(f"{m}   {temp:.2f} C°\n")
            QMessageBox.information(self, "Сохранение", f"Прогноз сохранён: {fName}")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900,600)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
