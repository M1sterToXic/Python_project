import sys
import matplotlib
matplotlib.use('Qt5Agg')  # Используем бэкенд Qt5Agg для совместимости с PyQt5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QPushButton, QLabel, QApplication
from PyQt5.QtCore import QTimer
import time

# Глобальная переменная для масштаба
zoom_level = 0.7  # Изменяем начальный масштаб на 0.7x
# Переменная для отслеживания времени последнего изменения
last_change_time = 0
# Задержка в миллисекундах перед автоматическим обновлением
AUTO_UPDATE_DELAY = 1500  # 1 секунда
# Начальные параметры
L = 0.5102  # м (изменено, чтобы граница синего и зелёного была на x = 2.5 см)
d = 1e-5   # м
light_type = "Белый свет"
# Фиксированные значения длины волны для каждого типа света (средние значения диапазонов, для цвета)
lambda_values = {
    "Белый свет": 550e-9,  # Среднее значение для белого света
    "Красный": (620e-9 + 730e-9) / 2,
    "Оранжевый": (590e-9 + 620e-9) / 2,
    "Жёлтый": (560e-9 + 590e-9) / 2,
    "Зелёный": (500e-9 + 560e-9) / 2,
    "Голубой": (480e-9 + 500e-9) / 2,
    "Синий": (450e-9 + 480e-9) / 2,
    "Фиолетовый": (400e-9 + 450e-9) / 2,
}
# Длины волн, соответствующие пику яркости для каждого светофильтра
peak_lambda_values = {
    "Белый свет": 550e-9,  # Не используется, но оставляем для консистентности
    "Красный": 655e-9,     # Пик яркости в диапазоне 620–760 нм (где factor = 1.0)
    "Оранжевый": 605e-9,   # Пик яркости в диапазоне 590–620 нм
    "Жёлтый": 575e-9,      # Пик яркости в диапазоне 560–590 нм
    "Зелёный": 530e-9,     # Пик яркости в диапазоне 500–560 нм
    "Голубой": 490e-9,     # Пик яркости в диапазоне 480–500 нм
    "Синий": 440e-9,       # Пик яркости в диапазоне 450–480 нм
    "Фиолетовый": 390e-9,  # Пик яркости в диапазоне 380–450 нм (где factor ближе к 1.0)
}
lambda_ = lambda_values[light_type]

# Функция для нормализации ввода (замена запятой на точку)
def normalize_input(value):
    return value.replace(',', '.') if isinstance(value, str) else value

# Функция преобразования длины волны в цвет
def wavelength_to_rgb(lambda_):
    lambda_nm = lambda_ * 1e9
    r = 0.0
    g = 0.0
    b = 0.0

    if 380 <= lambda_nm <= 440:
        r = -0.5 * (lambda_nm - 460) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 < lambda_nm <= 490:
        r = 0.0
        g = (lambda_nm - 440) / (490 - 440)
        b = 1.0
    elif 490 < lambda_nm <= 510:
        r = 0.0
        g = 1.0
        b = -(lambda_nm - 510) / (510 - 490)
    elif 510 < lambda_nm <= 580:
        r = (lambda_nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 < lambda_nm <= 645:
        r = 1.0
        g = -(lambda_nm - 645) / (645 - 580)
        b = 0.0
    elif 645 < lambda_nm <= 730:
        r = 1.0
        g = 0.0
        b = 0.0

    if 380 <= lambda_nm <= 420:
        factor = 0.3 + 0.7 * (lambda_nm - 380) / (420 - 380)
    elif 420 < lambda_nm <= 700:
        factor = 1.0
    elif 700 < lambda_nm <= 760:
        factor = 0.3 + 0.7 * (730 - lambda_nm) / (730 - 700)
    else:
        factor = 0.0

    r *= factor
    g *= factor
    b *= factor

    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))

    return (r, g, b)

# Функция построения дифракционной картины с Matplotlib
def plot_diffraction_pattern(ax, lambda_, d, L, light_type, zoom=1.0):
    ax.clear()
    L_cm = L * 100
    d_cm = d * 100

    # Фиксируем масштаб линейки, используя длину волны белого света (550 нм) независимо от типа света
    reference_lambda_cm = 550e-9 * 100  # Всегда используем 550 нм для масштаба

    max_m = 2
    max_theta = np.arcsin(max_m * reference_lambda_cm / d_cm)
    x_range_cm = L_cm * np.tan(max_theta) / zoom
    x = np.linspace(-x_range_cm, x_range_cm, 1000)

    # Центральный максимум (m = 0)
    central_width = 0.1  # 0.1 см
    mask = (x >= -central_width) & (x <= central_width)
    x_central = x[mask]
    y_central = np.ones_like(x_central)
    central_color = 'white' if light_type == "Белый свет" else wavelength_to_rgb(lambda_)
    ax.fill_between(x_central, 0, y_central, color=central_color, alpha=1)

    # Дифракционные полосы для m ≠ 0
    if light_type == "Белый свет":
        lambda_range = np.linspace(380e-9 * 100, 760e-9 * 100, 100)
        for m_val in range(1, max_m + 1):
            strip_width = 0.03  # Фиксированная ширина для белого света
            for lambda_i in lambda_range:
                arg = m_val * lambda_i / d_cm
                if abs(arg) <= 1:
                    theta = np.arcsin(arg)
                    x_m_pos = L_cm * np.tan(theta)
                    if (-x_range_cm) <= x_m_pos <= x_range_cm:
                        color = wavelength_to_rgb(lambda_i / 100)
                        mask = (x >= x_m_pos - strip_width) & (x <= x_m_pos + strip_width)
                        ax.fill_between(x[mask], 0, 1, color=color, alpha=1.0)
                arg = -m_val * lambda_i / d_cm
                if abs(arg) <= 1:
                    theta = np.arcsin(arg)
                    x_m_neg = L_cm * np.tan(theta)
                    if (-x_range_cm) <= x_m_neg <= x_range_cm:
                        color = wavelength_to_rgb(lambda_i / 100)
                        mask = (x >= x_m_neg - strip_width) & (x <= x_m_neg + strip_width)
                        ax.fill_between(x[mask], 0, 1, color=color, alpha=1.0)
    else:
        peak_lambda_cm = peak_lambda_values[light_type] * 100
        for m_val in range(1, max_m + 1):
            # Ширина полосы зависит от порядка m (например, m=1 -> 0.02, m=2 -> 0.04)
            strip_width = 0.02 * m_val  # Утолщение для высших порядков
            # Положительный порядок (m > 0)
            arg = m_val * peak_lambda_cm / d_cm
            if abs(arg) <= 1:
                theta = np.arcsin(arg)
                x_m_pos = L_cm * np.tan(theta)
                if (-x_range_cm) <= x_m_pos <= x_range_cm:
                    color = wavelength_to_rgb(lambda_)
                    mask = (x >= x_m_pos - strip_width) & (x <= x_m_pos + strip_width)
                    ax.fill_between(x[mask], 0, 1, color=color, alpha=1.0)
            # Отрицательный порядок (m < 0)
            arg = -m_val * peak_lambda_cm / d_cm
            if abs(arg) <= 1:
                theta = np.arcsin(arg)
                x_m_neg = L_cm * np.tan(theta)
                if (-x_range_cm) <= x_m_neg <= x_range_cm:
                    color = wavelength_to_rgb(lambda_)
                    mask = (x >= x_m_neg - strip_width) & (x <= x_m_neg + strip_width)
                    ax.fill_between(x[mask], 0, 1, color=color, alpha=1.0)

    # Остальные настройки графика (без изменений)
    ax.set_title(f"Дифракционная картина (Масштаб: {zoom:.1f}x)")
    ax.set_xlabel("Положение")
    ax.set_ylim(0, 1)
    ax.set_xlim(-x_range_cm, x_range_cm)
    ax.set_yticks([])
    ax.set_facecolor('black')

    # Настройка оси X
    major_tick_step = 1.0
    x_range_rounded = np.ceil(x_range_cm)
    major_ticks = np.arange(-x_range_rounded, x_range_rounded + major_tick_step, major_tick_step)
    if 0 not in major_ticks:
        major_ticks = np.append(major_ticks, 0)
        major_ticks = np.sort(major_ticks)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{tick:.0f}" for tick in major_ticks], rotation=45)

    minor_tick_step = 0.1
    minor_ticks = np.arange(-x_range_rounded, x_range_rounded + minor_tick_step, minor_tick_step)
    ax.set_xticks(minor_ticks, minor=True)

    ax.text(x_range_cm + 0.5, -0, "см", ha='center', va='top', transform=ax.get_xaxis_transform())

# Класс главного окна
class DiffractionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симуляция дифракции Фраунгофера с масштабированием")
        self.setGeometry(100, 100, 900, 400)

        # Основной виджет и компоновка
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Создаём фигуру и оси один раз
        self.figure, self.ax = plt.subplots(figsize=(8, 2))
        self.figure.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Компоновка для элементов управления
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # Левая колонка
        left_layout = QVBoxLayout()
        controls_layout.addLayout(left_layout)

        left_layout.addWidget(QLabel("Светофильтр:"))
        self.light_type_combo = QComboBox()
        self.light_type_combo.addItems(['Белый свет', 'Красный', 'Жёлтый', 'Зелёный', 'Синий', 'Фиолетовый'])
        self.light_type_combo.setCurrentText("Белый свет")
        left_layout.addWidget(self.light_type_combo)

        left_layout.addWidget(QLabel("L (м):"))
        self.l_input = QLineEdit(str(L))
        self.l_input.setToolTip("Начальное значение: 0.5102")
        left_layout.addWidget(self.l_input)

        # Средняя колонка
        middle_layout = QVBoxLayout()
        controls_layout.addLayout(middle_layout)

        middle_layout.addWidget(QLabel("Параметры дифракционной решётки:"))
        middle_layout.addWidget(QLabel("d (мкм):"))
        self.d_input = QLineEdit(str(d * 1e6))
        self.d_input.setToolTip("Начальное значение: 10")
        middle_layout.addWidget(self.d_input)

        # Кнопки масштабирования
        zoom_buttons_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("+ Увеличить")
        self.zoom_out_button = QPushButton("- Уменьшить")
        self.zoom_reset_button = QPushButton("Сброс масштаба")
        zoom_buttons_layout.addWidget(self.zoom_in_button)
        zoom_buttons_layout.addWidget(self.zoom_out_button)
        zoom_buttons_layout.addWidget(self.zoom_reset_button)
        middle_layout.addLayout(zoom_buttons_layout)

        # Кнопка "Сохранить"
        self.save_button = QPushButton("Сохранить картину")
        middle_layout.addWidget(self.save_button)

        # Метка для масштаба
        self.zoom_label = QLabel(f"Масштаб: {zoom_level:.1f}x")  # Обновляем начальную метку масштаба
        middle_layout.addWidget(self.zoom_label)

        # Подключение сигналов
        self.light_type_combo.currentTextChanged.connect(self.on_param_changed)
        self.l_input.textChanged.connect(self.on_param_changed)
        self.d_input.textChanged.connect(self.on_param_changed)

        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_reset_button.clicked.connect(self.zoom_reset)
        self.save_button.clicked.connect(self.save_plot)

        # Таймер для автоматического обновления
        self.timer = QTimer()
        self.timer.setInterval(500)  # Проверяем каждые 500 мс
        self.timer.timeout.connect(self.check_for_update)
        self.timer.start()

        # Флаг для предотвращения избыточных обновлений
        self.is_updating = False

        # Начальный график
        self.update_plot()

    def on_param_changed(self):
        global last_change_time
        last_change_time = time.time()

    def check_for_update(self):
        global last_change_time, L, d, light_type, lambda_
        if (time.time() - last_change_time) >= (AUTO_UPDATE_DELAY / 1000):
            try:
                L = float(normalize_input(self.l_input.text()))
                d = float(normalize_input(self.d_input.text())) * 1e-6
                light_type = self.light_type_combo.currentText()
                lambda_ = lambda_values[light_type]  # Используем фиксированное значение lambda_
                self.update_plot()
            except ValueError:
                pass

    def zoom_in(self):
        global zoom_level
        zoom_level *= 1.2
        self.zoom_label.setText(f"Масштаб: {zoom_level:.1f}x")
        self.update_plot()

    def zoom_out(self):
        global zoom_level
        zoom_level /= 1.2
        self.zoom_label.setText(f"Масштаб: {zoom_level:.1f}x")
        self.update_plot()

    def zoom_reset(self):
        global zoom_level
        zoom_level = 0.7  # Сбрасываем масштаб до 0.7x
        self.zoom_label.setText(f"Масштаб: {zoom_level:.1f}x")
        self.update_plot()

    def update_plot(self):
        if self.is_updating:
            return
        self.is_updating = True
        try:
            plot_diffraction_pattern(self.ax, lambda_, d, L, light_type, zoom_level)
            self.canvas.draw()
        finally:
            self.is_updating = False

    def save_plot(self):
        self.figure.savefig("diffraction_pattern.png")
        self.show_message("Сохранено как diffraction_pattern.png")

    def show_message(self, message):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Сообщение", message)

# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiffractionWindow()
    window.show()
    sys.exit(app.exec_())