#!/usr/bin/env python3
"""
YOLO Video Detector v2.0 - Единый файл запуска
Объединенная версия main_app.py и detected_yolo.py для PyCharm
Специализация: детекция блюд, напитков и посуды с кириллическими названиями
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QMessageBox, QDialog, QVBoxLayout, QTextEdit,
                             QPushButton, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont

# Добавляем текущую директорию в путь Python
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def show_help():
    """Показать справку по использованию"""
    help_text = """
🎯 YOLO Video Detector v2.0 - Единый запускаемый файл

📦 СТРУКТУРА ПРОЕКТА:
├── detected_yolo.py        # 🚀 ГЛАВНЫЙ ФАЙЛ (этот файл)
├── video_processor.py      # Логика обработки видео
├── designer.py             # Пользовательский интерфейс PyQt5
├── class_mapping.py        # Маппинг кириллических классов
└── classes_config.json     # Конфигурация классов

🚀 ЗАПУСК В PyCharm:
   1. Откройте проект в PyCharm
   2. Запустите detected_yolo.py (правый клик -> Run)
   3. Или используйте командную строку: python detected_yolo.py

⚠️ ТРЕБОВАНИЯ:
   📁 classes_config.json - обязательный файл конфигурации классов
   📁 video_processor.py - логика обработки видео  
   📁 designer.py - интерфейс PyQt5
   📁 class_mapping.py - создается автоматически если отсутствует

🎯 ПОДДЕРЖИВАЕМЫЕ ФОРМАТЫ:
   Видео: mp4, avi, mov, mkv, wmv, flv, webm, m4v, 3gp, mpg, mpeg, ts, mts
   Модели: YOLO (.pt файлы)

🇷🇺 ОСОБЕННОСТИ:
   ✅ Кириллические названия классов для блюд и еды
   ✅ Автоматический поворот MOV файлов на 90°
   ✅ Настройка параметров детекции в реальном времени
   ✅ Фиксированное разрешение детекции 640x640
   ✅ Цветовая схема по типам еды
   ✅ Детальная статистика детекций
   ✅ Поддержка JSON конфигурации классов

💡 ИСПОЛЬЗОВАНИЕ:
   1. Выберите модель YOLO и видео
   2. Настройте параметры через интерфейс
   3. Начните обработку
   4. Наблюдайте статистику в реальном времени

🔧 УСТАНОВКА ЗАВИСИМОСТЕЙ:
   pip install ultralytics opencv-python PyQt5 Pillow numpy
"""
    print(help_text)


def check_classes_config():
    """Проверить наличие classes_config.json"""
    config_path = current_dir / 'classes_config.json'
    if not config_path.exists():
        print("⚠️ Файл classes_config.json не найден!")
        print("💡 Создайте файл classes_config.json с конфигурацией классов")
        print("📁 Пример конфигурации можно найти в документации проекта")
        return False

    try:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Проверяем обязательные секции
        required_sections = ['class_mapping', 'class_colors', 'categories']
        missing_sections = []

        for section in required_sections:
            if section not in config:
                missing_sections.append(section)

        if missing_sections:
            print(f"⚠️ В classes_config.json отсутствуют секции: {missing_sections}")
            return False

        print(f"✅ classes_config.json загружен ({len(config.get('class_mapping', {}))} классов)")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ Ошибка в JSON файле classes_config.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка чтения classes_config.json: {e}")
        return False


def check_dependencies():
    """Проверка зависимостей"""
    dependencies = {
        'PyQt5': 'PyQt5',
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow'
    }

    missing = []

    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing.append(package)
            print(f"❌ {module} (установите: pip install {package})")

    if missing:
        print(f"\n💡 Установите недостающие пакеты:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


class ClassesInfoDialog(QDialog):
    """Диалог для отображения информации о классах"""

    def __init__(self, class_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📋 Информация о классах модели")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        # Текстовое поле с информацией
        text_edit = QTextEdit()
        text_edit.setFont(QFont("Consolas", 10))
        text_edit.setReadOnly(True)

        # Формируем текст с информацией
        info_text = self.format_class_info(class_info)
        text_edit.setPlainText(info_text)

        layout.addWidget(text_edit)

        # Кнопка закрытия
        button_layout = QHBoxLayout()
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def format_class_info(self, class_info):
        """Форматирование информации о классах"""
        text = "🎯 ИНФОРМАЦИЯ О КЛАССАХ YOLO МОДЕЛИ\n"
        text += "=" * 60 + "\n\n"

        # Основная информация
        text += f"📊 Общая информация:\n"
        text += f"   Всего классов: {class_info.get('total_classes', 0)}\n"
        text += f"   Разрешение детекции: {class_info.get('detection_resolution', 'Не указано')}\n"
        text += f"   Поддержка кириллицы: {'✅ Да' if class_info.get('cyrillic_available') else '❌ Нет'}\n"
        text += f"   PIL доступен: {'✅ Да' if class_info.get('pil_available') else '❌ Нет'}\n\n"

        # Статистика по категориям
        stats = class_info.get('statistics', {})
        if stats:
            text += f"📈 Статистика по категориям:\n"
            text += f"   🍽️ Блюда и еда: {stats.get('food_classes', 0)}\n"
            text += f"   🍴 Посуда и приборы: {stats.get('tableware_classes', 0)}\n"
            text += f"   📦 Прочие объекты: {stats.get('other_classes', 0)}\n\n"

        # Информация о модели
        model_info = class_info.get('model_info', {})
        if model_info:
            text += f"🤖 Информация о модели:\n"
            text += f"   Путь: {model_info.get('path', 'Не указан')}\n"
            model_names = model_info.get('names', {})
            if model_names:
                text += f"   Классов в модели: {len(model_names)}\n\n"

        # Маппинг классов
        mapping = class_info.get('mapping', {})
        if mapping:
            text += f"🔤 Маппинг классов:\n"
            text += "-" * 40 + "\n"
            for class_id, class_name in sorted(mapping.items()):
                text += f"   {class_id:2d}: {class_name}\n"

        return text


class AboutDialog(QDialog):
    """Диалог О программе"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе")
        self.setFixedSize(500, 400)

        layout = QVBoxLayout()

        # Заголовок
        title = QLabel("<h2>🎯 YOLO Video Detector v2.0</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Информация о программе
        info_text = """
<p><b>Специализированный детектор для анализа видео с едой и посудой</b></p>

<p><b>🌟 Особенности:</b></p>
<ul>
<li>🇷🇺 Поддержка кириллических названий классов</li>
<li>🔄 Автоматический поворот MOV файлов на 90°</li>
<li>📊 Детальная статистика детекций</li>
<li>🎯 Фиксированное разрешение детекции 640x640</li>
<li>🍽️ Цветовая схема по типам еды</li>
<li>⚙️ Настройка параметров в реальном времени</li>
<li>📁 JSON конфигурация классов</li>
</ul>

<p><b>🔧 Технологии:</b></p>
<ul>
<li>Python 3.7+</li>
<li>PyQt5 (интерфейс)</li>
<li>Ultralytics YOLO (детекция)</li>
<li>OpenCV (обработка видео)</li>
<li>PIL/Pillow (кириллица)</li>
</ul>

<p><b>📁 Поддерживаемые форматы:</b></p>
<ul>
<li>Видео: mp4, avi, mov, mkv, wmv и др.</li>
<li>Модели: YOLO .pt файлы</li>
</ul>
        """

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)


class YOLOVideoDetector(QMainWindow):
    """Основное окно приложения"""

    def __init__(self):
        super().__init__()

        # Импорт UI и логики
        try:
            from designer import Ui_MainWindow
            from video_processor import VideoProcessor
        except ImportError as e:
            QMessageBox.critical(None, "Ошибка импорта",
                                 f"Не удалось импортировать модули:\n{e}\n\n"
                                 "Убедитесь, что файлы designer.py и video_processor.py "
                                 "находятся в той же папке что и detected_yolo.py")
            sys.exit(1)

        # Инициализация UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Настройка стилей для диалогов
        self.setup_application_style()

        # Процессор видео
        self.video_processor = VideoProcessor()

        # Пути к файлам
        self.model_path = ""
        self.video_path = ""
        self.output_path = ""

        # Настройка соединений
        self.setup_connections()

        # Обновление интерфейса
        self.update_ui_state()

        print("✅ Приложение YOLO Video Detector запущено")

    def setup_application_style(self):
        """Настройка темных стилей для всего приложения включая диалоги"""

        # Стили для диалоговых окон
        dialog_style = """
        QMessageBox {
            background-color: #2c3e50;
            color: #ecf0f1;
            border: 2px solid #34495e;
            border-radius: 8px;
        }

        QMessageBox QLabel {
            color: #ecf0f1;
            font-size: 13px;
            font-weight: normal;
            padding: 15px;
            min-width: 300px;
        }

        QMessageBox QPushButton {
            background-color: #3498db;
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            min-width: 90px;
            margin: 5px;
        }

        QMessageBox QPushButton:hover {
            background-color: #2980b9;
        }

        QMessageBox QPushButton:pressed {
            background-color: #21618c;
        }

        QDialog {
            background-color: #2c3e50;
            color: #ecf0f1;
            border: 2px solid #34495e;
            border-radius: 8px;
        }

        QDialog QLabel {
            color: #ecf0f1;
            font-weight: normal;
        }

        QDialog QPushButton {
            background-color: #27ae60;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 80px;
        }

        QDialog QPushButton:hover {
            background-color: #229954;
        }

        QFileDialog {
            background-color: #2c3e50;
            color: #ecf0f1;
        }

        QFileDialog QLabel {
            color: #ecf0f1;
        }

        QFileDialog QLineEdit {
            background-color: #34495e;
            color: #ecf0f1;
            border: 1px solid #555;
            padding: 5px;
            border-radius: 3px;
        }

        QFileDialog QListView {
            background-color: #34495e;
            color: #ecf0f1;
            border: 1px solid #555;
            alternate-background-color: #3a4a5c;
        }

        QFileDialog QTreeView {
            background-color: #34495e;
            color: #ecf0f1;
            border: 1px solid #555;
            alternate-background-color: #3a4a5c;
        }

        QFileDialog QPushButton {
            background-color: #3498db;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }

        QFileDialog QPushButton:hover {
            background-color: #2980b9;
        }
        """

        # Применяем стили ко всему приложению
        QApplication.instance().setStyleSheet(dialog_style)

    def create_styled_message_box(self, title, text, icon_type=QMessageBox.Information, buttons=QMessageBox.Ok):
        """Создать стилизованный QMessageBox с темной темой"""

        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(icon_type)
        msg.setStandardButtons(buttons)

        # Определяем цвет рамки в зависимости от типа сообщения
        border_colors = {
            QMessageBox.Information: "#27ae60",  # Зеленый
            QMessageBox.Warning: "#f39c12",  # Оранжевый
            QMessageBox.Critical: "#e74c3c",  # Красный
            QMessageBox.Question: "#3498db"  # Синий
        }

        button_colors = {
            QMessageBox.Information: "#27ae60",
            QMessageBox.Warning: "#f39c12",
            QMessageBox.Critical: "#e74c3c",
            QMessageBox.Question: "#3498db"
        }

        button_hover_colors = {
            QMessageBox.Information: "#229954",
            QMessageBox.Warning: "#e67e22",
            QMessageBox.Critical: "#c0392b",
            QMessageBox.Question: "#2980b9"
        }

        border_color = border_colors.get(icon_type, "#34495e")
        button_color = button_colors.get(icon_type, "#3498db")
        button_hover_color = button_hover_colors.get(icon_type, "#2980b9")

        # Применяем стили
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 3px solid {border_color};
                border-radius: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}

            QMessageBox QLabel {{
                color: #ecf0f1;
                font-size: 14px;
                font-weight: normal;
                padding: 20px;
                min-width: 350px;
                background-color: transparent;
                border: none;
            }}

            QMessageBox QPushButton {{
                background-color: {button_color};
                border: none;
                color: white;
                padding: 12px 25px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                min-width: 100px;
                margin: 5px;
            }}

            QMessageBox QPushButton:hover {{
                background-color: {button_hover_color};
                border: 1px solid {button_color};
            }}

            QMessageBox QPushButton:pressed {{
                background-color: #34495e;
                border: 2px solid {button_color};
            }}

            QMessageBox QIcon {{
                margin: 10px;
            }}
        """)

        return msg

    # Удобные методы для разных типов сообщений
    def show_info(self, title, message):
        """Показать информационное сообщение"""
        msg = self.create_styled_message_box(title, message, QMessageBox.Information)
        return msg.exec_()

    def show_warning(self, title, message):
        """Показать предупреждение"""
        msg = self.create_styled_message_box(title, message, QMessageBox.Warning)
        return msg.exec_()

    def show_error(self, title, message):
        """Показать ошибку"""
        msg = self.create_styled_message_box(title, message, QMessageBox.Critical)
        return msg.exec_()

    def show_question(self, title, message):
        """Показать вопрос"""
        msg = self.create_styled_message_box(title, message, QMessageBox.Question,
                                             QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        return msg.exec_()

    def setup_connections(self):
        """Настройка соединений сигналов и слотов"""

        # Кнопки выбора файлов
        self.ui.modelBtn.clicked.connect(self.select_model)
        self.ui.videoBtn.clicked.connect(self.select_video)
        self.ui.outputBtn.clicked.connect(self.select_output)

        # Управление обработкой
        self.ui.startBtn.clicked.connect(self.start_processing)
        self.ui.pauseBtn.clicked.connect(self.pause_processing)
        self.ui.stopBtn.clicked.connect(self.stop_processing)

        # Слайдеры настроек
        self.ui.confSlider.valueChanged.connect(self.update_confidence)
        self.ui.iouSlider.valueChanged.connect(self.update_iou)
        self.ui.fontSlider.valueChanged.connect(self.update_font_scale)
        self.ui.thicknessSlider.valueChanged.connect(self.update_line_thickness)

        # Чекбокс поворота
        self.ui.rotateMovCheckbox.toggled.connect(self.update_rotate_mov)

        # Сигналы процессора видео
        self.video_processor.frame_ready.connect(self.display_frame)
        self.video_processor.progress.connect(self.ui.progressBar.setValue)
        self.video_processor.status.connect(self.ui.statusLabel.setText)
        self.video_processor.finished.connect(self.processing_finished)
        self.video_processor.statistics_updated.connect(self.update_statistics)

        # Кнопки статистики
        self.ui.clearStatsBtn.clicked.connect(self.clear_statistics)
        self.ui.saveStatsBtn.clicked.connect(self.save_statistics)

        # Меню
        self.ui.actionOpenModel.triggered.connect(self.select_model)
        self.ui.actionOpenVideo.triggered.connect(self.select_video)
        self.ui.actionLoadYaml.triggered.connect(self.load_yaml_classes)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionShowAllClasses.triggered.connect(self.show_classes_info)
        self.ui.actionClearStats.triggered.connect(self.clear_statistics)
        self.ui.actionSaveStats.triggered.connect(self.save_statistics)
        self.ui.actionRotateMov.triggered.connect(self.ui.rotateMovCheckbox.toggle)
        self.ui.actionAbout.triggered.connect(self.show_about)

        # Синхронизация чекбокса и меню
        self.ui.rotateMovCheckbox.toggled.connect(self.ui.actionRotateMov.setChecked)
        self.ui.actionRotateMov.toggled.connect(self.ui.rotateMovCheckbox.setChecked)

    def select_model(self):
        """Выбор модели YOLO"""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель YOLO",
            current_dir,
            "YOLO модели (*.pt);;Все файлы (*)",
            options=QFileDialog.DontUseNativeDialog  # Ключевая опция!
        )

        if file_path:
            self.model_path = file_path
            model_name = Path(file_path).name
            self.ui.modelLabel.setText(f"Модель: {model_name}")

            # Загружаем информацию о модели
            self.video_processor.load_model_classes(file_path)
            self.update_model_info()
            self.update_ui_state()

            print(f"✅ Модель выбрана: {model_name}")

    def select_video(self):
        """Выбор видео файла"""
        # Расширенный фильтр с поддержкой как нижнего, так и верхнего регистра
        video_filter = (
            "Видео файлы ("
            "*.mp4 *.MP4 "
            "*.avi *.AVI "
            "*.mov *.MOV "
            "*.mkv *.MKV "
            "*.wmv *.WMV "
            "*.flv *.FLV "
            "*.webm *.WEBM "
            "*.m4v *.M4V "
            "*.3gp *.3GP "
            "*.mpg *.MPG "
            "*.mpeg *.MPEG "
            "*.ts *.TS "
            "*.mts *.MTS"
            ");;"
            "MOV файлы (*.mov *.MOV);;"
            "MP4 файлы (*.mp4 *.MP4);;"
            "AVI файлы (*.avi *.AVI);;"
            "Все файлы (*)"
        )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео файл", "", video_filter,
            options=QFileDialog.DontUseNativeDialog
        )

        if file_path:
            self.video_path = file_path
            video_name = Path(file_path).name
            self.ui.videoLabel.setText(f"Видео: {video_name}")

            # Проверяем видео файл
            video_info = self.video_processor.verify_video_file(file_path)
            if video_info['is_valid']:
                info_text = (f"Видео: {video_info['width']}x{video_info['height']}, "
                             f"{video_info['frame_count']} кадров, "
                             f"{video_info['fps']:.1f} FPS, "
                             f"{video_info['duration']:.1f}s")
                self.ui.statusLabel.setText(info_text)

                # Дополнительная проверка для MOV файлов
                if Path(file_path).suffix.upper() == '.MOV':
                    self.ui.statusLabel.setText(f"{info_text} - MOV файл обнаружен")

            else:
                self.show_warning("Ошибка", f"Проблема с видео: {video_info['error']}")
                return

            self.update_ui_state()
            print(f"✅ Видео выбрано: {video_name}")

            # Логирование для отладки MOV файлов
            file_extension = Path(file_path).suffix
            print(f"📁 Расширение файла: '{file_extension}'")
            if file_extension.upper() == '.MOV':
                print(f"🔄 MOV файл обнаружен, будет применен поворот: {self.ui.rotateMovCheckbox.isChecked()}")

    def select_output(self):
        """Выбор выходного файла"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить обработанное видео", "",current_dir,
            "MP4 видео (*.mp4);;AVI видео (*.avi);;MOV видео (*.mov);;Все файлы (*)",
            options=QFileDialog.DontUseNativeDialog
        )

        if file_path:
            self.output_path = file_path
            output_name = Path(file_path).name
            self.ui.outputLabel.setText(f"Выходной файл: {output_name}")
            print(f"✅ Выходной файл: {output_name}")

    def update_confidence(self, value):
        """Обновление значения confidence"""
        conf_value = value / 100.0
        self.ui.confLabel.setText(f"{conf_value:.2f}")
        self.video_processor.set_detection_params(confidence=conf_value)

    def update_iou(self, value):
        """Обновление значения IoU"""
        iou_value = value / 100.0
        self.ui.iouLabel.setText(f"{iou_value:.2f}")
        self.video_processor.set_detection_params(iou=iou_value)

    def update_font_scale(self, value):
        """Обновление размера шрифта"""
        font_scale = value / 10.0
        self.ui.fontLabel.setText(f"{font_scale:.1f}")
        self.video_processor.set_display_params(font_scale=font_scale)

    def update_line_thickness(self, value):
        """Обновление толщины линий"""
        self.ui.thicknessLabel.setText(str(value))
        self.video_processor.set_display_params(line_thickness=value)

    def update_rotate_mov(self, checked):
        """Обновление настройки поворота MOV"""
        self.video_processor.set_display_params(rotate_mov=checked)

    def start_processing(self):
        """Начать обработку видео"""
        if not self.model_path or not self.video_path:
            self.show_warning("Ошибка", "Выберите модель и видео файл!")
            return

        # Настраиваем процессор
        self.video_processor.set_files(self.model_path, self.video_path, self.output_path)

        # Запускаем обработку
        self.video_processor.start()

        # Обновляем интерфейс
        self.ui.startBtn.setEnabled(False)
        self.ui.pauseBtn.setEnabled(True)
        self.ui.stopBtn.setEnabled(True)

        print("🚀 Начата обработка видео")

    def pause_processing(self):
        """Пауза/продолжить обработку"""
        self.video_processor.pause_resume()

        if self.video_processor.paused:
            self.ui.pauseBtn.setText("▶️ Продолжить")
            self.ui.statusLabel.setText("⏸️ Пауза")
        else:
            self.ui.pauseBtn.setText("⏸️ Пауза")
            self.ui.statusLabel.setText("▶️ Обработка продолжена")

    def stop_processing(self):
        """Остановить обработку"""
        self.video_processor.stop()
        self.ui.statusLabel.setText("⏹️ Обработка остановлена")

    def processing_finished(self, message):
        """Обработка завершена"""
        self.ui.startBtn.setEnabled(True)
        self.ui.pauseBtn.setEnabled(False)
        self.ui.pauseBtn.setText("⏸️ Пауза")
        self.ui.stopBtn.setEnabled(False)
        self.ui.progressBar.setValue(100)

        self.ui.statusLabel.setText(message)

        if "успешно" in message:
            self.show_info("Готово", message)
        else:
            self.show_error("Ошибка", message)

        print(f"✅ {message}")

    def display_frame(self, frame):
        """Отображение кадра видео"""
        try:
            # Конвертируем BGR в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Создаем QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Масштабируем для отображения
            display_size = self.ui.videoDisplay.size()
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(display_size, Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation)

            self.ui.videoDisplay.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"⚠️ Ошибка отображения кадра: {e}")

    @pyqtSlot(str)
    def update_statistics(self, stats_text):
        """Обновление статистики детекций"""
        self.ui.statisticsText.setPlainText(stats_text)

    def clear_statistics(self):
        """Очистить статистику"""
        self.ui.statisticsText.setPlainText("📊 СТАТИСТИКА ДЕТЕКЦИЙ\n\n🔄 Статистика очищена")
        print("🗑️ Статистика очищена")

    def save_statistics(self):
        """Сохранить статистику в файл"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить статистику", "statistics.txt",current_dir,
            "Текстовые файлы (*.txt);;Все файлы (*)",
            options=QFileDialog.DontUseNativeDialog
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.ui.statisticsText.toPlainText())

                self.show_info("Готово", f"Статистика сохранена: {Path(file_path).name}")
                print(f"💾 Статистика сохранена: {file_path}")

            except Exception as e:
                self.show_warning("Ошибка", f"Не удалось сохранить файл: {e}")

    def load_yaml_classes(self):
        """Загрузить классы из YAML файла"""
        # Получаем путь к текущему каталогу
        current_dir = os.getcwd()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите YAML файл с классами", current_dir,  # Используем текущий каталог
            "YAML файлы (*.yaml *.yml);;Все файлы (*)"
        )

        if file_path:
            try:
                from class_mapping import load_classes_from_yaml
                if load_classes_from_yaml(file_path):
                    self.show_info("Готово", "Классы успешно загружены из YAML")
                    self.update_model_info()
                else:
                    self.show_warning("Ошибка", "Не удалось загрузить классы из YAML")
            except Exception as e:
                self.show_warning("Ошибка", f"Ошибка загрузки YAML: {e}")

    def show_classes_info(self):
        """Показать информацию о классах"""
        class_info = self.video_processor.get_class_mapping_info()
        dialog = ClassesInfoDialog(class_info, self)
        dialog.exec_()

    def show_about(self):
        """Показать диалог О программе"""
        dialog = AboutDialog(self)
        dialog.exec_()

    def update_model_info(self):
        """Обновить информацию о модели"""
        if self.model_path:
            class_info = self.video_processor.get_class_mapping_info()

            info_text = f"📁 Модель: {Path(self.model_path).name}\n"
            info_text += f"📊 Классов: {class_info.get('total_classes', 0)}\n"
            info_text += f"🎯 Разрешение: {class_info.get('detection_resolution', '640x640')}\n"
            info_text += f"🇷🇺 Кириллица: {'✅' if class_info.get('cyrillic_available') else '❌'}\n"

            # Статистика по категориям
            stats = class_info.get('statistics', {})
            if stats:
                info_text += f"\n📈 Категории:\n"
                info_text += f"🍽️ Еда: {stats.get('food_classes', 0)}\n"
                info_text += f"🍴 Посуда: {stats.get('tableware_classes', 0)}\n"
                info_text += f"📦 Другое: {stats.get('other_classes', 0)}"

            self.ui.modelInfoText.setPlainText(info_text)
        else:
            self.ui.modelInfoText.setPlainText("Модель не загружена")

    def update_ui_state(self):
        """Обновление состояния интерфейса"""
        has_model = bool(self.model_path)
        has_video = bool(self.video_path)

        # Кнопка старта доступна только при наличии модели и видео
        self.ui.startBtn.setEnabled(has_model and has_video and not self.video_processor.running)

        # Обновляем информацию о модели
        self.update_model_info()

    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        if self.video_processor.running:
            reply = self.show_question("Выход",
                                       "Обработка видео не завершена. Завершить принудительно?")

            if reply == QMessageBox.Yes:
                self.video_processor.stop()
                self.video_processor.wait(3000)  # Ждем 3 секунды
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def check_required_files():
    """Проверка наличия обязательных файлов"""
    required_files = {
        'video_processor.py': 'Логика обработки видео',
        'designer.py': 'Интерфейс PyQt5',
        'class_mapping.py': 'Маппинг классов',
        'classes_config.json': 'Конфигурация классов (JSON)'
    }

    missing_files = []
    found_files = []

    for file_name, description in required_files.items():
        if not (current_dir / file_name).exists():
            missing_files.append((file_name, description))
        else:
            found_files.append((file_name, description))

    # Показываем найденные файлы
    if found_files:
        print("✅ Найденные файлы:")
        for file_name, description in found_files:
            print(f"   📁 {file_name} - {description}")

    # Показываем отсутствующие файлы
    if missing_files:
        print("\n❌ Отсутствующие файлы:")
        for file_name, description in missing_files:
            print(f"   📁 {file_name} - {description}")

    return [file_name for file_name, _ in missing_files]


def run_gui_application():
    """Запуск GUI приложения"""
    print("🚀 Запуск GUI приложения...")

    try:
        # Создаем приложение
        app = QApplication(sys.argv)
        app.setApplicationName("YOLO Video Detector")
        app.setApplicationVersion("2.0")

        # Создаем и показываем главное окно
        window = YOLOVideoDetector()
        window.show()

        print("✅ Приложение готово к работе")

        # Запускаем цикл событий
        return app.exec_()

    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print("💡 Проверьте наличие всех файлов в папке проекта:")
        print("   - video_processor.py")
        print("   - designer.py")
        print("   - class_mapping.py")
        print("   - classes_config.json")
        return 1

    except Exception as e:
        print(f"\n❌ Ошибка при запуске GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Главная функция"""
    print("🚀 YOLO Video Detector v2.0 - Единый запускаемый файл")
    print("📦 Объединенная версия для PyCharm IDE")
    print("🍽️ Специализация: детекция блюд, напитков и посуды")
    print("🇷🇺 Поддержка кириллических названий классов")
    print("=" * 75)

    # Проверяем наличие classes_config.json
    if not check_classes_config():
        print("\n💡 Для корректной работы необходим файл classes_config.json")
        print("📋 Образец конфигурации:")
        print("   - Скопируйте прилагаемый classes_config.json")
        print("   - Или создайте свой на основе документации")
        return 1

    # Проверяем наличие основных модулей
    missing_files = check_required_files()

    if missing_files:
        print(f"\n❌ Не удается запустить приложение")
        print("💡 Убедитесь, что все файлы находятся в одной папке с detected_yolo.py")
        return 1

    print("\n✅ Все обязательные модули найдены")

    # Проверяем зависимости
    print("\n🔍 Проверка зависимостей:")
    if not check_dependencies():
        print("\n💡 Установите недостающие зависимости и перезапустите приложение")
        return 1

    print("\n✅ Все зависимости установлены")

    # Запускаем GUI приложение
    return run_gui_application()


if __name__ == "__main__":
    # Обработка аргументов командной строки
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg in ['--help', '-h', 'help']:
            show_help()
            sys.exit(0)

        elif arg == '--version':
            print("YOLO Video Detector v2.0 - Единый запускаемый файл")
            print("Специализация: детекция блюд, напитков и посуды")
            print("Поддержка кириллических названий классов")
            print("Интеграция с PyCharm IDE")
            sys.exit(0)

        elif arg == '--check':
            print("🔍 Проверка системы:")
            print("✅ Python:", sys.version)
            print("\n📦 Проверка файлов:")
            missing = check_required_files()
            if missing:
                print(f"❌ Отсутствуют: {missing}")
            else:
                print("✅ Все файлы найдены")

            print("\n📋 Проверка конфигурации:")
            check_classes_config()

            print("\n🔍 Проверка зависимостей:")
            check_dependencies()
            sys.exit(0)

        elif arg == '--config-info':
            print("📋 Информация о конфигурации classes_config.json:")
            if check_classes_config():
                try:
                    from class_mapping import print_config_info

                    print_config_info()
                except ImportError:
                    print("⚠️ Модуль class_mapping недоступен")
            sys.exit(0)

        else:
            print(f"❌ Неизвестный аргумент: {arg}")
            print("💡 Используйте --help для справки")
            sys.exit(1)

    # Запуск основного приложения
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        print("\n🔧 Рекомендации:")
        print("1. Проверьте наличие всех файлов проекта:")
        print("   📁 classes_config.json (обязательный)")
        print("   📁 video_processor.py")
        print("   📁 designer.py")
        print("2. Убедитесь в правильности установки зависимостей")
        print("3. Используйте --check для диагностики")
        print("4. Проверьте права доступа к файлам")
        print("5. Используйте --config-info для проверки конфигурации")
        sys.exit(1)