#!/usr/bin/env python3
"""
YOLO Video Processor - Логика обработки видео с поддержкой кириллических названий классов
Фиксированное разрешение детекции: 640x640
Конфигурация загружается из JSON файла
+ Добавлена статистика детекций
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from collections import defaultdict, Counter

try:
    from ultralytics import YOLO

    print("✅ YOLO загружен")
except ImportError:
    print("❌ Установите: pip install ultralytics")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
    print("✅ PIL загружен для поддержки кириллицы")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL не найден. Установите: pip install Pillow")

# Импорт маппинга классов из JSON
try:
    from class_mapping import (
        CLASS_MAPPING, get_cyrillic_class_name, auto_detect_classes_from_model,
        get_class_by_synonym, is_food_class, is_tableware_class, get_class_color,
        get_coco_class_name, load_classes_from_yaml, check_fonts
    )

    CYRILLIC_MAPPING_AVAILABLE = True
    print("✅ Маппинг кириллических классов загружен из JSON")
except ImportError:
    CYRILLIC_MAPPING_AVAILABLE = False
    print("⚠️ Файл class_mapping.py не найден. Создайте его для поддержки кириллицы")

    # Создаем базовый маппинг
    CLASS_MAPPING = {
        0: "салат", 1: "суп", 2: "мясо", 3: "овощи", 4: "хлеб",
        5: "тарелка", 6: "чашка", 7: "стакан", 8: "background"
    }


    def get_cyrillic_class_name(class_id, fallback_name=None):
        return CLASS_MAPPING.get(class_id, fallback_name or f"класс_{class_id}")


    def get_class_by_synonym(synonym):
        return None


    def is_food_class(class_name):
        return class_name in ["салат", "суп", "мясо", "овощи", "хлеб"]


    def is_tableware_class(class_name):
        return class_name in ["тарелка", "чашка", "стакан"]


    def get_class_color(class_name):
        return "#808080"


    def get_coco_class_name(class_id):
        return None


def load_classes_from_yaml(yaml_path):
    """Загрузить классы из YAML файла датасета"""
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'names' in data:
            names = data['names']
            if isinstance(names, list):
                for i, name in enumerate(names):
                    CLASS_MAPPING[i] = name
            elif isinstance(names, dict):
                CLASS_MAPPING.update(names)

            print(f"✅ Загружено {len(CLASS_MAPPING)} классов из {Path(yaml_path).name}")
            for k, v in list(CLASS_MAPPING.items())[:5]:  # Показываем первые 5
                print(f"   {k}: {v}")
            if len(CLASS_MAPPING) > 5:
                print(f"   ... и еще {len(CLASS_MAPPING) - 5}")

    except Exception as e:
        print(f"⚠️ Не удалось загрузить классы из YAML: {e}")


def check_fonts():
    """Проверка доступности шрифтов с поддержкой кириллицы"""
    font_paths = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    available_fonts = []
    for font_path in font_paths:
        if os.path.exists(font_path):
            available_fonts.append(font_path)

    if available_fonts:
        print(f"✅ Найдены шрифты с поддержкой кириллицы: {available_fonts[:2]}")
    else:
        print("⚠️ Шрифты с поддержкой кириллицы не найдены")

    return available_fonts


def rotate_frame_90_clockwise(frame):
    """Поворот кадра на 90 градусов по часовой стрелке"""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


class DetectionStatistics:
    """Класс для ведения статистики детекций"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Сбросить статистику"""
        self.total_detections = 0
        self.food_count = 0
        self.tableware_count = 0
        self.other_count = 0
        self.class_counts = Counter()
        self.confidence_sum = 0
        self.high_confidence_count = 0  # conf >= 0.8
        self.medium_confidence_count = 0  # 0.5 <= conf < 0.8
        self.low_confidence_count = 0  # conf < 0.5

    def update(self, class_name, confidence):
        """
        Обновить статистику для одной детекции

        Args:
            class_name (str): Название класса
            confidence (float): Уверенность детекции
        """
        # Пропускаем фон и негативные образцы
        background_variants = ['background', 'Background', 'BACKGROUND', 'bg', 'BG',
                               'фон', 'Фон', 'ФОН', 'задний_план', 'задний план', 'negative']
        if str(class_name).lower() in [variant.lower() for variant in background_variants]:
            return

        self.total_detections += 1
        self.class_counts[class_name] += 1
        self.confidence_sum += confidence

        # Категоризация по типам
        if is_food_class(class_name):
            self.food_count += 1
        elif is_tableware_class(class_name):
            self.tableware_count += 1
        else:
            self.other_count += 1

        # Категоризация по уверенности
        if confidence >= 0.8:
            self.high_confidence_count += 1
        elif confidence >= 0.5:
            self.medium_confidence_count += 1
        else:
            self.low_confidence_count += 1

    def get_statistics_text(self):
        """
        Получить текстовое представление статистики

        Returns:
            str: Форматированная статистика
        """
        if self.total_detections == 0:
            return "📊 СТАТИСТИКА ДЕТЕКЦИЙ\n\nНет детекций на текущем кадре"

        avg_confidence = self.confidence_sum / self.total_detections if self.total_detections > 0 else 0

        stats_text = f"""📊 СТАТИСТИКА ДЕТЕКЦИЙ НА СТОЛЕ

📈 Общая информация:
  Всего объектов: {self.total_detections}
  Средняя уверенность: {avg_confidence:.2f}

🍽️ По категориям:
  🥗 Блюда и еда: {self.food_count} ({self.food_count / self.total_detections * 100:.1f}%)
  🍴 Посуда и приборы: {self.tableware_count} ({self.tableware_count / self.total_detections * 100:.1f}%)
  📦 Прочие объекты: {self.other_count} ({self.other_count / self.total_detections * 100:.1f}%)

🎯 По уверенности:
  🟢 Высокая (≥80%): {self.high_confidence_count}
  🟡 Средняя (50-79%): {self.medium_confidence_count}
  🔴 Низкая (<50%): {self.low_confidence_count}

📋 Детали по классам:"""

        # Сортируем классы по количеству детекций
        sorted_classes = self.class_counts.most_common()

        for class_name, count in sorted_classes[:10]:  # Показываем топ 10
            percentage = count / self.total_detections * 100

            # Добавляем иконку в зависимости от типа
            if is_food_class(class_name):
                icon = "🍽️"
            elif is_tableware_class(class_name):
                icon = "🍴"
            else:
                icon = "📦"

            stats_text += f"\n  {icon} {class_name}: {count} ({percentage:.1f}%)"

        if len(sorted_classes) > 10:
            stats_text += f"\n  ... и еще {len(sorted_classes) - 10} классов"

        return stats_text


class VideoProcessor(QThread):
    """Класс для обработки видео с YOLO моделями"""

    # Сигналы для связи с UI
    frame_ready = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    statistics_updated = pyqtSignal(str)  # Новый сигнал для статистики

    def __init__(self):
        super().__init__()

        # Пути к файлам
        self.model_path = ""
        self.video_path = ""
        self.output_path = ""

        # Состояние обработки
        self.running = False
        self.paused = False

        # Настройки детекции
        self.confidence = 0.25
        self.iou = 0.45

        # Настройки отображения
        self.font_scale = 1.5
        self.line_thickness = 3
        self.rotate_mov = True

        # Пользовательский маппинг классов
        self.custom_class_mapping = {}

        # Информация о модели
        self.model_info = {}

        # Фиксированное разрешение для детекции
        self.detection_size = 640  # Всегда 640x640

        # Статистика детекций
        self.statistics = DetectionStatistics()

    def set_files(self, model_path, video_path, output_path=None):
        """Установить пути файлов"""
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path

        # Загружаем классы при установке модели
        if model_path and os.path.exists(model_path):
            self.load_model_classes(model_path)

    def set_detection_params(self, confidence=None, iou=None):
        """Установить параметры детекции"""
        if confidence is not None:
            self.confidence = confidence
        if iou is not None:
            self.iou = iou

    def set_display_params(self, font_scale=None, line_thickness=None, rotate_mov=None):
        """Установить параметры отображения"""
        if font_scale is not None:
            self.font_scale = font_scale
        if line_thickness is not None:
            self.line_thickness = line_thickness
        if rotate_mov is not None:
            self.rotate_mov = rotate_mov

    def load_model_classes(self, model_path):
        """Загрузить классы из модели"""
        try:
            # Загружаем модель для получения информации о классах
            model = YOLO(model_path)
            self.model_info = {
                'names': getattr(model, 'names', {}),
                'path': model_path
            }

            print(f"📋 Модель загружена: {Path(model_path).name}")
            if self.model_info['names']:
                print(f"   Классов в модели: {len(self.model_info['names'])}")

                # Показываем первые несколько классов
                for i, (class_id, class_name) in enumerate(list(self.model_info['names'].items())[:5]):
                    print(f"   {class_id}: {class_name}")
                if len(self.model_info['names']) > 5:
                    print(f"   ... и еще {len(self.model_info['names']) - 5}")

            # Попытка автоматического извлечения из модели
            if CYRILLIC_MAPPING_AVAILABLE:
                auto_detect_classes_from_model(model_path)

            # Поиск соответствующего YAML файла
            model_dir = Path(model_path).parent
            possible_yaml_files = []

            # Ищем в папке с моделью и родительских папках
            for search_dir in [model_dir, model_dir.parent, model_dir.parent.parent]:
                if search_dir.exists():
                    yaml_files = list(search_dir.glob("*dataset*.yaml")) + list(search_dir.glob("*data*.yaml"))
                    possible_yaml_files.extend(yaml_files)

            # Загружаем из первого найденного YAML
            for yaml_file in possible_yaml_files:
                if yaml_file.exists():
                    print(f"🔍 Найден YAML файл: {yaml_file}")
                    load_classes_from_yaml(yaml_file)
                    break

        except Exception as e:
            print(f"⚠️ Ошибка загрузки классов: {e}")

    def is_mov_file(self, video_path):
        """Проверка, является ли файл MOV форматом"""
        return Path(video_path).suffix.lower() in ['.mov', '.MOV']

    def resize_frame_for_detection(self, frame):
        """
        Изменить размер кадра до 640x640 для детекции с сохранением пропорций

        Args:
            frame: Исходный кадр

        Returns:
            tuple: (кадр_640x640, коэффициент_масштабирования, x_смещение, y_смещение)
        """
        original_height, original_width = frame.shape[:2]

        # Вычисляем коэффициент масштабирования для сохранения пропорций
        scale = min(self.detection_size / original_width, self.detection_size / original_height)

        # Новые размеры с сохранением пропорций
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Изменяем размер
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Создаем кадр 640x640 с черными полосами
        result = np.zeros((self.detection_size, self.detection_size, 3), dtype=np.uint8)

        # Центрируем изображение
        y_offset = (self.detection_size - new_height) // 2
        x_offset = (self.detection_size - new_width) // 2

        result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        return result, scale, x_offset, y_offset

    def adjust_detections_to_original(self, detections, scale, x_offset, y_offset, original_width, original_height):
        """
        Преобразовать координаты детекций от 640x640 обратно к оригинальному разрешению
        """
        if not detections or len(detections) == 0:
            return detections

        # Создаем новые результаты вместо изменения существующих
        adjusted_results = []

        for result in detections:
            if result.boxes is not None and len(result.boxes) > 0:
                try:
                    # Получаем исходные данные
                    if hasattr(result.boxes, 'data'):
                        # Новые версии ultralytics
                        original_data = result.boxes.data.clone()
                        boxes_data = original_data[:, :4]  # xyxy координаты

                        # Корректируем координаты
                        boxes_data[:, [0, 2]] -= x_offset  # x координаты
                        boxes_data[:, [1, 3]] -= y_offset  # y координаты
                        boxes_data[:, [0, 2]] /= scale  # x координаты
                        boxes_data[:, [1, 3]] /= scale  # y координаты

                        # Ограничиваем координаты
                        boxes_data[:, [0, 2]] = boxes_data[:, [0, 2]].clamp(0, original_width)
                        boxes_data[:, [1, 3]] = boxes_data[:, [1, 3]].clamp(0, original_height)

                        # Обновляем данные
                        original_data[:, :4] = boxes_data
                        result.boxes.data = original_data

                    else:
                        # Старые версии - пропускаем корректировку координат
                        pass

                except Exception as e:
                    # Если не удалось скорректировать, продолжаем с исходными координатами
                    pass

            adjusted_results.append(result)

        return adjusted_results

    def get_cyrillic_class_name_for_detection(self, class_id, original_name=None):
        """
        Получить название класса на кириллице для детекции
        Приоритет:
        1. Пользовательский маппинг
        2. COCO маппинг (загружается из JSON)
        3. Английский маппинг (синонимы)
        4. Label Studio маппинг (CLASS_MAPPING)
        5. Проверка на кириллицу
        6. Fallback
        """

        # 1. Пользовательский маппинг (высший приоритет)
        if class_id in self.custom_class_mapping:
            return self.custom_class_mapping[class_id]

        # 2. COCO маппинг из JSON файла
        if CYRILLIC_MAPPING_AVAILABLE:
            coco_name = get_coco_class_name(class_id)
            if coco_name:
                return coco_name

        # 3. Английский маппинг (синонимы из JSON)
        if original_name:
            original_lower = str(original_name).lower().strip()
            synonym_result = get_class_by_synonym(original_lower)
            if synonym_result:
                return synonym_result

        # 4. Label Studio маппинг (CLASS_MAPPING)
        if class_id in CLASS_MAPPING:
            return CLASS_MAPPING[class_id]

        # 5. Если оригинальное название уже на кириллице, используем его
        if original_name and any(ord(char) > 127 for char in str(original_name)):
            return str(original_name)

        # 6. Fallback - используем функцию из class_mapping
        return get_cyrillic_class_name(class_id, original_name)

    def draw_text_with_pil(self, img, text, position, conf=0.0):
        """Отрисовка текста с поддержкой кириллицы через PIL"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            # Вычисляем размер шрифта на основе font_scale
            base_font_size = 32
            font_size = int(base_font_size * self.font_scale)

            # Цвета
            text_color = (255, 255, 255)  # Белый текст
            stroke_color = (0, 0, 0)  # Черная обводка
            stroke_width = max(2, int(self.font_scale))

            # Очищаем и проверяем текст
            try:
                # Убираем проблемные символы
                clean_text = ''.join(char for char in text if ord(char) < 65536 and char.isprintable())
                if not clean_text or len(clean_text.strip()) == 0:
                    clean_text = f"Объект: {conf:.2f}"
            except Exception:
                clean_text = f"Объект: {conf:.2f}"

            # Конвертируем BGR в RGB для PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            # Пытаемся загрузить шрифт с поддержкой кириллицы
            font = None
            font_paths_to_try = [
                # Linux
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]

            for font_path in font_paths_to_try:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                except Exception:
                    continue

            # Если не удалось загрузить TTF шрифт, используем стандартный
            if font is None:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    return self.draw_text_with_opencv(img, clean_text, position, conf)

            x, y = position

            # Рисуем обводку
            if stroke_width > 0:
                for adj_x in range(-stroke_width, stroke_width + 1):
                    for adj_y in range(-stroke_width, stroke_width + 1):
                        if adj_x != 0 or adj_y != 0:
                            try:
                                draw.text((x + adj_x, y + adj_y), clean_text, font=font, fill=stroke_color)
                            except Exception:
                                pass

            # Рисуем основной текст
            try:
                draw.text((x, y), clean_text, font=font, fill=text_color)
            except Exception:
                draw.text((x, y), f"Класс: {conf:.2f}", font=font, fill=text_color)

            # Конвертируем обратно в BGR для OpenCV
            img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img_with_text

        except ImportError:
            return self.draw_text_with_opencv(img, text, position, conf)
        except Exception as e:
            return self.draw_text_with_opencv(img, text, position, conf)

    def draw_text_with_opencv(self, img, text, position, conf=0.0):
        """Fallback метод рисования текста через OpenCV"""
        try:
            safe_text = text if isinstance(text, str) else f"Класс: {conf:.2f}"

            cv2.putText(img, safe_text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0),
                        self.line_thickness + 2)  # Черная обводка
            cv2.putText(img, safe_text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255),
                        max(1, self.line_thickness - 1))  # Белый текст
        except Exception:
            cv2.putText(img, f"Объект: {conf:.2f}", position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    def draw_detections(self, frame, results):
        """Отрисовка детекций с кириллическими названиями и обновление статистики"""
        if not results or len(results) == 0:
            # Сбрасываем статистику если нет детекций
            self.statistics.reset()
            self.statistics_updated.emit(self.statistics.get_statistics_text())
            return frame

        # Сбрасываем статистику для нового кадра
        self.statistics.reset()

        # Цвета для разных типов объектов
        food_color = (0, 255, 0)  # Зеленый для еды
        tableware_color = (255, 0, 0)  # Красный для посуды
        default_color = (255, 255, 0)  # Желтый для прочего

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # Получаем названия классов из модели
            model_class_names = self.model_info.get('names', {})

            for i, box in enumerate(boxes):
                try:
                    # Получаем координаты (совместимость с разными версиями ultralytics)
                    if hasattr(box, 'xyxy') and box.xyxy is not None:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                    elif hasattr(box, 'data') and box.data is not None:
                        # Для новых версий ultralytics
                        data = box.data.cpu().numpy()
                        x1, y1, x2, y2 = data[:4].astype(int)
                        conf = data[4]
                        cls = int(data[5])
                    else:
                        # Если структура неизвестна, пропускаем
                        continue

                    if conf < self.confidence:
                        continue

                    # Получаем оригинальное название из модели
                    original_class_name = model_class_names.get(cls, f"class_{cls}")

                    # Получаем кириллическое название через улучшенную функцию
                    cyrillic_class_name = self.get_cyrillic_class_name_for_detection(cls, original_class_name)

                    # Проверяем валидность названия
                    if not cyrillic_class_name or len(str(cyrillic_class_name).strip()) == 0:
                        cyrillic_class_name = f"класс_{cls}"

                    # Пропускаем фон и негативные образцы для отрисовки, но не для статистики
                    background_variants = ['background', 'Background', 'BACKGROUND', 'bg', 'BG',
                                           'фон', 'Фон', 'ФОН', 'задний_план', 'задний план', 'negative']
                    is_background = str(cyrillic_class_name).lower() in [variant.lower() for variant in
                                                                         background_variants]

                    # Обновляем статистику (включая фон для полной картины)
                    self.statistics.update(cyrillic_class_name, conf)

                    if is_background:
                        continue  # Не отрисовываем фон

                    # Определяем тип объекта и цвет согласно конфигурации
                    if is_food_class(cyrillic_class_name):
                        icon = "🍽️"
                        color = food_color
                        thickness = self.line_thickness + 1
                    elif is_tableware_class(cyrillic_class_name):
                        icon = "🍴"
                        color = tableware_color
                        thickness = self.line_thickness
                    else:
                        icon = "📦"
                        color = default_color
                        thickness = self.line_thickness

                    # Используем цвет из JSON конфигурации если доступен
                    try:
                        config_color = get_class_color(cyrillic_class_name)
                        if config_color and config_color != "#808080":
                            # Конвертируем hex в BGR
                            hex_color = config_color.lstrip('#')
                            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
                            color = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR
                    except Exception:
                        pass  # Используем стандартный цвет

                    # Рисуем прямоугольник
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    # Создаем кириллическую метку с иконкой для высокой уверенности
                    if conf >= 0.8:
                        label = f"{icon} {cyrillic_class_name}: {conf:.2f}"
                    else:
                        label = f"{cyrillic_class_name}: {conf:.2f}"

                    # Рисуем фон для текста
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, 2)[0]

                    bg_x1 = x1
                    bg_y1 = y1 - label_size[1] - 10
                    bg_x2 = x1 + label_size[0] + 10
                    bg_y2 = y1

                    # Полупрозрачный фон
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                    alpha = 0.7
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Отрисовываем текст с поддержкой кириллицы
                    frame = self.draw_text_with_pil(frame, label, (x1 + 5, y1 - label_size[1] - 5), conf=conf)

                except Exception as e:
                    print(f"⚠️ Ошибка обработки детекции {i}: {e}")
                    continue

        # Отправляем обновленную статистику
        self.statistics_updated.emit(self.statistics.get_statistics_text())

        return frame

    def verify_video_file(self, video_path):
        """Проверка видео файла"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if frame_count > 0 and fps > 0:
                    duration = frame_count / fps

                    # Информация о видео
                    video_info = {
                        'width': width,
                        'height': height,
                        'frame_count': frame_count,
                        'fps': fps,
                        'duration': duration,
                        'is_valid': True
                    }

                    cap.release()
                    return video_info
                else:
                    cap.release()
                    return {'is_valid': False, 'error': 'Поврежденный файл'}
            else:
                return {'is_valid': False, 'error': 'Не удалось открыть файл'}

        except Exception as e:
            return {'is_valid': False, 'error': str(e)}

    def pause_resume(self):
        """Пауза/продолжить"""
        self.paused = not self.paused

    def stop(self):
        """Остановить"""
        self.running = False

    def run(self):
        """Основная обработка видео с фиксированным разрешением детекции 640x640"""
        self.running = True

        try:
            # Загрузить модель
            self.status.emit("Загрузка модели...")
            model = YOLO(self.model_path)
            self.status.emit("Модель загружена")

            # Вывести информацию о классах модели
            if hasattr(model, 'names') and model.names:
                print(f"📋 Классы в модели: {len(model.names)}")
                # Проверяем тип модели
                if len(model.names) == 80 and 'person' in model.names.values():
                    print("🎯 Обнаружена стандартная COCO модель")
                    self.status.emit("COCO модель - используется COCO маппинг из JSON")
                elif len(model.names) == 31:
                    print("🍽️ Обнаружена модель для блюд (31 класс)")
                    self.status.emit("Модель для блюд - используется конфигурация из JSON")
                else:
                    print(f"📊 Пользовательская модель ({len(model.names)} классов)")
                    self.status.emit(f"Пользовательская модель ({len(model.names)} классов)")

            # Проверяем, нужно ли поворачивать видео
            should_rotate = self.is_mov_file(self.video_path) and self.rotate_mov
            if should_rotate:
                self.status.emit("Обнаружен MOV файл - будет применен поворот на 90°")

            # Открыть видео
            self.status.emit("Открытие видео...")
            cap = cv2.VideoCapture(self.video_path)

            if not cap.isOpened():
                self.finished.emit("Ошибка: не удалось открыть видео")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Если поворачиваем, меняем размеры местами для выходного видео
            output_width = height if should_rotate else width
            output_height = width if should_rotate else height

            self.status.emit(f"Видео: {width}x{height} → Детекция: 640x640, {total_frames} кадров, {fps:.1f} FPS")
            if should_rotate:
                self.status.emit(f"После поворота: {output_width}x{output_height}")

            # Выходное видео
            out = None
            if self.output_path:
                output_ext = Path(self.output_path).suffix.lower()

                if output_ext == '.mp4':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                elif output_ext == '.avi':
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                elif output_ext == '.mov':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                elif output_ext == '.mkv':
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                elif output_ext == '.wmv':
                    fourcc = cv2.VideoWriter_fourcc(*'WMV2')
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                self.status.emit(f"Создание выходного файла: {Path(self.output_path).name}")
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (output_width, output_height))

                if not out.isOpened():
                    self.status.emit("⚠️ Проблемы с VideoWriter, пробую альтернативный кодек...")
                    fourcc_alt = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(self.output_path, fourcc_alt, fps, (output_width, output_height))
                    if not out.isOpened():
                        self.finished.emit("❌ Не удалось создать выходной видео файл")
                        return

            frame_count = 0

            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Поворачиваем кадр, если это MOV файл
                if should_rotate:
                    frame = rotate_frame_90_clockwise(frame)

                # Получаем текущие размеры кадра (после возможного поворота)
                current_height, current_width = frame.shape[:2]

                # Изменяем размер кадра до 640x640 для детекции
                detection_frame, scale, x_offset, y_offset = self.resize_frame_for_detection(frame)

                # Детекция на кадре 640x640
                results = model(detection_frame, conf=self.confidence, iou=self.iou, verbose=False)

                # Корректируем координаты детекций обратно к оригинальному размеру
                results = self.adjust_detections_to_original(
                    results, scale, x_offset, y_offset, current_width, current_height
                )

                # Отрисовка детекций на оригинальном кадре (с обновлением статистики)
                processed_frame = self.draw_detections(frame, results)

                # Сохранение
                if out is not None:
                    out.write(processed_frame)

                # Отправка кадра
                self.frame_ready.emit(processed_frame)

                # Прогресс
                progress_val = int((frame_count / total_frames) * 100)
                self.progress.emit(progress_val)

                # Пауза для отображения
                self.msleep(30)

            cap.release()
            if out:
                out.release()

            rotation_info = " (с поворотом на 90°)" if should_rotate else ""
            self.finished.emit(f"Обработка завершена успешно!{rotation_info} (Детекция: 640x640)")

        except Exception as e:
            self.finished.emit(f"Ошибка: {e}")

    def get_class_mapping_info(self):
        """Получить информацию о загруженных классах"""
        info = {
            'total_classes': len(CLASS_MAPPING),
            'mapping': dict(CLASS_MAPPING),
            'cyrillic_available': CYRILLIC_MAPPING_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'model_info': self.model_info,
            'detection_resolution': f"{self.detection_size}x{self.detection_size}"
        }

        # Добавляем статистику по типам классов
        food_count = 0
        tableware_count = 0
        other_count = 0

        for class_name in CLASS_MAPPING.values():
            if is_food_class(class_name):
                food_count += 1
            elif is_tableware_class(class_name):
                tableware_count += 1
            else:
                other_count += 1

        info['statistics'] = {
            'food_classes': food_count,
            'tableware_classes': tableware_count,
            'other_classes': other_count
        }

        return info