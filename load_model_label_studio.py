#!/usr/bin/env python3
"""
Альтернативный YOLO ML Backend для Label Studio
Работает с локальными файлами вместо URL
"""

import os
import sys
import json
import requests
from urllib.parse import urlparse

try:
    from ultralytics import YOLO
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.api import init_app
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    class LocalFileYOLOBackend(LabelStudioMLBase):
        def __init__(self, **kwargs):
            # Устанавливаем model_dir перед вызовом родительского __init__
            model_dir = os.path.dirname(os.path.abspath(__file__))
            kwargs['model_dir'] = model_dir

            super().__init__(**kwargs)

            # Параметры модели
            self.model_path = "/yolo_dataset_albumentations_v11n_geom/albumentations_training/weights/best.pt"
            self.confidence_threshold = 0.25
            self.iou_threshold = 0.45

            # Путь к данным Label Studio (настройте под ваш случай)
            self.label_studio_data_dir = "/home/asneg/.local/share/label-studio/media"

            # Устанавливаем model_dir явно во всех местах
            self.model_dir = model_dir
            if not hasattr(self, '_model_dir'):
                self._model_dir = model_dir

            logger.info(f"Инициализация Local File YOLO Backend")
            logger.info(f"Model dir: {self.model_dir}")
            logger.info(f"Label Studio data dir: {self.label_studio_data_dir}")
            logger.info(f"Модель: {self.model_path}")
            logger.info(f"Confidence: {self.confidence_threshold}")
            logger.info(f"IoU: {self.iou_threshold}")

            # Загружаем модель YOLO
            self.model = None
            self.class_names = {}
            self.load_model()

        def load_model(self):
            """Загрузка модели YOLO"""
            try:
                if os.path.exists(self.model_path):
                    logger.info(f"Загрузка модели YOLO...")
                    self.model = YOLO(self.model_path)
                    self.class_names = self.model.names if self.model else {}
                    logger.info(f"✅ Модель YOLO загружена успешно")
                    logger.info(f"Классы: {list(self.class_names.values())}")
                else:
                    logger.error(f"✗ Файл модели не найден: {self.model_path}")
                    raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            except Exception as e:
                logger.error(f"✗ Ошибка загрузки модели: {str(e)}")
                raise

        def find_local_image_path(self, image_path):
            """Поиск локального пути к изображению"""
            try:
                logger.info(f"Поиск локального файла для: {image_path}")

                # Если это уже локальный файл и он существует
                if os.path.exists(image_path):
                    logger.info(f"Найден прямой путь: {image_path}")
                    return image_path

                # Если это путь вида /data/upload/...
                if image_path.startswith('/data/'):
                    # Убираем /data/ и ищем в директории Label Studio
                    relative_path = image_path[6:]  # убираем '/data/'

                    # Возможные пути
                    possible_paths = [
                        os.path.join(self.label_studio_data_dir, relative_path),
                        os.path.join(self.label_studio_data_dir, "upload", relative_path),
                        os.path.join("/tmp/label-studio", relative_path),
                        os.path.join("/var/lib/label-studio", relative_path),
                        os.path.join(os.path.expanduser("~"), ".local/share/label-studio/media", relative_path)
                    ]

                    for path in possible_paths:
                        if os.path.exists(path):
                            logger.info(f"✅ Найден локальный файл: {path}")
                            return path
                        else:
                            logger.debug(f"Не найден: {path}")

                # Последняя попытка - поиск по имени файла
                if '/' in image_path:
                    filename = os.path.basename(image_path)
                    # Ищем файл рекурсивно в директории данных
                    for root, dirs, files in os.walk(self.label_studio_data_dir):
                        if filename in files:
                            found_path = os.path.join(root, filename)
                            logger.info(f"✅ Найден файл по имени: {found_path}")
                            return found_path

                logger.warning(f"❌ Локальный файл не найден для: {image_path}")
                return None

            except Exception as e:
                logger.error(f"Ошибка поиска локального файла: {e}")
                return None

        def predict(self, tasks, **kwargs):
            """Предсказание для задач Label Studio"""
            if not self.model:
                logger.error("Модель не загружена")
                return []

            logger.info(f"Обработка {len(tasks)} задач")
            predictions = []

            for i, task in enumerate(tasks):
                try:
                    # Получаем путь к изображению
                    image_path = task['data'].get('image')
                    if not image_path:
                        logger.warning(f"Задача {i}: нет изображения")
                        predictions.append({"result": []})
                        continue

                    # Ищем локальный файл
                    local_image_path = self.find_local_image_path(image_path)
                    if not local_image_path:
                        logger.error(f"Задача {i}: не удалось найти локальный файл для {image_path}")
                        predictions.append({"result": []})
                        continue

                    logger.info(f"Обработка локального изображения: {local_image_path}")

                    # Выполняем предсказание
                    results = self.model(
                        local_image_path,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )

                    # Преобразуем результаты в формат Label Studio
                    annotations = []

                    if results and len(results) > 0:
                        result = results[0]

                        # Получаем размеры изображения
                        img_height, img_width = result.orig_shape
                        logger.info(f"Размер изображения: {img_width}x{img_height}")

                        if result.boxes is not None and len(result.boxes) > 0:
                            logger.info(f"Найдено {len(result.boxes)} объектов")

                            for j, box in enumerate(result.boxes):
                                # Координаты bbox в формате xyxy
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                                # Конвертируем в проценты для Label Studio и приводим к обычному float
                                x = float((x1 / img_width) * 100)
                                y = float((y1 / img_height) * 100)
                                width = float(((x2 - x1) / img_width) * 100)
                                height = float(((y2 - y1) / img_height) * 100)

                                # Получаем класс и confidence
                                class_id = int(box.cls[0].cpu().numpy())
                                confidence = float(box.conf[0].cpu().numpy())
                                class_name = self.class_names.get(class_id, f"class_{class_id}")

                                logger.info(f"Объект {j}: {class_name} ({confidence:.2f})")

                                # Создаем аннотацию в формате Label Studio
                                annotation = {
                                    "from_name": "food_label",
                                    "to_name": "image",
                                    "type": "rectanglelabels",
                                    "value": {
                                        "x": x,
                                        "y": y,
                                        "width": width,
                                        "height": height,
                                        "rectanglelabels": [class_name]
                                    },
                                    "score": confidence
                                }

                                annotations.append(annotation)
                        else:
                            logger.info("Объекты не найдены")

                    predictions.append({
                        "result": annotations,
                        "score": 0.9 if annotations else 0.1
                    })

                except Exception as e:
                    logger.error(f"Ошибка предсказания для задачи {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Возвращаем пустой результат вместо краша
                    predictions.append({"result": []})

            logger.info(f"Завершено предсказание для {len(predictions)} задач")
            return predictions

        def fit(self, completions, workdir=None, **kwargs):
            """Дообучение модели (не реализовано)"""
            logger.info("Дообучение модели не реализовано")
            return {"model_path": self.model_path, "classes": self.class_names}

        def _job_dir(self, job_id):
            """Переопределяем метод для исправления ошибки с model_dir"""
            # Гарантируем, что model_dir всегда установлен
            if not hasattr(self, 'model_dir') or self.model_dir is None:
                self.model_dir = os.path.dirname(os.path.abspath(__file__))
                logger.warning(f"model_dir не был установлен, используем: {self.model_dir}")

            job_dir = os.path.join(self.model_dir, str(job_id))
            logger.info(f"Job dir для job_id {job_id}: {job_dir}")
            return job_dir

        def start_run(self, event, data, job_id):
            """Переопределяем метод для дополнительной проверки model_dir"""
            # Дополнительная проверка model_dir
            if not hasattr(self, 'model_dir') or self.model_dir is None:
                self.model_dir = os.path.dirname(os.path.abspath(__file__))
                logger.warning(f"model_dir восстановлен в start_run: {self.model_dir}")

            return super().start_run(event, data, job_id)


    if __name__ == "__main__":
        print("=" * 50)
        print("🚀 Local File YOLO ML Backend для Label Studio")
        print("=" * 50)
        print("Особенности:")
        print("✅ Работа с локальными файлами")
        print("✅ Автоматический поиск изображений")
        print("✅ Нет проблем с URL и сетью")
        print("✅ Поддержка различных путей Label Studio")
        print("=" * 50)

        try:
            # Создаем Flask приложение
            app = init_app(model_class=LocalFileYOLOBackend)

            print(f"🌐 Запуск Flask сервера на http://0.0.0.0:9090")
            print("Нажмите Ctrl+C для остановки")

            # Запускаем Flask сервер
            app.run(
                host="0.0.0.0",
                port=9090,
                debug=False,
                threaded=True
            )

        except KeyboardInterrupt:
            print("\n👋 Сервер остановлен")
        except Exception as e:
            print(f"❌ Ошибка запуска: {str(e)}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

except ImportError as e:
    print(f"❌ Отсутствует зависимость: {e}")
    print("Установите зависимости:")
    print("pip install redis flask label-studio-ml ultralytics torch")
    sys.exit(1)