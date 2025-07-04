#!/usr/bin/env python3
"""
YOLO обучение с аугментацией данных на основе Albumentations
Упрощенная версия только с профессиональной аугментацией
"""

import os
import sys
import json
import requests
import yaml
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import cv2





# ===========================================
# НАСТРОЙКИ
# ===========================================

SESSION_ID = ".eJxVj81uxCAMhN-FcxLxG0KOvfcZIgMmoRvBKiRSu6u-e6Hay95sz_gbzZNc0ZOZGEeZkCPvDVDbS21CDyB5nZwHbjin1JGO5GOFFB9wxpyW-43MrCM7lHPZ8xpTXbVqGCXl0HhUd2SB69yWq-Cx_Ccx8naz4G6YmuC_IK15cDmdR7RDswwvtQyf2eP-8fK-ATYoW_0OcnLcKe3DNBkInDGNKFAIJYKkI1iL0nHwganRCEQmleXV4ClVyqJp0IKltGL4fY_HD5mlqLXp7x9f41sv:1uWTX9:QTo4Fp90Idh9tTt6izl8sM6VZuQNYTCjes0asHuJOKk"
LABEL_STUDIO_URL = "http://localhost:8080"
PROJECT_NAME = "New Project #3"
OUTPUT_DIR = "./yolo_dataset_albumentations_v11n_au_full"

# Настройки для обучения
CONFIG = {
    'epochs': 150,
    'batch_size': 10,
    'image_size': 320,
    'learning_rate': 0.001,
    'device': 'cpu',
    'model': 'yolo11n.pt',
    'patience': 20,
    'max_tasks': None,

    # НАСТРОЙКИ АУГМЕНТАЦИИ ALBUMENTATIONS
    'augmentation': {
        'enabled': True,
        'multiplier': 30,  # Во сколько раз увеличить данные
        'min_visibility': 0.8,  # Минимальная видимость bbox
        'min_area': 0.05,  # Минимальная площадь bbox

        # Параметры трансформаций
        'geometric': {
            'horizontal_flip': 0.0,
            'vertical_flip': 0.0,
            'rotate_limit': 15,
            'rotate_prob': 0.4,
            'shift_limit': 0.2,
            'scale_limit': 0.3,
            'shear_prob': 0.3,
        },
        'color': {
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'brightness_contrast_prob': 0.4,
            'hue_shift_limit': 10,
            'sat_shift_limit': 20,
            'val_shift_limit': 10,
            'hsv_prob': 0.3,
        },
        'blur_noise': {
            'blur_limit': 3,
            'blur_prob': 0.2,
            'gauss_noise_var': (10, 30),
            'noise_prob': 0.2,
        },
        'weather': {
            'rain_prob': 0.1,
            'fog_prob': 0.1,
            'sun_flare_prob': 0.05,
        }
    }
}

# Проверка ULTRALYTICS
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Установите ultralytics: pip install ultralytics")
    sys.exit(1)

# Проверка ALBUMENTATIONS (обязательно)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    print("✅ Albumentations загружен успешно")
except ImportError:
    print("❌ Установите albumentations: pip install albumentations opencv-python")
    print("   Этот скрипт требует Albumentations для работы!")
    sys.exit(1)


def create_albumentations_pipeline():
    """Создать настраиваемый пайплайн аугментации Albumentations"""

    aug_config = CONFIG['augmentation']
    transforms = []

    # ==============================================
    # ГЕОМЕТРИЧЕСКИЕ ТРАНСФОРМАЦИИ (можно отключить)
    # ==============================================
    ENABLE_GEOMETRIC = True  # ❌ Установить False для отключения всех геометрических

    if ENABLE_GEOMETRIC:
        # Отражения
        if aug_config['geometric']['horizontal_flip'] > 0:
            transforms.append(A.HorizontalFlip(p=aug_config['geometric']['horizontal_flip']))

        if aug_config['geometric']['vertical_flip'] > 0:
            transforms.append(A.VerticalFlip(p=aug_config['geometric']['vertical_flip']))

        # Повороты
        if aug_config['geometric']['rotate_prob'] > 0 and aug_config['geometric']['rotate_limit'] > 0:
            transforms.append(
                A.Rotate(
                    limit=aug_config['geometric']['rotate_limit'],
                    p=aug_config['geometric']['rotate_prob'],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            )

        # Комбинированные трансформации
        if (aug_config['geometric']['shift_limit'] > 0 or
                aug_config['geometric']['scale_limit'] > 0):
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=aug_config['geometric']['shift_limit'],
                    scale_limit=aug_config['geometric']['scale_limit'],
                    rotate_limit=0,  # Отдельно управляется выше
                    p=0.4,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            )

        # Дополнительные геометрические эффекты
        ENABLE_ADVANCED_GEOMETRIC = True  # ❌ Отключить продвинутые эффекты
        if ENABLE_ADVANCED_GEOMETRIC:
            transforms.append(A.Perspective(scale=(0.02, 0.05), p=0.2))
            transforms.append(
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.2,
                    border_mode=cv2.BORDER_CONSTANT, value=0
                )
            )

    # ==============================================
    # ЦВЕТОВЫЕ ТРАНСФОРМАЦИИ (можно отключить)
    # ==============================================
    ENABLE_COLOR = True  # ❌ Установить False для отключения всех цветовых

    if ENABLE_COLOR:
        # Базовые цветовые трансформации
        if aug_config['color']['brightness_contrast_prob'] > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=aug_config['color']['brightness_limit'],
                    contrast_limit=aug_config['color']['contrast_limit'],
                    p=aug_config['color']['brightness_contrast_prob']
                )
            )

        # HSV трансформации
        if aug_config['color']['hsv_prob'] > 0:
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=aug_config['color']['hue_shift_limit'],
                    sat_shift_limit=aug_config['color']['sat_shift_limit'],
                    val_shift_limit=aug_config['color']['val_shift_limit'],
                    p=aug_config['color']['hsv_prob']
                )
            )

        # Дополнительные цветовые эффекты
        ENABLE_ADVANCED_COLOR = False  # ❌ Отключить продвинутые цветовые эффекты
        if ENABLE_ADVANCED_COLOR:
            transforms.extend([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.ToGray(p=0.05),
                A.ChannelShuffle(p=0.1),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
            ])

    # ==============================================
    # РАЗМЫТИЕ И ШУМ (можно отключить)
    # ==============================================
    ENABLE_BLUR_NOISE = True  # ❌ Отключить размытие и шум

    if ENABLE_BLUR_NOISE and aug_config['blur_noise']['blur_prob'] > 0:
        # Размытие
        if aug_config['blur_noise']['blur_limit'] > 0:
            transforms.append(
                A.OneOf([
                    A.Blur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
                    A.GaussianBlur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
                    A.MotionBlur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
                ], p=aug_config['blur_noise']['blur_prob'])
            )

        # Шум
        if aug_config['blur_noise']['noise_prob'] > 0:
            transforms.append(
                A.OneOf([
                    A.GaussNoise(var_limit=aug_config['blur_noise']['gauss_noise_var'], p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=aug_config['blur_noise']['noise_prob'])
            )

    # ==============================================
    # ПОГОДНЫЕ ЭФФЕКТЫ (можно отключить)
    # ==============================================
    ENABLE_WEATHER = False  # ❌ Отключить все погодные эффекты

    if ENABLE_WEATHER:
        weather_transforms = []

        if aug_config['weather']['rain_prob'] > 0:
            weather_transforms.append(
                A.RandomRain(
                    slant_lower=-10, slant_upper=10, drop_length=10, drop_width=1,
                    drop_color=(200, 200, 200), blur_value=1, brightness_coefficient=0.8,
                    rain_type="drizzle", p=1.0
                )
            )

        if aug_config['weather']['fog_prob'] > 0:
            weather_transforms.append(
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=1.0)
            )

        if aug_config['weather']['sun_flare_prob'] > 0:
            weather_transforms.append(
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
                    num_flare_circles_lower=6, num_flare_circles_upper=10,
                    src_radius=160, src_color=(255, 255, 255), p=1.0
                )
            )

        if weather_transforms:
            transforms.append(A.OneOf(weather_transforms, p=0.1))  # Низкая вероятность

    # ==============================================
    # ДОПОЛНИТЕЛЬНЫЕ ЭФФЕКТЫ (можно отключить)
    # ==============================================
    ENABLE_ADDITIONAL = False  # ❌ Отключить дополнительные эффекты

    if ENABLE_ADDITIONAL:
        transforms.extend([
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
            A.Downscale(scale_min=0.7, scale_max=0.9, p=0.1),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2,
                shadow_dimension=5, p=0.1
            ),
        ])

    # Проверка что хотя бы одна трансформация включена
    if not transforms:
        print("⚠️ Все трансформации отключены! Добавляем минимальную...")
        transforms.append(A.HorizontalFlip(p=0.5))  # Минимальная трансформация

    # Создаем композицию
    composition = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=aug_config['min_visibility'],
            min_area=aug_config['min_area'],
        )
    )

    print(f"✅ Создан пайплайн с {len(transforms)} трансформациями")
    return composition


def apply_albumentations_augmentation(image_path, labels, class_names, num_augmentations):
    """Применить аугментацию Albumentations к изображению"""

    try:
        # Загрузить изображение
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Парсинг YOLO меток
        bboxes = []
        class_labels = []

        for label_line in labels:
            parts = label_line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Валидация координат
                if (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                        0 < width <= 1 and 0 < height <= 1):
                    bboxes.append([center_x, center_y, width, height])
                    class_labels.append(class_id)

        if not bboxes:
            print(f"⚠️ Нет валидных bbox для {image_path}")
            return []

        # Создать пайплайн аугментации
        transform = create_albumentations_pipeline()

        augmented_data = []
        successful_augmentations = 0
        max_attempts = num_augmentations * 3  # Больше попыток

        for attempt in range(max_attempts):
            if successful_augmentations >= num_augmentations:
                break

            try:
                # Применить трансформацию
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                # Проверить результат
                if (transformed['bboxes'] and
                        len(transformed['bboxes']) > 0 and
                        len(transformed['bboxes']) == len(transformed['class_labels'])):

                    # Дополнительная валидация bbox
                    valid_bboxes = []
                    valid_labels = []

                    for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                        center_x, center_y, width, height = bbox

                        if (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                                0 < width <= 1 and 0 < height <= 1):
                            valid_bboxes.append(bbox)
                            valid_labels.append(class_id)

                    # Сохранить только если есть валидные bbox
                    if valid_bboxes and len(valid_bboxes) >= len(bboxes) * 0.7:  # Сохранить хотя бы 70% bbox
                        yolo_labels = []
                        for bbox, class_id in zip(valid_bboxes, valid_labels):
                            center_x, center_y, width, height = bbox
                            yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

                        augmented_data.append({
                            'image': transformed['image'],
                            'labels': yolo_labels,
                            'suffix': f'_albu_{successful_augmentations}',
                            'transform_info': f"attempt_{attempt}"
                        })

                        successful_augmentations += 1

            except Exception as e:
                # Логируем только критические ошибки
                if "bbox" not in str(e).lower():
                    print(f"⚠️ Ошибка трансформации (попытка {attempt}): {e}")
                continue

        if successful_augmentations == 0:
            print(f"⚠️ Не удалось создать аугментации для {image_path}")

        return augmented_data

    except Exception as e:
        print(f"❌ Критическая ошибка аугментации для {image_path}: {e}")
        return []


def setup_session():
    """Настроить сессию для Label Studio"""
    session = requests.Session()
    session.cookies.set('sessionid', SESSION_ID)
    session.headers.update({
        'Content-Type': 'application/json',
        'User-Agent': 'YOLO-Albumentations-Train/1.0'
    })
    return session


def get_project_tasks(session):
    """Получить все задачи проекта"""
    # Получить проекты
    response = session.get(f"{LABEL_STUDIO_URL}/api/projects/")
    if response.status_code != 200:
        raise Exception(f"Ошибка получения проектов: {response.status_code}")

    projects_data = response.json()
    if isinstance(projects_data, dict) and 'results' in projects_data:
        projects = projects_data['results']
    else:
        projects = projects_data

    # Найти нужный проект
    project_id = None
    for project in projects:
        if project.get('title') == PROJECT_NAME:
            project_id = project.get('id')
            break

    if not project_id:
        available = [p.get('title') for p in projects]
        raise Exception(f"Проект '{PROJECT_NAME}' не найден. Доступные: {available}")

    print(f"✅ Найден проект: {PROJECT_NAME} (ID: {project_id})")

    # Получить все задачи с пагинацией
    all_tasks = []
    page = 1

    while True:
        response = session.get(f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks/?page={page}")
        if response.status_code != 200:
            break

        data = response.json()

        if isinstance(data, dict) and 'results' in data:
            tasks = data['results']
            all_tasks.extend(tasks)
            if not data.get('next'):
                break
        else:
            all_tasks.extend(data)
            break

        page += 1

    return all_tasks


def download_image(session, image_url, save_path):
    """Скачать изображение с повторными попытками"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if image_url.startswith('/data/'):
                full_url = f"{LABEL_STUDIO_URL}{image_url}"
            elif image_url.startswith('http'):
                full_url = image_url
            else:
                full_url = f"{LABEL_STUDIO_URL}/data/{image_url}"

            response = session.get(full_url, timeout=30)
            response.raise_for_status()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)

            return True

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Попытка {attempt + 1} неудачна для {image_url}: {e}")
                continue
            else:
                print(f"❌ Не удалось скачать {image_url} после {max_retries} попыток: {e}")
                return False

    return False


def validate_annotations(annotations):
    """Проверить валидность аннотаций"""
    valid_labels = []

    for annotation in annotations:
        for result in annotation.get('result', []):
            if result.get('type') == 'rectanglelabels':
                value = result.get('value', {})

                if all(key in value for key in ['x', 'y', 'width', 'height']):
                    x = value.get('x', 0) / 100
                    y = value.get('y', 0) / 100
                    w = value.get('width', 0) / 100
                    h = value.get('height', 0) / 100

                    if (0 <= x <= 1 and 0 <= y <= 1 and
                            0 < w <= 1 and 0 < h <= 1 and
                            x + w <= 1 and y + h <= 1):

                        labels = value.get('rectanglelabels', [])
                        if labels:
                            valid_labels.append({
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'class': labels[0]
                            })

    return valid_labels


def process_task(session, task, output_dir, classes_dict, class_names, stats):
    """Обработать одну задачу"""
    stats['total_tasks'] += 1

    image_data = task.get('data', {})
    annotations = task.get('annotations', [])

    if not annotations:
        stats['no_annotations'] += 1
        return None

    # Найти URL изображения
    image_url = None
    for key, value in image_data.items():
        if isinstance(value, str) and any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
            image_url = value
            break

    if not image_url:
        stats['no_image_url'] += 1
        return None

    # Валидация аннотаций
    valid_labels = validate_annotations(annotations)
    if not valid_labels:
        stats['invalid_annotations'] += 1
        return None

    # Имя файла
    image_name = Path(image_url).name
    if not image_name or '.' not in image_name:
        image_name = f"image_{task.get('id', 'unknown')}.jpg"

    # Временный путь для скачивания
    temp_image_path = output_dir / 'temp' / image_name

    # Скачать изображение
    if not download_image(session, image_url, temp_image_path):
        stats['download_failed'] += 1
        return None

    # Проверить изображение
    try:
        with Image.open(temp_image_path) as img:
            img_width, img_height = img.size

        if img_width < 32 or img_height < 32:
            stats['too_small'] += 1
            return None

    except Exception as e:
        print(f"⚠️ Поврежденное изображение {image_name}: {e}")
        stats['corrupted_images'] += 1
        return None

    # Конвертировать аннотации в YOLO формат
    yolo_labels = []
    for label_data in valid_labels:
        class_name = label_data['class']

        if class_name not in classes_dict:
            classes_dict[class_name] = len(classes_dict)
            class_names.append(class_name)

        class_id = classes_dict[class_name]

        center_x = label_data['x'] + label_data['w'] / 2
        center_y = label_data['y'] + label_data['h'] / 2

        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {label_data['w']:.6f} {label_data['h']:.6f}")

    stats['successful_tasks'] += 1
    stats['total_labels'] += len(yolo_labels)

    return {
        'image_path': temp_image_path,
        'image_name': image_name,
        'labels': yolo_labels
    }


def create_augmented_dataset(processed_tasks, output_dir, class_names):
    """Создать аугментированный датасет с Albumentations"""
    augmented_tasks = []

    # Добавляем оригинальные изображения
    augmented_tasks.extend(processed_tasks)

    if not CONFIG['augmentation']['enabled']:
        print("⚠️ Аугментация отключена")
        return augmented_tasks

    print(f"🎨 Применение Albumentations аугментации (x{CONFIG['augmentation']['multiplier']})...")

    total_original = len(processed_tasks)
    total_augmented = 0

    for task in tqdm(processed_tasks, desc="Albumentations аугментация"):
        # Создать аугментированные версии
        augmented_data = apply_albumentations_augmentation(
            task['image_path'],
            task['labels'],
            class_names,
            CONFIG['augmentation']['multiplier']
        )

        # Сохранить аугментированные изображения
        for aug_data in augmented_data:
            base_name = Path(task['image_name']).stem
            ext = Path(task['image_name']).suffix
            aug_name = f"{base_name}{aug_data['suffix']}{ext}"

            aug_image_path = output_dir / 'temp' / aug_name

            try:
                # Albumentations возвращает numpy array
                aug_image_bgr = cv2.cvtColor(aug_data['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_image_path), aug_image_bgr)

                augmented_tasks.append({
                    'image_path': aug_image_path,
                    'image_name': aug_name,
                    'labels': aug_data['labels']
                })

                total_augmented += 1

            except Exception as e:
                print(f"⚠️ Ошибка сохранения аугментированного изображения: {e}")
                continue

    print(f"✅ Создано {len(augmented_tasks)} изображений:")
    print(f"   Оригинальных: {total_original}")
    print(f"   Аугментированных: {total_augmented}")
    print(f"   Коэффициент увеличения: {len(augmented_tasks) / total_original:.1f}x")

    return augmented_tasks


def create_dataset_split(processed_tasks, output_dir):
    """Создать разделение на train/val"""
    # Создать структуру папок
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            (output_dir / split / folder).mkdir(parents=True, exist_ok=True)

    if len(processed_tasks) == 0:
        raise Exception("Нет обработанных задач")

    # Перемешать и разделить
    random.shuffle(processed_tasks)
    total = len(processed_tasks)
    train_count = max(1, int(total * 0.8))

    train_tasks = processed_tasks[:train_count]
    val_tasks = processed_tasks[train_count:] if total > 1 else [processed_tasks[0]]

    print(f"📊 Разделение датасета: {len(train_tasks)} train, {len(val_tasks)} val")

    # Переместить файлы train
    for task in train_tasks:
        src_img = task['image_path']
        dst_img = output_dir / 'train' / 'images' / task['image_name']
        shutil.move(str(src_img), str(dst_img))

        label_name = Path(task['image_name']).stem + '.txt'
        dst_label = output_dir / 'train' / 'labels' / label_name
        with open(dst_label, 'w') as f:
            f.write('\n'.join(task['labels']))

    # Переместить файлы val
    for task in val_tasks:
        if task in train_tasks:
            # Дубликат - копируем
            src_img = output_dir / 'train' / 'images' / task['image_name']
            dst_img = output_dir / 'val' / 'images' / task['image_name']
            shutil.copy(str(src_img), str(dst_img))

            label_name = Path(task['image_name']).stem + '.txt'
            src_label = output_dir / 'train' / 'labels' / label_name
            dst_label = output_dir / 'val' / 'labels' / label_name
            shutil.copy(str(src_label), str(dst_label))
        else:
            src_img = task['image_path']
            dst_img = output_dir / 'val' / 'images' / task['image_name']
            shutil.move(str(src_img), str(dst_img))

            label_name = Path(task['image_name']).stem + '.txt'
            dst_label = output_dir / 'val' / 'labels' / label_name
            with open(dst_label, 'w') as f:
                f.write('\n'.join(task['labels']))

    # Удалить временную папку
    temp_dir = output_dir / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return len(train_tasks), len(val_tasks)


def create_yaml_config(output_dir, class_names):
    """Создать YAML конфигурацию"""
    config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return yaml_path


def print_statistics(stats):
    """Вывести подробную статистику"""
    print("\n📊 СТАТИСТИКА ОБРАБОТКИ:")
    print("-" * 40)
    print(f"Всего задач в проекте: {stats['total_tasks']}")
    print(f"✅ Успешно обработано: {stats['successful_tasks']}")
    print(f"📋 Всего меток: {stats['total_labels']}")
    print("\n❌ ПРОПУЩЕНО:")
    print(f"   Без аннотаций: {stats['no_annotations']}")
    print(f"   Без URL изображения: {stats['no_image_url']}")
    print(f"   Невалидные аннотации: {stats['invalid_annotations']}")
    print(f"   Ошибки загрузки: {stats['download_failed']}")
    print(f"   Поврежденные изображения: {stats['corrupted_images']}")
    print(f"   Слишком маленькие: {stats['too_small']}")

    success_rate = (stats['successful_tasks'] / stats['total_tasks']) * 100 if stats['total_tasks'] > 0 else 0
    print(f"\n🎯 Успешность: {success_rate:.1f}%")


def create_augmentation_report(output_dir, original_tasks, augmented_tasks, class_names):
    """Создать подробный отчет об аугментации Albumentations"""
    try:
        report_path = output_dir / 'albumentations_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ОБ АУГМЕНТАЦИИ ALBUMENTATIONS\n")
            f.write("=" * 50 + "\n\n")

            import datetime
            f.write(f"Дата создания: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("ПАРАМЕТРЫ АУГМЕНТАЦИИ:\n")
            f.write(f"- Библиотека: Albumentations\n")
            f.write(f"- Множитель увеличения: {CONFIG['augmentation']['multiplier']}\n")
            f.write(f"- Минимальная видимость bbox: {CONFIG['augmentation']['min_visibility']}\n")
            f.write(f"- Минимальная площадь bbox: {CONFIG['augmentation']['min_area']}\n\n")

            f.write("ГЕОМЕТРИЧЕСКИЕ ТРАНСФОРМАЦИИ:\n")
            geo = CONFIG['augmentation']['geometric']
            f.write(f"- Горизонтальное отражение: {geo['horizontal_flip']}\n")
            f.write(f"- Вертикальное отражение: {geo['vertical_flip']}\n")
            f.write(f"- Поворот (лимит): ±{geo['rotate_limit']}°\n")
            f.write(f"- Вероятность поворота: {geo['rotate_prob']}\n")
            f.write(f"- Сдвиг: {geo['shift_limit']}\n")
            f.write(f"- Масштаб: {geo['scale_limit']}\n\n")

            f.write("ЦВЕТОВЫЕ ТРАНСФОРМАЦИИ:\n")
            color = CONFIG['augmentation']['color']
            f.write(f"- Яркость/Контраст: ±{color['brightness_limit']}/±{color['contrast_limit']}\n")
            f.write(
                f"- HSV сдвиги: H±{color['hue_shift_limit']}, S±{color['sat_shift_limit']}, V±{color['val_shift_limit']}\n")
            f.write(f"- Вероятность HSV: {color['hsv_prob']}\n\n")

            f.write("РАЗМЫТИЕ И ШУМ:\n")
            blur = CONFIG['augmentation']['blur_noise']
            f.write(f"- Размытие (лимит): {blur['blur_limit']}\n")
            f.write(f"- Вероятность размытия: {blur['blur_prob']}\n")
            f.write(f"- Гауссов шум: {blur['gauss_noise_var']}\n")
            f.write(f"- Вероятность шума: {blur['noise_prob']}\n\n")

            f.write("ПОГОДНЫЕ ЭФФЕКТЫ:\n")
            weather = CONFIG['augmentation']['weather']
            f.write(f"- Дождь: {weather['rain_prob']}\n")
            f.write(f"- Туман: {weather['fog_prob']}\n")
            f.write(f"- Солнечные блики: {weather['sun_flare_prob']}\n\n")

            f.write("СТАТИСТИКА ДАННЫХ:\n")
            f.write(f"- Оригинальных изображений: {len(original_tasks)}\n")
            f.write(f"- Аугментированных изображений: {len(augmented_tasks) - len(original_tasks)}\n")
            f.write(f"- Итого изображений: {len(augmented_tasks)}\n")
            multiplier = len(augmented_tasks) / len(original_tasks) if len(original_tasks) > 0 else 1
            f.write(f"- Фактическое увеличение: {multiplier:.1f}x\n\n")

            f.write("КЛАССЫ:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"- {i}: {class_name}\n")

            f.write(f"\nВсего классов: {len(class_names)}\n\n")

            f.write("ПРИМЕНЕННЫЕ УЛУЧШЕНИЯ:\n")
            f.write("- Расширенный набор трансформаций Albumentations\n")
            f.write("- Строгая валидация bbox после каждой трансформации\n")
            f.write("- Погодные эффекты (дождь, туман, солнечные блики)\n")
            f.write("- Множественные попытки для гарантии успеха\n")
            f.write("- Отключение конфликтующих параметров YOLO\n")
            f.write("- Детальная статистика и отчетность\n")

        print(f"📄 Отчет об аугментации сохранен: {report_path}")

    except Exception as e:
        print(f"⚠️ Не удалось создать отчет: {e}")


def main():
    """Основная функция"""
    print("🚀 YOLO ОБУЧЕНИЕ С ALBUMENTATIONS АУГМЕНТАЦИЕЙ")
    print("=" * 55)

    if CONFIG['max_tasks']:
        print(f"⚠️ Ограничение: максимум {CONFIG['max_tasks']} задач")
    else:
        print("♾️ Без ограничений - все задачи будут обработаны")

    print(f"⚙️ Настройки обучения: {CONFIG['epochs']} эпох, batch={CONFIG['batch_size']}")
    print(f"🏗️ Модель: {CONFIG['model']}, размер: {CONFIG['image_size']}px")
    print(f"🎨 Аугментация: Albumentations (x{CONFIG['augmentation']['multiplier']})")
    print(f"   📐 Геометрические преобразования: ✅")
    print(f"   🎨 Цветовые эффекты: ✅")
    print(f"   🌫️ Размытие и шум: ✅")
    print(f"   🌦️ Погодные эффекты: ✅")
    print("-" * 55)

    output_dir = Path(OUTPUT_DIR)

    # Статистика
    stats = {
        'total_tasks': 0,
        'successful_tasks': 0,
        'total_labels': 0,
        'no_annotations': 0,
        'no_image_url': 0,
        'invalid_annotations': 0,
        'download_failed': 0,
        'corrupted_images': 0,
        'too_small': 0
    }

    try:
        # Очистить и создать выходную директорию
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # Настроить сессию
        print("🔌 Подключение к Label Studio...")
        session = setup_session()

        # Получить все задачи
        print("📥 Загрузка всех задач...")
        all_tasks = get_project_tasks(session)
        print(f"   Найдено {len(all_tasks)} задач в проекте")

        # Применить ограничение если есть
        if CONFIG['max_tasks'] and len(all_tasks) > CONFIG['max_tasks']:
            all_tasks = all_tasks[:CONFIG['max_tasks']]
            print(f"   Ограничено до {len(all_tasks)} задач")

        # Обработать все задачи
        print("🔄 Обработка всех задач...")
        classes_dict = {}
        class_names = []
        processed_tasks = []

        for task in tqdm(all_tasks, desc="Обработка задач"):
            result = process_task(session, task, output_dir, classes_dict, class_names, stats)
            if result:
                processed_tasks.append(result)

        # Вывести статистику обработки
        print_statistics(stats)

        if len(processed_tasks) == 0:
            print("❌ Нет данных для обучения!")
            return

        print(f"\n🏷️ Найдено классов: {len(class_names)}")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")

        # Применить аугментацию Albumentations
        augmented_tasks = create_augmented_dataset(processed_tasks, output_dir, class_names)

        # Создать разделение датасета
        print("\n📁 Создание структуры датасета...")
        train_count, val_count = create_dataset_split(augmented_tasks, output_dir)

        # Создать YAML конфигурацию
        print("📄 Создание конфигурации...")
        yaml_path = create_yaml_config(output_dir, class_names)

        # Финальная проверка
        train_images = list((output_dir / 'train' / 'images').glob('*'))
        val_images = list((output_dir / 'val' / 'images').glob('*'))

        print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        print(f"   Train: {len(train_images)} изображений")
        print(f"   Val: {len(val_images)} изображений")
        print(f"   Классов: {len(class_names)}")
        original_count = len(processed_tasks)
        augmented_count = len(augmented_tasks) - original_count
        print(f"   Оригинальных: {original_count}")
        print(f"   Аугментированных: {augmented_count}")
        print(f"   Коэффициент увеличения: {len(augmented_tasks) / original_count:.1f}x")

        if len(train_images) == 0:
            print("❌ Нет изображений для обучения!")
            return

        # Запуск обучения
        print(f"\n🚀 ЗАПУСК ОБУЧЕНИЯ YOLO...")
        print(f"⏰ Ожидаемое время увеличено из-за аугментированных данных")

        model = YOLO(CONFIG['model'])

        # Оптимизированные настройки для работы с Albumentations
        train_args = {
            'data': str(yaml_path),
            'epochs': CONFIG['epochs'],
            'imgsz': CONFIG['image_size'],
            'batch': CONFIG['batch_size'],
            'lr0': CONFIG['learning_rate'],
            'device': CONFIG['device'],
            'project': str(output_dir),
            'name': 'albumentations_training',
            'patience': CONFIG['patience'],
            'cache': False,
            'workers': 2 if CONFIG['device'] == 'cpu' else 4,
            'verbose': True,
            'save_period': 10,
            'plots': True,

            # Отключаем встроенную аугментацию YOLO, так как используем Albumentations
            'optimizer': 'AdamW',  # Современный оптимизатор
            'close_mosaic': 0,  # Отключаем мозаику в конце
            'copy_paste': 0.0,  # Отключаем copy-paste
            'mixup': 0.0,  # Отключаем mixup
            'mosaic': 0.0,  # Отключаем мозаику

            # Отключаем все геометрические трансформации YOLO
            'degrees': 0.0,  # Поворот
            'translate': 0.0,  # Сдвиг
            'scale': 0.0,  # Масштаб
            'shear': 0.0,  # Наклон
            'perspective': 0.0,  # Перспектива
            'flipud': 0.0,  # Вертикальное отражение
            'fliplr': 0.0,  # Горизонтальное отражение

            # Отключаем цветовые трансформации YOLO
            'hsv_h': 0.0,  # Тон
            'hsv_s': 0.0,  # Насыщенность
            'hsv_v': 0.0,  # Яркость

            # Дополнительные параметры для стабильности
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'weight_decay': 0.0005,
            'box': 7.5,  # Вес loss для bbox
            'cls': 0.5,  # Вес loss для классификации
            'dfl': 1.5,  # Вес loss для распределения
        }

        print("📋 Ключевые параметры обучения:")
        key_params = ['epochs', 'batch', 'lr0', 'optimizer', 'mosaic', 'mixup']
        for key in key_params:
            if key in train_args:
                print(f"   {key}: {train_args[key]}")

        print(f"\n🎯 Встроенная аугментация YOLO: ОТКЛЮЧЕНА")
        print(f"🎨 Albumentations аугментация: АКТИВНА")

        # Запуск обучения
        results = model.train(**train_args)

        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📁 Результаты: {output_dir}/albumentations_training/")
        print(f"📊 Лучшая модель: {output_dir}/albumentations_training/weights/best.pt")

        # Команды для использования модели
        best_model = output_dir / 'albumentations_training' / 'weights' / 'best.pt'
        if best_model.exists():
            print(f"\n🔮 КОМАНДЫ ДЛЯ ИСПОЛЬЗОВАНИЯ МОДЕЛИ:")
            print(f"   # Предсказание")
            print(f"   yolo predict model='{best_model}' source='image.jpg' show=True")
            print(f"   # Валидация")
            print(f"   yolo val model='{best_model}' data='{yaml_path}'")
            print(f"   # Экспорт в ONNX")
            print(f"   yolo export model='{best_model}' format=onnx")
            print(f"   # Экспорт в TensorRT")
            print(f"   yolo export model='{best_model}' format=engine")

        # Создать детальный отчет
        create_augmentation_report(output_dir, processed_tasks, augmented_tasks, class_names)

        print(f"\n✨ Аугментация Albumentations успешно применена!")
        print(f"📈 Данные увеличены в {len(augmented_tasks) / len(processed_tasks):.1f} раза")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

        # Показать статистику даже при ошибке
        if stats['total_tasks'] > 0:
            print_statistics(stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Training с Albumentations аугментацией')
    parser.add_argument('--no-aug', action='store_true', help='Отключить аугментацию')
    parser.add_argument('--epochs', type=int, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, help='Размер батча')
    parser.add_argument('--multiplier', type=int, help='Множитель аугментации')
    parser.add_argument('--max-tasks', type=int, help='Максимальное количество задач')

    args = parser.parse_args()

    if args.no_aug:
        CONFIG['augmentation']['enabled'] = False
        print("🔧 Аугментация принудительно отключена")

    if args.epochs:
        CONFIG['epochs'] = args.epochs
        print(f"🔧 Установлено эпох: {args.epochs}")

    if args.batch_size:
        CONFIG['batch_size'] = args.batch_size
        print(f"🔧 Установлен размер батча: {args.batch_size}")

    if args.multiplier:
        CONFIG['augmentation']['multiplier'] = args.multiplier
        print(f"🔧 Установлен множитель аугментации: {args.multiplier}")

    if args.max_tasks:
        CONFIG['max_tasks'] = args.max_tasks
        print(f"🔧 Ограничение задач: {args.max_tasks}")

    main()