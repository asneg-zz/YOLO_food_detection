#!/usr/bin/env python3
"""
YOLO обучение с аугментацией данных на основе Albumentations
Упрощенная версия
"""

import os
import sys
import requests
import yaml
import shutil
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image

# Импорт модуля аугментации
from YOLO_train.albumentations_augmentor import create_augmented_dataset

# ===========================================
# НАСТРОЙКИ
# ===========================================

SESSION_ID = ".eJxVj8luhDAQRP_FZ0De2sscc883oLbdBmcQHmGQsij_HojmMseuelWt-mFHSezGguHCGsQeEne9Fg57L6PugyWpwfqsAFjH6jbhWr5xL3UdH3d2Ex1bsO3jUqeynqcFAcp4rwajtHIWOjbisc_j0Wgb_18J9qIFjHdaLyN94DrVIdZ130oYLmR4um14r4mWtyf7UjBjm8901i7KCDZl5zxmKYQlUqQUqKy5wRBIR4kpCzBeEQkNQZ5A4hwgkL9KG7V2LaPPR9m-2E0ryTn__QOhEltj:1uXhLS:2WLC-RPM9gtSTciGzJj1fsNAguI4A3z0qU2RHPiDVNM"
LABEL_STUDIO_URL = "http://localhost:8080"
PROJECT_NAME = "New Project #3"
OUTPUT_DIR = "../yolo_dataset_albumentations_v11n_au_full"

# Настройки для обучения
CONFIG = {
    'epochs': 150,
    'batch_size': 10,
    'image_size': 320,
    'learning_rate': 0.005,
    'device': 'cpu',
    'model': 'yolo11n.pt',
    'patience': 5,
    'max_tasks': None,

    # НАСТРОЙКИ АУГМЕНТАЦИИ ALBUMENTATIONS
    'augmentation': {
        'enabled': True,
        'multiplier': 30,
        'min_visibility': 0.8,
        'min_area': 0.05,

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
    response = session.get(f"{LABEL_STUDIO_URL}/api/projects/")
    if response.status_code != 200:
        raise Exception(f"Ошибка получения проектов: {response.status_code}")

    projects_data = response.json()
    if isinstance(projects_data, dict) and 'results' in projects_data:
        projects = projects_data['results']
    else:
        projects = projects_data

    project_id = None
    for project in projects:
        if project.get('title') == PROJECT_NAME:
            project_id = project.get('id')
            break

    if not project_id:
        available = [p.get('title') for p in projects]
        raise Exception(f"Проект '{PROJECT_NAME}' не найден. Доступные: {available}")

    print(f"✅ Найден проект: {PROJECT_NAME} (ID: {project_id})")

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
    """Скачать изображение"""
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
        print(f"❌ Не удалось скачать {image_url}: {e}")
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


def process_task(session, task, output_dir, classes_dict, class_names):
    """Обработать одну задачу"""
    image_data = task.get('data', {})
    annotations = task.get('annotations', [])

    if not annotations:
        return None

    image_url = None
    for key, value in image_data.items():
        if isinstance(value, str) and any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
            image_url = value
            break

    if not image_url:
        return None

    valid_labels = validate_annotations(annotations)
    if not valid_labels:
        return None

    image_name = Path(image_url).name
    if not image_name or '.' not in image_name:
        image_name = f"image_{task.get('id', 'unknown')}.jpg"

    temp_image_path = output_dir / 'temp' / image_name

    if not download_image(session, image_url, temp_image_path):
        return None

    try:
        with Image.open(temp_image_path) as img:
            img_width, img_height = img.size

        if img_width < 32 or img_height < 32:
            return None

    except Exception:
        return None

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

    return {
        'image_path': temp_image_path,
        'image_name': image_name,
        'labels': yolo_labels
    }


def create_dataset_split(processed_tasks, output_dir):
    """Создать разделение на train/val/test"""
    # Создаем папки для всех разделов включая test
    for split in ['train', 'val', 'test']:
        for folder in ['images', 'labels']:
            (output_dir / split / folder).mkdir(parents=True, exist_ok=True)

    if len(processed_tasks) == 0:
        raise Exception("Нет обработанных задач")

    random.shuffle(processed_tasks)
    total = len(processed_tasks)

    # Разделяем на train (60%), val (20%), test (20%)
    train_count = max(1, int(total * 0.6))
    val_count = max(1, int(total * 0.2))
    test_count = total - train_count - val_count

    if test_count <= 0:
        test_count = 1
        val_count = total - train_count - test_count

    train_tasks = processed_tasks[:train_count]
    val_tasks = processed_tasks[train_count:train_count + val_count]
    test_tasks = processed_tasks[train_count + val_count:]

    # Если test_tasks пустой, берем один элемент из val_tasks
    if not test_tasks and val_tasks:
        test_tasks = [val_tasks.pop()]

    print(f"📊 Разделение: {len(train_tasks)} train, {len(val_tasks)} val, {len(test_tasks)} test")

    # Обработка train задач
    for task in train_tasks:
        src_img = task['image_path']
        dst_img = output_dir / 'train' / 'images' / task['image_name']
        shutil.move(str(src_img), str(dst_img))

        label_name = Path(task['image_name']).stem + '.txt'
        dst_label = output_dir / 'train' / 'labels' / label_name
        with open(dst_label, 'w') as f:
            f.write('\n'.join(task['labels']))

    # Обработка val задач
    for task in val_tasks:
        if task in train_tasks:
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

    # Обработка test задач
    for task in test_tasks:
        if task in train_tasks:
            src_img = output_dir / 'train' / 'images' / task['image_name']
            dst_img = output_dir / 'test' / 'images' / task['image_name']
            shutil.copy(str(src_img), str(dst_img))

            label_name = Path(task['image_name']).stem + '.txt'
            src_label = output_dir / 'train' / 'labels' / label_name
            dst_label = output_dir / 'test' / 'labels' / label_name
            shutil.copy(str(src_label), str(dst_label))
        elif task in val_tasks:
            src_img = output_dir / 'val' / 'images' / task['image_name']
            dst_img = output_dir / 'test' / 'images' / task['image_name']
            shutil.copy(str(src_img), str(dst_img))

            label_name = Path(task['image_name']).stem + '.txt'
            src_label = output_dir / 'val' / 'labels' / label_name
            dst_label = output_dir / 'test' / 'labels' / label_name
            shutil.copy(str(src_label), str(dst_label))
        else:
            src_img = task['image_path']
            dst_img = output_dir / 'test' / 'images' / task['image_name']
            shutil.move(str(src_img), str(dst_img))

            label_name = Path(task['image_name']).stem + '.txt'
            dst_label = output_dir / 'test' / 'labels' / label_name
            with open(dst_label, 'w') as f:
                f.write('\n'.join(task['labels']))

    temp_dir = output_dir / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return len(train_tasks), len(val_tasks), len(test_tasks)


def create_yaml_config(output_dir, class_names):
    """Создать YAML конфигурацию с поддержкой test данных"""
    config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',  # Добавляем путь к test данным
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return yaml_path


def main():
    """Основная функция"""
    print("🚀 YOLO ОБУЧЕНИЕ С ALBUMENTATIONS АУГМЕНТАЦИЕЙ")
    print(f"🎨 Аугментация: x{CONFIG['augmentation']['multiplier']}")

    output_dir = Path(OUTPUT_DIR)

    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        print("🔌 Подключение к Label Studio...")
        session = setup_session()

        print("📥 Загрузка задач...")
        all_tasks = get_project_tasks(session)
        print(f"   Найдено {len(all_tasks)} задач")

        if CONFIG['max_tasks'] and len(all_tasks) > CONFIG['max_tasks']:
            all_tasks = all_tasks[:CONFIG['max_tasks']]
            print(f"   Ограничено до {len(all_tasks)} задач")

        print("🔄 Обработка задач...")
        classes_dict = {}
        class_names = []
        processed_tasks = []

        for task in tqdm(all_tasks, desc="Обработка задач"):
            result = process_task(session, task, output_dir, classes_dict, class_names)
            if result:
                processed_tasks.append(result)

        if len(processed_tasks) == 0:
            print("❌ Нет данных для обучения!")
            return

        print(f"🏷️ Найдено классов: {len(class_names)}")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")

        # Применить аугментацию Albumentations
        augmented_tasks = create_augmented_dataset(processed_tasks, output_dir, class_names, CONFIG)

        print("📁 Создание структуры датасета...")
        train_count, val_count, test_count = create_dataset_split(augmented_tasks, output_dir)

        print("📄 Создание конфигурации...")
        yaml_path = create_yaml_config(output_dir, class_names)

        train_images = list((output_dir / 'train' / 'images').glob('*'))
        val_images = list((output_dir / 'val' / 'images').glob('*'))
        test_images = list((output_dir / 'test' / 'images').glob('*'))

        print(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        print(f"   Train: {len(train_images)} изображений")
        print(f"   Val: {len(val_images)} изображений")
        print(f"   Test: {len(test_images)} изображений")
        print(f"   Классов: {len(class_names)}")

        if len(train_images) == 0:
            print("❌ Нет изображений для обучения!")
            return

        print(f"🚀 ЗАПУСК ОБУЧЕНИЯ YOLO...")

        model = YOLO(CONFIG['model'])

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

            # Отключаем встроенную аугментацию YOLO
            'optimizer': 'AdamW',
            'close_mosaic': 0,
            'copy_paste': 0.0,
            'mixup': 0.0,
            'mosaic': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'weight_decay': 0.0005,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
        }

        print("🎯 Встроенная аугментация YOLO: ОТКЛЮЧЕНА")
        print("🎨 Albumentations аугментация: АКТИВНА")

        results = model.train(**train_args)

        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📁 Результаты: {output_dir}/albumentations_training/")
        print(f"📊 Лучшая модель: {output_dir}/albumentations_training/weights/best.pt")

        best_model = output_dir / 'albumentations_training' / 'weights' / 'best.pt'
        if best_model.exists():
            print(f"🔮 КОМАНДЫ ДЛЯ ИСПОЛЬЗОВАНИЯ МОДЕЛИ:")
            print(f"   yolo predict model='{best_model}' source='image.jpg' show=True")
            print(f"   yolo val model='{best_model}' data='{yaml_path}'")
            print(f"   yolo export model='{best_model}' format=onnx")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


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