#!/usr/bin/env python3
"""
Улучшенный скрипт обучения YOLO на основе анализа результатов
Автоматически сгенерирован анализатором качества
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

# Улучшенная конфигурация
IMPROVED_CONFIG = {'epochs': 1000, 'batch_size': 16, 'image_size': 640, 'learning_rate': 0.001, 'device': 'cpu', 'model': 'yolo11s.pt', 'patience': 50, 'augmentation': {'enabled': True, 'multiplier': 12, 'min_visibility': 0.7, 'min_area': 0.03, 'geometric': {'horizontal_flip': 0.5, 'vertical_flip': 0.1, 'rotate_limit': 20, 'rotate_prob': 0.6, 'shift_limit': 0.1, 'scale_limit': 0.2, 'shear_prob': 0.2}, 'color': {'brightness_limit': 0.3, 'contrast_limit': 0.3, 'brightness_contrast_prob': 0.6, 'hue_shift_limit': 15, 'sat_shift_limit': 30, 'val_shift_limit': 15, 'hsv_prob': 0.5}, 'blur_noise': {'blur_limit': 5, 'blur_prob': 0.3, 'gauss_noise_var': (10, 50), 'noise_prob': 0.3}, 'weather': {'rain_prob': 0.15, 'fog_prob': 0.15, 'sun_flare_prob': 0.1}}}

# Настройки Label Studio (обновите под свои данные)
SESSION_ID = ".eJxVj81uxCAMhN-FcxLJBB9xyOXeZIgMmoRvBKiRSu6u-e6Hay95sz_gbzZNc0ZOZGEeZkCPvDVDbS21CDyB5nZwHbjin1JGO5GOFFB9wxpyW-43MrCM7lHPZ8xpTXbVqGCXl0HhUd2SB69yWq-Cx_Scx8naz4G6YmuC_IK15cDmdR7RDswwvtQyf2eP-8fK-ATYoW_0OcnLcKe3DNBkInDGNKFAIJYKkI1iL0nHwganRCEQmleXV4ClVyqJp0IKltGL4fY_HD5mlqLXp7x9f41sv:1uWTX9:QTo4Fp90Idh9tTt6izl8sM6VZuQNYTCjes0asHuJOKk"
LABEL_STUDIO_URL = "http://localhost:8080"
PROJECT_NAME = "New Project #3"
OUTPUT_DIR = "./improved_yolo_dataset_v2"

# Проверка библиотек
try:
    from ultralytics import YOLO
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    print("✅ Все библиотеки загружены успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установите: pip install ultralytics albumentations opencv-python")
    sys.exit(1)

# [Здесь будет остальной код обучения с улучшенными параметрами]
# Функции setup_session, get_project_tasks, etc. остаются теми же

def create_improved_albumentations_pipeline():
    """Улучшенный пайплайн аугментации"""
    aug_config = IMPROVED_CONFIG['augmentation']
    transforms = []

    # Включаем все геометрические трансформации
    transforms.extend([
        A.HorizontalFlip(p=aug_config['geometric']['horizontal_flip']),
        A.VerticalFlip(p=aug_config['geometric']['vertical_flip']),
        A.Rotate(
            limit=aug_config['geometric']['rotate_limit'],
            p=aug_config['geometric']['rotate_prob'],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.ShiftScaleRotate(
            shift_limit=aug_config['geometric']['shift_limit'],
            scale_limit=aug_config['geometric']['scale_limit'],
            rotate_limit=0,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
    ])

    # Включаем цветовые трансформации
    transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=aug_config['color']['brightness_limit'],
            contrast_limit=aug_config['color']['contrast_limit'],
            p=aug_config['color']['brightness_contrast_prob']
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug_config['color']['hue_shift_limit'],
            sat_shift_limit=aug_config['color']['sat_shift_limit'],
            val_shift_limit=aug_config['color']['val_shift_limit'],
            p=aug_config['color']['hsv_prob']
        ),
    ])

    # Размытие и шум
    transforms.extend([
        A.OneOf([
            A.Blur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
            A.GaussianBlur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
            A.MotionBlur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
        ], p=aug_config['blur_noise']['blur_prob']),

        A.OneOf([
            A.GaussNoise(var_limit=aug_config['blur_noise']['gauss_noise_var'], p=1.0),
            A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=1.0),
        ], p=aug_config['blur_noise']['noise_prob'])
    ])

    # Погодные эффекты
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

    if weather_transforms:
        transforms.append(A.OneOf(weather_transforms, p=0.2))

    composition = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=aug_config['min_visibility'],
            min_area=aug_config['min_area'],
        )
    )

    print(f"✅ Создан улучшенный пайплайн с {len(transforms)} трансформациями")
    return composition

if __name__ == "__main__":
    print("🚀 ЗАПУСК УЛУЧШЕННОГО ОБУЧЕНИЯ YOLO")
    print("=" * 50)
    print("Основано на анализе предыдущих результатов")
    print(f"Новые параметры:")
    print(f"- Эпохи: {IMPROVED_CONFIG['epochs']}")
    print(f"- Размер изображения: {IMPROVED_CONFIG['image_size']}")
    print(f"- Множитель аугментации: {IMPROVED_CONFIG['augmentation']['multiplier']}")
    print(f"- Модель: {IMPROVED_CONFIG['model']}")

    # Здесь должен быть основной код обучения
    # main()
