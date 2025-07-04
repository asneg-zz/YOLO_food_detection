#!/usr/bin/env python3
"""
Модуль для аугментации данных с использованием Albumentations
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("❌ Установите albumentations: pip install albumentations opencv-python")
    raise


def create_albumentations_pipeline(aug_config):
    """Создать настраиваемый пайплайн аугментации Albumentations"""

    transforms = []

    # Геометрические трансформации
    ENABLE_GEOMETRIC = True
    if ENABLE_GEOMETRIC:
        if aug_config['geometric']['horizontal_flip'] > 0:
            transforms.append(A.HorizontalFlip(p=aug_config['geometric']['horizontal_flip']))

        if aug_config['geometric']['vertical_flip'] > 0:
            transforms.append(A.VerticalFlip(p=aug_config['geometric']['vertical_flip']))

        if aug_config['geometric']['rotate_prob'] > 0 and aug_config['geometric']['rotate_limit'] > 0:
            transforms.append(
                A.Rotate(
                    limit=aug_config['geometric']['rotate_limit'],
                    p=aug_config['geometric']['rotate_prob'],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            )

        if (aug_config['geometric']['shift_limit'] > 0 or aug_config['geometric']['scale_limit'] > 0):
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=aug_config['geometric']['shift_limit'],
                    scale_limit=aug_config['geometric']['scale_limit'],
                    rotate_limit=0,
                    p=0.4,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            )

        ENABLE_ADVANCED_GEOMETRIC = False
        if ENABLE_ADVANCED_GEOMETRIC:
            transforms.append(A.Perspective(scale=(0.02, 0.05), p=0.2))
            transforms.append(
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.2,
                    border_mode=cv2.BORDER_CONSTANT, value=0
                )
            )

    # Цветовые трансформации
    ENABLE_COLOR = True
    if ENABLE_COLOR:
        if aug_config['color']['brightness_contrast_prob'] > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=aug_config['color']['brightness_limit'],
                    contrast_limit=aug_config['color']['contrast_limit'],
                    p=aug_config['color']['brightness_contrast_prob']
                )
            )

        if aug_config['color']['hsv_prob'] > 0:
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=aug_config['color']['hue_shift_limit'],
                    sat_shift_limit=aug_config['color']['sat_shift_limit'],
                    val_shift_limit=aug_config['color']['val_shift_limit'],
                    p=aug_config['color']['hsv_prob']
                )
            )

        ENABLE_ADVANCED_COLOR = True
        if ENABLE_ADVANCED_COLOR:
            transforms.extend([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.ToGray(p=0.05),
                A.ChannelShuffle(p=0.1),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
            ])

    # Размытие и шум
    ENABLE_BLUR_NOISE = True
    if ENABLE_BLUR_NOISE and aug_config['blur_noise']['blur_prob'] > 0:
        if aug_config['blur_noise']['blur_limit'] > 0:
            transforms.append(
                A.OneOf([
                    A.Blur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
                    A.GaussianBlur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
                    A.MotionBlur(blur_limit=aug_config['blur_noise']['blur_limit'], p=1.0),
                ], p=aug_config['blur_noise']['blur_prob'])
            )

        if aug_config['blur_noise']['noise_prob'] > 0:
            transforms.append(
                A.OneOf([
                    A.GaussNoise(var_limit=aug_config['blur_noise']['gauss_noise_var'], p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=aug_config['blur_noise']['noise_prob'])
            )

    # Погодные эффекты
    ENABLE_WEATHER = True
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
            transforms.append(A.OneOf(weather_transforms, p=0.1))

    # Дополнительные эффекты
    ENABLE_ADDITIONAL = False
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

    if not transforms:
        transforms.append(A.HorizontalFlip(p=0.5))

    composition = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=aug_config['min_visibility'],
            min_area=aug_config['min_area'],
        )
    )

    return composition


def apply_albumentations_augmentation(image_path, labels, class_names, num_augmentations, aug_config):
    """Применить аугментацию Albumentations к изображению"""

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

                if (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                        0 < width <= 1 and 0 < height <= 1):
                    bboxes.append([center_x, center_y, width, height])
                    class_labels.append(class_id)

        if not bboxes:
            return []

        transform = create_albumentations_pipeline(aug_config)

        augmented_data = []
        successful_augmentations = 0
        max_attempts = num_augmentations * 3

        for attempt in range(max_attempts):
            if successful_augmentations >= num_augmentations:
                break

            try:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                if (transformed['bboxes'] and
                        len(transformed['bboxes']) > 0 and
                        len(transformed['bboxes']) == len(transformed['class_labels'])):

                    valid_bboxes = []
                    valid_labels = []

                    for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                        center_x, center_y, width, height = bbox

                        if (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                                0 < width <= 1 and 0 < height <= 1):
                            valid_bboxes.append(bbox)
                            valid_labels.append(class_id)

                    if valid_bboxes and len(valid_bboxes) >= len(bboxes) * 0.7:
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

            except Exception:
                continue

        return augmented_data

    except Exception as e:
        print(f"❌ Ошибка аугментации для {image_path}: {e}")
        return []


def create_augmented_dataset(processed_tasks, output_dir, class_names, config):
    """Создать аугментированный датасет с Albumentations"""

    augmented_tasks = []
    augmented_tasks.extend(processed_tasks)

    if not config['augmentation']['enabled']:
        return augmented_tasks

    print(f"🎨 Применение Albumentations аугментации (x{config['augmentation']['multiplier']})...")

    total_original = len(processed_tasks)
    total_augmented = 0

    for task in tqdm(processed_tasks, desc="Albumentations аугментация"):
        augmented_data = apply_albumentations_augmentation(
            task['image_path'],
            task['labels'],
            class_names,
            config['augmentation']['multiplier'],
            config['augmentation']
        )

        for aug_data in augmented_data:
            base_name = Path(task['image_name']).stem
            ext = Path(task['image_name']).suffix
            aug_name = f"{base_name}{aug_data['suffix']}{ext}"

            aug_image_path = output_dir / 'temp' / aug_name

            try:
                aug_image_bgr = cv2.cvtColor(aug_data['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_image_path), aug_image_bgr)

                augmented_tasks.append({
                    'image_path': aug_image_path,
                    'image_name': aug_name,
                    'labels': aug_data['labels']
                })

                total_augmented += 1

            except Exception:
                continue

    print(f"✅ Создано {len(augmented_tasks)} изображений")
    print(f"   Оригинальных: {total_original}")
    print(f"   Аугментированных: {total_augmented}")

    return augmented_tasks