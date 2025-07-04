#!/usr/bin/env python3
"""
Упрощенный маппинг кириллических классов для YOLO детектора
Конфигурация загружается из JSON файла
"""

import json
import os
from pathlib import Path

# Глобальные переменные для кэширования
CLASS_MAPPING = {}
CLASS_COLORS = {}
CLASS_CATEGORIES = {}
SYNONYMS = {}
COCO_MAPPING = {}

# Путь к конфигурационному файлу
CONFIG_PATH = Path(__file__).parent / "classes_config.json"


def load_config():
    """Загрузить конфигурацию из JSON файла"""
    global CLASS_MAPPING, CLASS_COLORS, CLASS_CATEGORIES, SYNONYMS, COCO_MAPPING

    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Загружаем маппинг классов (конвертируем ключи в int)
            CLASS_MAPPING = {int(k): v for k, v in config.get('class_mapping', {}).items()}
            CLASS_COLORS = config.get('class_colors', {})
            CLASS_CATEGORIES = config.get('categories', {})
            SYNONYMS = config.get('synonyms', {})

            # Загружаем COCO маппинг (конвертируем ключи в int)
            coco_data = config.get('coco_mapping', {})
            COCO_MAPPING = {int(k): v for k, v in coco_data.items()}

            if COCO_MAPPING:
                print(f"✅ COCO маппинг: {len(COCO_MAPPING)} классов")
            return True

        else:
            raise FileNotFoundError(f"Конфигурационный файл не найден: {CONFIG_PATH}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Создайте файл classes_config.json с конфигурацией классов")
        return False

    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        print("💡 Проверьте корректность JSON файла")
        return False


def get_cyrillic_class_name(class_id, fallback_name=None):
    """
    Получить кириллическое название класса

    Args:
        class_id (int): ID класса
        fallback_name (str): Резервное название

    Returns:
        str: Кириллическое название класса
    """
    if not CLASS_MAPPING:
        if not load_config():
            return fallback_name or f"класс_{class_id}"

    if class_id in CLASS_MAPPING:
        return CLASS_MAPPING[class_id]

    if fallback_name:
        return str(fallback_name)

    return f"класс_{class_id}"


def get_class_by_synonym(synonym):
    """
    Найти класс по синониму

    Args:
        synonym (str): Синоним класса

    Returns:
        str or None: Найденный класс или None
    """
    if not SYNONYMS:
        load_config()

    if synonym:
        synonym_lower = str(synonym).lower().strip()
        return SYNONYMS.get(synonym_lower, None)

    return None


def get_class_color(class_name):
    """
    Получить цвет класса

    Args:
        class_name (str): Название класса

    Returns:
        str: Hex цвет класса
    """
    if not CLASS_COLORS:
        load_config()

    return CLASS_COLORS.get(class_name, "#808080")


def get_class_color_bgr(class_name):
    """
    Получить цвет класса в BGR формате для OpenCV

    Args:
        class_name (str): Название класса

    Returns:
        tuple: BGR цвет для OpenCV
    """
    hex_color = get_class_color(class_name).lstrip('#')
    if len(hex_color) == 6:
        try:
            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            return (rgb[2], rgb[1], rgb[0])  # RGB -> BGR
        except ValueError:
            pass

    return (128, 128, 128)  # Серый по умолчанию


def is_food_class(class_name):
    """
    Проверить, является ли класс едой

    Args:
        class_name (str): Название класса

    Returns:
        bool: True если класс относится к еде
    """
    if not CLASS_CATEGORIES:
        load_config()

    food_categories = ["основные_блюда", "закуски_гарниры", "десерты", "напитки", "еда"]

    for category in food_categories:
        if class_name in CLASS_CATEGORIES.get(category, []):
            return True

    return False


def is_tableware_class(class_name):
    """
    Проверить, является ли класс посудой

    Args:
        class_name (str): Название класса

    Returns:
        bool: True если класс относится к посуде
    """
    if not CLASS_CATEGORIES:
        load_config()

    return class_name in CLASS_CATEGORIES.get("посуда_приборы", []) or \
        class_name in CLASS_CATEGORIES.get("посуда", [])


def get_classes_by_category(category):
    """
    Получить список классов по категории

    Args:
        category (str): Название категории

    Returns:
        list: Список классов в категории
    """
    if not CLASS_CATEGORIES:
        load_config()

    return CLASS_CATEGORIES.get(category, [])


def auto_detect_classes_from_model(model_path):
    """
    Автоопределение классов из модели (заглушка)

    Args:
        model_path (str): Путь к модели

    Returns:
        bool: True если определение успешно
    """
    print(f"🔍 Автоопределение классов из модели: {Path(model_path).name}")

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)

        if hasattr(model, 'names') and model.names:
            model_classes = model.names
            print(f"📋 Модель содержит {len(model_classes)} классов")

            # Показываем первые 5 классов
            for i, (class_id, class_name) in enumerate(list(model_classes.items())[:5]):
                cyrillic_name = get_class_by_synonym(class_name) or class_name
                print(f"   {class_id}: {class_name} -> {cyrillic_name}")

            if len(model_classes) > 5:
                print(f"   ... и еще {len(model_classes) - 5}")

        return True

    except Exception as e:
        print(f"⚠️ Ошибка автоопределения: {e}")
        return False


def get_coco_class_name(class_id):
    """
    Получить кириллическое название для COCO класса

    Args:
        class_id (int): ID COCO класса

    Returns:
        str or None: Кириллическое название или None
    """
    if not COCO_MAPPING:
        if not load_config():
            return None

    return COCO_MAPPING.get(class_id, None)


def reload_config():
    """Перезагрузить конфигурацию из файла"""
    global CLASS_MAPPING, CLASS_COLORS, CLASS_CATEGORIES, SYNONYMS, COCO_MAPPING

    CLASS_MAPPING.clear()
    CLASS_COLORS.clear()
    CLASS_CATEGORIES.clear()
    SYNONYMS.clear()
    COCO_MAPPING.clear()

    return load_config()


def save_config(new_mapping=None, new_colors=None):
    """
    Сохранить изменения в конфигурационный файл

    Args:
        new_mapping (dict): Новый маппинг классов
        new_colors (dict): Новые цвета классов

    Returns:
        bool: True если сохранение успешно
    """
    try:
        # Загружаем текущую конфигурацию
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {"metadata": {"version": "2.0"}}

        # Обновляем данные
        if new_mapping:
            config['class_mapping'] = {str(k): v for k, v in new_mapping.items()}
            config['metadata']['total_classes'] = len(new_mapping)

        if new_colors:
            config['class_colors'] = new_colors

        # Сохраняем
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"✅ Конфигурация сохранена: {CONFIG_PATH}")
        return True

    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return False


def print_config_info():
    """Вывести информацию о текущей конфигурации"""
    if not CLASS_MAPPING:
        if not load_config():
            print("❌ Конфигурация не загружена")
            return

    print(f"\n🍽️ КОНФИГУРАЦИЯ КЛАССОВ:")
    print(f"Файл: {CONFIG_PATH}")
    print(f"Всего классов: {len(CLASS_MAPPING)}")
    print("=" * 50)

    # Группируем по категориям
    if CLASS_CATEGORIES:
        for category, classes in CLASS_CATEGORIES.items():
            if classes:
                print(f"\n📂 {category.title()} ({len(classes)}):")
                for class_name in classes:
                    # Находим ID класса
                    class_id = None
                    for cid, cname in CLASS_MAPPING.items():
                        if cname == class_name:
                            class_id = cid
                            break

                    color = get_class_color(class_name)
                    if class_id is not None:
                        print(f"  {class_id:2d}: {class_name} ({color})")
    else:
        # Если категорий нет, показываем все классы
        print("\n📋 Все классы:")
        for class_id, class_name in sorted(CLASS_MAPPING.items()):
            color = get_class_color(class_name)
            print(f"  {class_id:2d}: {class_name} ({color})")


def print_food_mapping_info():
    """Вывести информацию о маппинге блюд (алиас для совместимости)"""
    print_config_info()


def validate_food_mapping():
    """
    Проверить корректность маппинга блюд

    Returns:
        dict: Результаты валидации
    """
    if not CLASS_MAPPING:
        if not load_config():
            return {
                'valid': False,
                'errors': ['Конфигурация не загружена'],
                'warnings': [],
                'stats': {'total_classes': 0, 'food_classes': 0, 'tableware_classes': 0, 'service_classes': 0}
            }

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'total_classes': len(CLASS_MAPPING),
            'food_classes': 0,
            'tableware_classes': 0,
            'service_classes': 0
        }
    }

    # Подсчитываем статистику по типам
    for class_name in CLASS_MAPPING.values():
        if is_food_class(class_name):
            results['stats']['food_classes'] += 1
        elif is_tableware_class(class_name):
            results['stats']['tableware_classes'] += 1
        elif class_name in ['background', 'negative']:
            results['stats']['service_classes'] += 1

    # Проверяем наличие служебных классов
    has_background = any(name in ['background', 'фон'] for name in CLASS_MAPPING.values())
    has_negative = any(name in ['negative', 'негатив'] for name in CLASS_MAPPING.values())

    if not has_background:
        results['warnings'].append("Отсутствует класс 'background'")

    if not has_negative:
        results['warnings'].append("Отсутствует класс 'negative'")

    # Проверяем цвета
    missing_colors = []
    for class_name in CLASS_MAPPING.values():
        if class_name not in CLASS_COLORS:
            missing_colors.append(class_name)

    if missing_colors:
        results['warnings'].append(f"Отсутствуют цвета для классов: {missing_colors}")

    return results


def load_classes_from_yaml(yaml_path):
    """
    Загрузить классы из YAML файла (для совместимости)

    Args:
        yaml_path (str): Путь к YAML файлу

    Returns:
        bool: True если загрузка успешна
    """
    try:
        import yaml
        from pathlib import Path

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'names' in data:
            names = data['names']
            new_mapping = {}

            if isinstance(names, list):
                for i, name in enumerate(names):
                    new_mapping[i] = name
            elif isinstance(names, dict):
                new_mapping = {int(k) if isinstance(k, str) and k.isdigit() else k: v
                               for k, v in names.items()}

            # Сохраняем в JSON
            if save_config(new_mapping=new_mapping):
                # Перезагружаем конфигурацию
                reload_config()
                print(f"✅ Загружено {len(new_mapping)} классов из {Path(yaml_path).name}")
                return True

        print(f"⚠️ Не найдены классы в YAML файле: {yaml_path}")
        return False

    except ImportError:
        print("❌ Для загрузки YAML установите: pip install PyYAML")
        return False
    except Exception as e:
        print(f"❌ Ошибка загрузки YAML: {e}")
        return False


def check_fonts():
    """Проверка шрифтов (заглушка для совместимости)"""
    print("🔤 Проверка шрифтов с поддержкой кириллицы...")

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    available_fonts = []
    for font_path in font_paths:
        if os.path.exists(font_path):
            available_fonts.append(font_path)

    if available_fonts:
        print(f"✅ Найдены шрифты: {len(available_fonts)}")
    else:
        print("⚠️ Системные шрифты с кириллицей не найдены")

    return available_fonts


# Автоматическая загрузка конфигурации при импорте
if not load_config():
    print("⚠️ ВНИМАНИЕ: Конфигурация классов не загружена!")
    print("📝 Создайте файл classes_config.json с настройками классов")
    print("📁 Пример файла можно найти в документации проекта")

if __name__ == "__main__":
    print("🔧 Маппинг классов YOLO v2.0 - JSON конфигурация")
    print("=" * 60)

    # Показать текущую конфигурацию
    print_config_info()

    # Статистика
    food_count = sum(1 for name in CLASS_MAPPING.values() if is_food_class(name))
    tableware_count = sum(1 for name in CLASS_MAPPING.values() if is_tableware_class(name))

    print(f"\n📊 СТАТИСТИКА:")
    print(f"🍽️ Еда: {food_count}")
    print(f"🍴 Посуда: {tableware_count}")
    print(f"📦 Другое: {len(CLASS_MAPPING) - food_count - tableware_count}")
    print(f"🎨 Цветов: {len(CLASS_COLORS)}")
    print(f"🔗 Синонимов: {len(SYNONYMS)}")

    print(f"\n💾 Конфигурационный файл: {CONFIG_PATH}")
    print(f"✅ Готово к использованию!")