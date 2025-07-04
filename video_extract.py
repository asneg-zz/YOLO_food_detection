import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np


def extract_frames_with_ssim(video_path, output_dir="frames", max_frames=10, ssim_threshold=0.95, frame_step=10):
    """
    Извлекает каждый N-й кадр из видеофайла, изменяет размер до 640x480,
    поворачивает на 90 градусов по часовой стрелке и сохраняет их в формате PNG
    только если SSIM с предыдущим сохраненным кадром меньше порогового значения

    Args:
        video_path (str): Путь к видеофайлу
        output_dir (str): Директория для сохранения кадров
        max_frames (int): Максимальное количество кадров для извлечения
        ssim_threshold (float): Пороговое значение SSIM (0.95 по умолчанию)
        frame_step (int): Шаг обработки кадров (обрабатывать каждый N-й кадр)
    """

    # Проверяем существование видеофайла
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл {video_path} не найден!")
        return

    # Получаем имя файла без расширения для использования в имени кадров
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Создаем директорию для сохранения кадров, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория: {output_dir}")

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {video_path}")
        return

    # Получаем информацию о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Информация о видео:")
    print(f"- Исходный файл: {video_filename}")
    print(f"- FPS: {fps}")
    print(f"- Общее количество кадров: {total_frames}")
    print(f"- Продолжительность: {duration:.2f} секунд")
    print(f"- Шаг обработки: каждый {frame_step}-й кадр")
    print(f"- Максимальное количество кадров для извлечения: {max_frames}")
    print(f"- Пороговое значение SSIM: {ssim_threshold}")
    print(f"- Размер сохраняемых кадров: 640x480")
    print(f"- Поворот: 90 градусов по часовой стрелке")
    print(f"- Ожидаемое количество обрабатываемых кадров: {total_frames // frame_step}")

    frame_count = 0
    processed_count = 0
    saved_count = 0
    previous_frame = None

    while saved_count < max_frames and frame_count < total_frames:
        # Читаем кадр
        ret, frame = cap.read()

        if not ret:
            print("Достигнут конец видео или ошибка чтения кадра")
            break

        # Обрабатываем только каждый N-й кадр
        if frame_count % frame_step == 0:
            processed_count += 1

            # Изменяем размер кадра до 640x480
            frame_resized = cv2.resize(frame, (640, 480))

            # Поворачиваем кадр на 90 градусов по часовой стрелке
            # cv2.ROTATE_90_CLOCKWISE поворачивает на 90 градусов по часовой стрелке
            frame_rotated = cv2.rotate(frame_resized, cv2.ROTATE_90_CLOCKWISE)

            # Конвертируем обработанный кадр в градации серого для вычисления SSIM
            current_frame_gray = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2GRAY)

            should_save = False

            if previous_frame is None:
                # Первый обрабатываемый кадр всегда сохраняем
                should_save = True
                print(f"Сохраняем первый обрабатываемый кадр (кадр #{frame_count + 1})")
            else:
                # Вычисляем SSIM между текущим и предыдущим сохраненным кадром
                ssim_value = ssim(previous_frame, current_frame_gray)
                print(f"Кадр #{frame_count + 1} (обработанный #{processed_count}): SSIM = {ssim_value:.4f}", end="")

                if ssim_value < ssim_threshold:
                    should_save = True
                    print(f" - СОХРАНЯЕМ (SSIM < {ssim_threshold})")
                else:
                    print(f" - пропускаем (SSIM >= {ssim_threshold})")

            if should_save:
                # Формируем имя файла с именем входного файла и номером исходного кадра
                filename = f"{video_filename}_frame_{frame_count + 1:06d}_640x480_rot90.png"
                filepath = os.path.join(output_dir, filename)

                # Сохраняем обработанный кадр в формате PNG
                success = cv2.imwrite(filepath, frame_rotated)

                if success:
                    print(f"Сохранен кадр: {filepath}")
                    saved_count += 1
                    # Обновляем предыдущий кадр для сравнения
                    previous_frame = current_frame_gray.copy()
                else:
                    print(f"Ошибка при сохранении кадра: {filepath}")

        frame_count += 1

    # Освобождаем ресурсы
    cap.release()

    print(f"\nЗавершено!")
    print(f"Общее количество кадров в видео: {total_frames}")
    print(f"Обработано кадров: {frame_count}")
    print(f"Проанализировано кадров (каждый {frame_step}-й): {processed_count}")
    print(f"Извлечено и сохранено кадров: {saved_count}")
    print(f"Размер сохраненных кадров: 640x480 (повернуто на 90° по часовой)")
    print(f"Кадры сохранены в директории: '{output_dir}'")


if __name__ == "__main__":
    # Путь к видеофайлу
    video_file = "/video/3_1.MOV"

    # Директория для сохранения кадров (будет создана в той же папке, что и скрипт)
    output_directory = "extracted_frames"

    # Количество кадров для извлечения
    frames_to_extract = 7000

    # Пороговое значение SSIM (0.80 означает, что кадры должны отличаться более чем на 20%)
    ssim_threshold = 0.85

    # Шаг обработки кадров (обрабатывать каждый 10-й кадр)
    frame_step = 10

    print("Начинаем извлечение каждого 10-го кадра из видео с изменением размера до 640x480,")
    print("поворотом на 90° по часовой стрелке и фильтрацией по SSIM...")
    extract_frames_with_ssim(video_file, output_directory, frames_to_extract, ssim_threshold, frame_step)