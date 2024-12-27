# experiment-processing
# Sweep Efficiency Video Processing

## Описание
Скрипт обрабатывает видео микрофлюидной установки, рассчитывает коэффициент вытеснения и накладывает метрики на кадры видео. Выходные данные включают аннотированное видео, бинарное видео и график зависимости коэффициента вытеснения от порового объема и времени.

## Входные данные
- Видео: `LTG.mp4` (или другое в формате `.mp4`).
- Параметры обработки:
  - `blockSize`: Размер блока для адаптивной бинаризации.
  - `c`: Константа для порогового значения.
  - `Q`: Расход жидкости в мл/мин.
  - `volume_cubic_micrometres`: Объем микроструктуры в кубических микрометрах.

## Выходные данные
1. Аннотированное видео: `sweep_LTG.mp4`
2. Бинарное видео: `binary_sweep_LTG.mp4`
3. График зависимости коэффициента вытеснения от порового объема и времени.

## Запуск
1. Установите зависимости: `opencv-python`, `matplotlib`, `numpy`.
2. Запустите скрипт: `python script_name.py`.
3. Проверьте результаты: видеофайлы и график.
