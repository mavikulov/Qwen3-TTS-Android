# Квантизация Qwen3-TTS и запуск модели на мобильном устройстве.

Репозиторий содержит решение для эффективного запуска **Qwen3-TTS** на мобильных устройствах на Android. Основной фокус на исследовании **квантизации** для значительного уменьшения размера модели и ускорения инференса при сохранении качества генерации.

## Структура проекта

* QwenMobileTTS/ -   **Android-приложение**
* python/ -          **Скрипты для конвертации, квантизации и запуска модели**
* benchmark/ -       **Оценка качества и производительности**
* requirements.txt - **Python-зависимости**
* pyproject.toml -   **Настройки проекта (uv)**
* README.md -        **Этот файл**

## Быстрый старт

```bash
git clone https://github.com/mievst/Mobile-TTS-YDS.git
cd Mobile-TTS-YDS

# Установка зависимостей
pip install -r requirements.txt

# Или с использованием uv
uv sync
```

## Вся информация о мобильном приложении в [QwenModelTTS/README.md](https://github.com/mievst/Mobile-TTS-YDS/tree/main/QwenMobileTTS)
## Вся информация о бенчмарках в [benchmarks/README.md](https://github.com/mievst/Mobile-TTS-YDS/tree/main/benchmark)
