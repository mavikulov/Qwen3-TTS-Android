# TTS Benchmarks

В каталоге `benchmark/` теперь два независимых контура:

- latency/perf бенч: `bench_tts_hf.py` + `run_benchmark.sh`
- quality бенч: `bench_tts_quality.py` + Hydra-конфиги в `benchmark/configs`

## Установка

```bash
uv sync
```

## 1) Perf benchmark (как раньше)

```bash
bash benchmark/run_benchmark.sh
```

Пример override:

```bash
GPU_DEVICE=0 NUM_PROMPTS=20 MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice bash benchmark/run_benchmark.sh
```

## 2) Quality benchmark (DNSMOS + WER/CER)

### Что считается

- `DNSMOS` (локально, через `torchmetrics.audio`)
- `WER/CER` (ASR через Whisper, реализация `faster-whisper`)

### Датасеты

- RU: `Vikhrmodels/ToneBooksPlus`, split `validation`
- EN: `openslr/librispeech_asr`, split `validation.clean`

Для воспроизводимости формируются frozen-manifest файлы в `benchmark/data/manifests`.

### Подготовить/обновить манифесты

```bash
python benchmark/prepare_quality_manifests.py
```

Пересоздать манифесты:

```bash
python benchmark/prepare_quality_manifests.py dataset.recreate_manifests=true
```

### Запуск quality-прогона

```bash
python benchmark/bench_tts_quality.py
```

Или через скрипт:

```bash
bash benchmark/run_quality_benchmark.sh
```

Примеры override через Hydra:

```bash
# GPU для TTS + Whisper на CUDA
python benchmark/bench_tts_quality.py model.gpu_device="0" metrics.wer_cer.device=cuda

# Только WER/CER, без DNSMOS
python benchmark/bench_tts_quality.py metrics.dnsmos.enabled=false

# Отключить сохранение WAV и ускорить прогон
python benchmark/bench_tts_quality.py run.save_audio=false

# Включить батч-генерацию и потоковую оценку метрик
python benchmark/bench_tts_quality.py run.batch_size=4 run.score_workers=4

# Другое имя эксперимента
python benchmark/bench_tts_quality.py run.config_name=my_experiment
```

## Hydra-конфиги

- `benchmark/configs/quality.yaml` — базовый entrypoint
- `benchmark/configs/model/*.yaml` — модель/voice/lang map
- `benchmark/configs/dataset/*.yaml` — HF-источники и sampling
- `benchmark/configs/metrics/*.yaml` — параметры DNSMOS и Whisper
- `benchmark/configs/experiment/*.yaml` — преднастроенные эксперименты

## Артефакты quality run

Результаты складываются в `benchmark/tmp_results/quality/<run_id>/`:

- `summary.json` — агрегаты по `ru`, `en`, `overall`
- `per_sample.json` — построчные результаты
- `per_sample.csv` — CSV-экспорт
- `audio/` — синтезированные wav
- `resolved_config.yaml/json` — конфиг прогона

## Сравнение нескольких quality run

```bash
python benchmark/compare_quality_runs.py \
  --runs benchmark/tmp_results/quality/20260408_120000 benchmark/tmp_results/quality/20260408_153000
```
