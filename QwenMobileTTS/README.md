# Qwen3 TTS Android Demo

Android-приложение для локального запуска ONNX-экспорта модели Qwen3 TTS с выбором голоса, генерацией WAV-файла и воспроизведением результата на устройстве.

# 1. Описание проекта

Этот проект является демонстрацией того, что ONNX-экспорт Qwen3 TTS можно запустить локально на Android-устройстве, с собственным пайплайном:

- ввод текста;
- выбор голоса (спикера);
- прогон текста через tokenizer;
- сборка входов для модели;
- генерация кодов аудио;
- прогон кодов через vocoder;
- сохранение результата в WAV;
- воспроизведение результата в приложении.
---


# 2. Файлы модели и Orchestration layer

| File | Description | Size |
|------|-------------|------|
| `talker_prefill.onnx` + `.data` | Talker LM prefill (28 layers) | ~1.7 GB |
| `talker_decode.onnx` + `.data` | Talker LM single-step decode | ~1.7 GB |
| `code_predictor.onnx` | Code Predictor (5 layers, 15 groups) | ~420 MB |
| `vocoder.onnx` + `.data` | Vocoder decoder (24kHz output) | ~437 MB |
| `embeddings/` | Text/codec embeddings as .npy + config | ~1.4 GB |
| `tokenizer/` | BPE tokenizer (vocab.json, merges.txt) | ~4 MB |

В данном проект bundle состоит не из одной модели `text -> wav`, а из нескольких отдельных частей:
- `talker_prefill.onnx`
- `talker_decode.onnx`
- `code_predictor.onnx`
- `vocoder.onnx`
- `embeddings/`
- `tokenizer/`
- `config.json`
- `speaker_ids.json`

Поскольку ONNX Runtime умеет исполнять отдельный граф, но не умеет сам токенизировать текст, собирать промпт и текстовые эмбеддинги в скрытое пространство модели, собирать коды по `codebooks` на каждом timestep и прогонять каждый итоговый код через vocoder, то необходимо было имплементировать **полный orchestration layer**, который, в свою очередь, делал следующее:

- токенизация входного текста
- сборка prompt в необходимом формате
- чтение embedding-матрицы из `.npy`;
- проецирование текстовых эмбеддингов в скрытое пространство модели;
- построение `inputs_embeds`, `attention_mask`, `position_ids`;
- запуск цикла `prefill -> decode -> code predictor` с поддержкой сэмплинга
  по логитам (отслеживая EOS)
- сборка `codes` по 16 `codebooks` на timestep;
- финальный прогон итоговых кодов через vocoder;
- сохранение WAV.
---

# 3.  Описание модельного bundle

Ниже перечислены основные файлы model bundle и их описание.

## 3.1. `talker_prefill.onnx` + `.data`
Используется для начального прогона prompt-а модели talker. Является стартовой фазой инференса после подачи текста.

Задача:
- принять подготовленные `inputs_embeds`;
- вернуть стартовые:
  - `logits`
  - `hidden_states`
  - `present_keys`
  - `present_values`
---

## 3.2. `talker_decode.onnx` + `.data`
Это single-step decode модель, которая используется для пошагового decode после prefill.

Задача:
- принять `nextInputEmbeds`;
- принять `past_keys` / `past_values`;
- вернуть обновлённые:
  - `logits`
  - `hidden_states`
  - `present_keys`
  - `present_values`
---

## 3.3. `code_predictor.onnx`
Используется для предсказания кодов аудио по группам.

Задача:
- после выбора `group0` по talker logits;
- последовательно достроить остальные codebooks;
- сформировать полный набор из 16 кодов на один timestep.
---

## 3.4. `vocoder.onnx` + `.data`
Используется для преобразования уже сгенерированных кодов в waveform.

На вход получает:
- codes `[1, 16, T]`

На выходе возвращает:
- waveform

---

## 3.5. `embeddings/`
Используется в модельном пайплайне для сборки `inputs_embeds`
Содержит:
- text embeddings;
- codec embeddings;
- projection веса и bias;
---

## 3.6. `tokenizer/`
Содержит данные для BPE токенизатора
- `vocab.json`
- `merges.txt`
---

## 3.7. `config.json`
 Конфигурационный файл модели:
- hidden sizes;
- vocab sizes;
- ids специальных токенов;
- language ids;
- ids для talker/code predictor/tts.

---

## 3.8. `speaker_ids.json`
Содержит отображение имени голоса в `speakerId`.

Например:
- `ryan`
- `serena`
- `...`

Приложение использует этот файл для заполнения выпадающего списка голосов и последующей передачи `speakerId` в пайплайн.

---

# 4. Общая схема инференса
Общая схема инференса выглядит следующим образом:
```text
Input Text -> Tokenizer -> Embeddings
           -> Talker Prefill -> Talker Decode -> Code Predictor
           -> Codes -> Vocoder -> Waveform -> WAV
```

Ниже схема всего процесса генерации аудио в Android-приложении.

```text
MainActivity
  -> QwenTTSPipeline.run(...)
    -> normalizeInputText(...)
    -> splitIntoSentenceChunks(...)
    -> SpeakerCatalogLoader -> speakerId
    -> QwenEmbeddingRepository.loadRequiredCoreEmbeddings()

    для каждого chunk-а:
      -> QwenTextTokenizer.buildCustomVoicePromptIds(...)
      -> QwenGenerateCodesRunner.run(...)
          -> QwenPrefillBuilder.build(...)
          -> QwenTalkerPrefillRunner.run(...)
          -> QwenCodePredictorLoopRunner.runFromPrefill(...)
          -> QwenNextInputBuilder.buildFromCodes(...)
          -> QwenTalkerDecodeRunner.runOneStepAfterPrefill(...)
          -> цикл:
               -> QwenCodePredictorLoopRunner.runFromDecode(...)
               -> QwenNextInputBuilder.buildFromCodes(...)
               -> QwenTalkerDecodeRunner.runOneStepFromNextInput(...)
          -> result: codes [1,16,T]
      -> QwenVocoderRunner.run(...)
      -> chunk waveform
    -> concatWaveforms(...)
    -> normalizeWaveform(...)
    -> WavFileWriter.writeMono16BitPcm(...)
    -> output.wav
```
# 5. Структура проекта
```
com.example.qwen3_tts
├── MainActivity.kt						   # Главный экран приложения
├── audio
│   └── WavFileWriter.kt				   # Запись итоговой waveform в WAV-файл
├── config
│   └── QwenConfigLoader.kt				   # Загрузка и обработка config.json
├── data
│   ├── npy
│   │   └── NpyReader.kt				   # Reader для numpy массивов
│   └── repository
│       ├── QwenEmbeddingRepository.kt	   # Доступ к embedding-матрицам и projection-весам, получение embedding по token/code id
│       ├── QwenModelRepository.kt		   # Главная точка путей к model bundle
│       └── SpeakerCatalogLoader.kt		   # Загрузка speaker_ids.json и маппинг speakerName: speakerId
├── domain
│   ├── builders
│   │   ├── QwenNextInputBuilder.kt		   # Сборка nextInputEmbeds для следующего decode шага из сгенерированных codes
│   │   └── QwenPrefillBuilder.kt		   # Подготовка inputs_embeds и trailing_text_hidden для начального prefill
│   ├── pipeline
│   │   ├── QwenGenerateCodesRunner.kt     # Основной orchestrator текст -> аудио codes
│   │   └── QwenTTSPipeline.kt			   # Верхнеуровневый пайплайн 
│   └── sampling
│       └── QwenSampling.kt				   # Сэмплинг по логитам с выбором top-k/temperature/repetition penalty
├── inference
│   └── runners
│       ├── QwenCodePredictorLoopRunner.kt # Прогон code_predictor.onnx и генерация оставшихся 15 codebooks
│       ├── QwenTalkerDecodeRunner.kt.     # Пошаговый decode через talker_decode.onnx с past key/value cache
│       ├── QwenTalkerPrefillRunner.kt.    # Начальный через talker_prefill.onnx, получение logits и KV cache
│       └── QwenVocoderRunner.kt	       # Преобразование codes в waveform
├── storage
│   ├── BundleImporter.kt				   # Импорт zip bundle модели
│   └── QwenStoragePaths.kt				   # Построение путей хранения bundle model на устройстве
└── tokenizer
    ├── QwenBpeResources.kt				   # Загрузка BPE-ресурсов tokenizer-а
    ├── QwenByteEncoder.kt				   # Byte-level encoder для BPE tokenizer-а
    └── QwenTextTokenizer.kt      		   # Построение prompt token idx с помощью BPE 
```

# 6. Как приложение хранит модельные файлы

На текущем этапе model bundle **не хранится в `assets/`**, а лежит **во внешнем доступном хранилище / импортируемой директории**, после чего приложение работает с ним через файловые пути.

Это сделано потому, что:
-   ONNX-файлы крупные;
-   `.data` большие;
-   `embeddings/` очень тяжёлая;
-   помещать всё это в `assets` неудобно и нецелесообразно (приложение падает при запуске).

Текущая стратегия для хранения модели следующая:
-   пользователь подготавливает zip bundle отдельно;
-   в приложении нажимает `Import Bundle`;
-   приложение распаковывает bundle и дальше работает с ним через `QwenModelRepository`.
# 7. Как подготовить bundle модели
Необходимы следующие компоненты bundle:
-   `talker_prefill.onnx`
-   `talker_prefill.onnx.data`
-   `talker_decode.onnx`
-   `talker_decode.onnx.data`
-   `code_predictor.onnx`
-   `vocoder.onnx`
-   `vocoder.onnx.data`
-   `embeddings/`
-   `tokenizer/`
-   `config.json`
-   `speaker_ids.json`


Для запуска приложения требуются предварительные шаги: (TODO: bash скрипт сюда опишу)
1.  Скачать model bundle из источника, который используется в команде.
2.  Проверить, что внутри есть все файлы и папки из списка выше.
3.  Упаковать их в zip, если используете импорт через кнопку.
4.  На устройстве импортировать этот zip через `Import Bundle`.

# 8. Использование приложения 

`Import Bundle` - Импортирует model bundle.  
`Speaker Spinner` - Выбор спикера по имени из speaker_ids.json  
`Generate Speech` - Запуск инференса и получение итогового WAV  
`Play Output` - Воспроизведение последнего сохраненного (синтезированного) аудио  
`Stop Playback` - Остановка аудио  
<img width="255" height="428" alt="App Snapshot" src="https://github.com/user-attachments/assets/9b50c7d9-1415-44c4-b471-218e6325dbfb" />

# 9. TODO
- Квантованную модель положить в assets
- bash скрипт для весов с HG (elbruno C# runtime)
- Добавить гранулярность в Progress Bar (возможно)
- UI покрасивее сделать и поинтереснее
