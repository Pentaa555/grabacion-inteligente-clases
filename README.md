# Prototipo — Grabación inteligente de clases (híbrido)

Prototipo del sistema descrito en el anteproyecto *"Sistema de grabación inteligente de
clases presenciales con filtrado automático de contenido mediante NLP"* (Universidad
Distrital Francisco José de Caldas).

Cubre los 3 requisitos del prototipo:

1. **Captura** de video (webcam) + audio (micrófono).
2. **Transcripción / subtítulos** del audio.
3. **Diarización de hablantes** (quién habla y cuándo).

## Arquitectura híbrida (backend intercambiable)

El pipeline está desacoplado del motor de ASR/diarización mediante una interfaz común
(`TranscriptionEngine`). Se elige en tiempo de ejecución con `--engine`:

| `--engine` | Implementación | Naturaleza |
|------------|----------------|------------|
| `gcp`      | Google Cloud Speech-to-Text v2 (diarización nativa) | Pago por uso (tu petición) |
| `local`   | WhisperX + pyannote.audio | Offline / gratuito (stack del PDF) |
| `mock`    | Datos sintéticos, sin dependencias | Para probar el pipeline sin credenciales ni GPU |

Así el prototipo cumple tu pedido (GCP) **y** conserva la ruta offline del anteproyecto.

```
webcam+mic ──► grabación (FFmpeg) ──► extracción audio (FLAC 16k mono)
                                          │
                                          ▼
                            TranscriptionEngine (gcp | local | mock)
                                          │
                          segmentos [ (t0, t1, hablante, texto) ]
                                          │
                       ┌──────────────────┼──────────────────┐
                       ▼                  ▼                  ▼
                   subtitulos.srt     subtitulos.vtt     transcripcion.json
```

## Supuestos tomados

- **Idioma:** español de Colombia (`es-CO`).
- **Formato de subtítulos:** SRT + WebVTT, con etiqueta de hablante por segmento.
- **Audio para ASR:** FLAC mono 16 kHz (recomendado por GCP para diarización).
- **Credenciales GCP:** se leen de `GOOGLE_APPLICATION_CREDENTIALS`. Sin ellas, el motor
  `gcp` no arranca y se sugiere `--engine mock`.
- **Lenguaje del prototipo:** Python (coincide con el backend FastAPI del PDF).

## Requisitos del sistema

- FFmpeg (captura + extracción de audio) — ya instalado.
- Python 3.11+.
- Webcam en `/dev/video0` y micrófono (ALSA `default`).

## Instalación

```bash
cd "/home/pentaa/Documentos/proyecto /prototipo"
python3 -m venv .venv && source .venv/bin/activate

# Motor mock: sin dependencias extra.
# Motor GCP:
pip install -r requirements-gcp.txt
# Motor local (requiere GPU para rendimiento razonable):
pip install -r requirements-local.txt
```

## Uso

```bash
# 1) Grabar 30 s de webcam + micrófono
python -m src.cli record --duration 30 --out grabacion.mkv

# 2) Procesar la grabación con el motor elegido
python -m src.cli process grabacion.mkv --engine mock   --lang es-CO
python -m src.cli process grabacion.mkv --engine gcp    --lang es-CO
python -m src.cli process grabacion.mkv --engine local  --lang es-CO

# 3) Todo en uno (grabar + procesar)
python -m src.cli run --duration 30 --engine mock
```

Salidas en `./salida/<nombre>/`: `subtitulos.srt`, `subtitulos.vtt`, `transcripcion.json`.

## Configurar GCP (cuando tengas cuenta)

1. Crea un proyecto en Google Cloud y habilita **Cloud Speech-to-Text API**.
2. Crea una *Service Account* con rol `Cloud Speech Client` y descarga la clave JSON.
3. Exporta la ruta:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/ruta/a/clave.json
   export GCP_PROJECT_ID=tu-proyecto
   ```
4. Ejecuta con `--engine gcp`.

## Alcance del prototipo

Incluye: captura, extracción de audio, transcripción, diarización, subtítulos SRT/VTT/JSON.
**No** incluye todavía (fases posteriores del PDF): reducción de ruido (DeepFilterNet),
filtrado semántico (BETO), resumen automático, ni la plataforma web (FastAPI + React).
