# Plan de avance del prototipo — Sistema de grabación inteligente de clases

> Universidad Distrital · proyecto de grado. Documento vivo; refleja lo validado
> en sesión y la hoja de ruta técnica. Última actualización: 2026-09-15.

## Supuestos (basados en lo observado)

- **Idioma:** español de Colombia (`es-CO`), confirmado en todas las pruebas.
- **Formato de salida:** SRT + WebVTT + JSON, con etiqueta de hablante. VTT es el
  estándar para reproducir sobre `<video>` en la plataforma web del PDF.
- **Hardware disponible:** este equipo — 16 cores, **sin GPU NVIDIA**, RAM muy
  limitada (0.5–4 GB libres observados). Webcam USB `/dev/video0`, mic del sistema
  ALC256 (`arecord`).
- **Volumen real:** clases de ~90 min. El POC actual manda audio inline a GCP
  (límite ~1 min síncrono) — insuficiente para producción sin GCS.
- **Credenciales GCP:** ADC vía `gcloud` (proyecto `nlp-proyecto-508714`), sin clave
  JSON por la política `iam.disableServiceAccountKeyCreation`. (Nota: el archivo ADC
  debe regenerarse con `gcloud auth application-default login` si desaparece.)
- **Hallazgo clave:** el mic del sistema (`arecord`) da señal muy superior a la
  webcam para voz (9% vs 65% silencio). El modelo local `base` superó a GCP en
  precisión y segmentación con una sola voz.

---

## 1. Captura de webcam + audio y sincronización  — **IMPLEMENTADO (P1)**

**Problema observado:** al grabar webcam+mic juntos con un solo FFmpeg, el audio se
truncó a ~20s aunque el video duraba 2 min (desincronización v4l2+ALSA), con
warnings de DTS no monotónico.

**Solución implementada — capturas separadas y sincronizadas** (`src/capture.py`):
- **Video:** proceso FFmpeg `v4l2` desde `/dev/video0`, **solo video** (`-an`).
- **Audio:** proceso FFmpeg `alsa` independiente, **solo audio** (`-vn`), WAV mono
  16 kHz PCM (sin pérdida, fuente para el ASR).
- **Sincronía:** ambos procesos arrancan casi simultáneos; se mide el desfase real
  de arranque (`audio_offset`) y al muxear se aplica `-itsoffset` para realinear.
  Los subtítulos se anclan al **reloj del audio** (referencia temporal maestra).
- **Integridad:** la CLI compara duraciones de video/audio y avisa si difieren >1s.
- **API:** `record_synced()` devuelve `SyncedCapture` (video, audio, muxed, offset,
  duraciones). `record()` legacy se conserva (`--legacy-capture`).
- **Riesgo pendiente:** deriva de reloj en grabaciones largas. Mitigación futura:
  segmentar en bloques (~10 min) y validar duración por bloque.

Uso: `python -m src.cli record --duration 30` (sincronizado por defecto).

## 2. Reducción de ruido  — **IMPLEMENTADO (P2)**

**Solución implementada** (`src/denoise.py`): cadena de filtros FFmpeg, elegida
sobre DeepFilterNet para **no añadir dependencias pesadas** (crítico con RAM baja):

```
highpass=f=80         # quita retumbe/ruido de baja frecuencia
lowpass=f=8000        # recorta siseo agudo fuera de la banda de voz
afftdn=nf=-25         # denoise espectral (FFT), ruido estacionario
loudnorm=I=-16...     # normaliza volumen (EBU R128)
```

- Corre en CPU a ~27× tiempo real, no compite por memoria con Whisper/pyannote.
- **DeepFilterNet** queda como mejora futura seleccionable
  (`--denoise-method deepfilternet`), para cuando el hardware lo permita.

Uso: `python -m src.cli process <audio> --engine <e> --denoise`.

## 3. Integración diarización + transcripción

**Estado:** faster-whisper produce segmentos con timestamps; pyannote produce turnos
por hablante; se cruzan por máximo solapamiento temporal (`_assign_speakers`).

**Mejoras necesarias:**
- **Identificar al DOCENTE** (objetivo del PDF): muestra de voz al inicio + comparar
  embeddings (ECAPA-TDNN / coseno) para etiquetar `SPEAKER_XX` → "Docente"/"Estudiante".
- **Ajustar `min_speakers`** según contexto (1 para voz única, 2+ para aula).
- **Riesgo:** en aula con ruido la DER sube — el PDF fija meta <15%.

## 4. Generación y formato de subtítulos

**Estado:** SRT, VTT (`<v Hablante>`) y JSON generados y verificados.

**Pendiente para la plataforma:**
- Word-level timestamps para resaltado karaoke (faster-whisper con `word_timestamps=True`).
- Empalme sobre el video: VTT como `<track kind="subtitles">`, anclado al audio maestro.
- **Riesgo:** desfase si video/audio derivan (mitigado por la sección 1).

---

## 5. Próximos pasos por prioridad

| # | Paso | Estado | Depende de | Riesgo |
|---|------|--------|-----------|--------|
| **P1** | Separar captura video/audio y sincronizar | ✅ **hecho** | — | Deriva de reloj en clases largas |
| **P2** | Reducción de ruido (FFmpeg; DeepFilterNet futuro) | ✅ **hecho** | P1 | — (FFmpeg ligero) |
| **P3** | Adaptar GCP para audio largo vía Cloud Storage (`long_running` + URI) | pendiente | ADC OK | Costo, setup bucket, cuota |
| **P4** | Identificación del docente por embedding de voz | pendiente | P1 | Precisión del matching en ruido |
| **P5** | Grabar clase real y medir WER/DER vs metas (<20% / <15%) | pendiente | P1, P2 | Puede exigir `large` → GPU |
| **P6** | Filtrado semántico BETO (pertinente vs charla paralela) | pendiente | P5 | Requiere corpus etiquetado |
| **P7** | Resumen automático de la sesión | pendiente | P6 | Calidad del modelo en español |
| **P8** | Plataforma web FastAPI + React | pendiente | P1–P7 | Alcance grande; hosting |

**Ruta crítica:** P1 → P2 → P5 valida el núcleo técnico del PDF. P3 solo si se adopta
GCP en producción. P6–P8 son fases posteriores del anteproyecto.

**Bloqueo actual:** la RAM crítica del equipo impide `large-v3` y pruebas de campo
serias. Antes de P5 conviene resolver hardware (el PDF presupuesta Colab Pro o
servidor con GPU ≥8 GB).

## Estrategia de motor (local vs GCP)

| Escenario | Motor | Motivo |
|-----------|-------|--------|
| Desarrollo / clases piloto en este equipo | Local `base`/`small` | Gratis, offline, buena segmentación; cabe en RAM baja |
| Clase completa alta calidad con GPU/servidor | Local `large-v3` | Máxima precisión, alineado al PDF, sin costo por uso |
| Sin GPU y se necesita rapidez / respaldo | GCP | Descarga el cómputo a la nube |

Regla operativa: **procesar en diferido** (grabar primero, transcribir como job
encolado); nunca cargar Whisper + pyannote a la vez que otra tarea pesada.
