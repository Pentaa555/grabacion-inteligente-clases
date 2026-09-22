# Pasos a seguir — Sistema de grabación inteligente de clases

> Documento de estado y hoja de ruta. Última actualización: 2026-09-21.
> Basado en el anteproyecto (Epa-2.pdf) y en lo validado en las sesiones de trabajo.

---

## Estado actual (lo que YA funciona)

| Módulo | Estado | Notas |
|---|---|---|
| **P1 · Captura + sincronización** | ✅ hecho | Video/audio en procesos FFmpeg separados y realineados. En Windows la captura CLI (v4l2/ALSA) no corre; la grabación por navegador (getUserMedia) sí. |
| **P2 · Reducción de ruido** | ✅ hecho | Cadena FFmpeg. Filtro **suavizado** (`afftdn=nf=-20:nr=6` + `dynaudnorm`) tras detectar que el original (`nf=-25`) aplanaba la voz y hacía alucinar a Whisper. |
| **Transcripción (Whisper)** | ✅ hecho | Motor local con GPU (RTX 3070). Procesa por ventanas para evitar el MemoryError de la STFT global de faster-whisper. Validado en clases reales de 1 h y 2 h. |
| **Comparación de modelos** | ✅ hecho | medium vs large-v3 con métricas y **tiempo de proceso** (registrado en el JSON). |
| **Diarización** | ⚠️ arreglada, falta validar en audio real | Bug corregido: la reconciliación de hablantes entre ventanas ya no colapsa todo en `SPEAKER_00`. Pendiente probarlo con una grabación con intervenciones. |
| **Subtítulos** | ✅ hecho | SRT, VTT (`<v Hablante>`) y JSON. |
| **Plataforma web (prototipo)** | ✅ hecho | FastAPI de una página: subir/grabar, barras de progreso reales, reproductor con subtítulos, descargas. |

### Entorno del equipo (Windows)
- Python 3.14.5 (`py`), GPU NVIDIA RTX 3070 (8 GB VRAM), CUDA 12.6.
- Instalados: faster-whisper, pyannote.audio, torch 2.11.0+cu126, FFmpeg 9.0.2, token HF.
- Webapp: `py -m uvicorn src.webapp:app --host 127.0.0.1 --port 8000`.

---

## Pendientes técnicos inmediatos (calidad del core)

1. **Validar la diarización arreglada** — procesar una grabación con intervenciones
   claras (docente + estudiantes), con `min_speakers=2`, y confirmar que aparecen
   varios `SPEAKER_XX`.
   - *Limitación conocida:* la reconciliación por solape de ventanas solo reconecta a
     un hablante si habla durante la zona de solape (30 s). Un estudiante que
     interviene brevemente y calla en los solapes puede salir como dos hablantes.
     Solución robusta futura: reconciliar por **embeddings de voz**, no por tiempo.

2. **Liberar modelos de RAM/VRAM al terminar cada job** — hoy Whisper y pyannote
   quedan cargados entre procesos. En este equipo con RAM justa conviene liberarlos y
   llamar `torch.cuda.empty_cache()` al finalizar.

3. **Blindar contra alucinación de Whisper en silencio** — activar `vad_filter` o
   filtrar segmentos repetidos ("Gracias." en bucle), para tramos de silencio largos.

---

## Hoja de ruta según el anteproyecto (PDF)

| # | Paso | Estado | Por qué importa |
|---|------|--------|-----------------|
| **P4** | **Identificar al DOCENTE** por huella de voz | pendiente | El PDF lo pide explícitamente: etiquetar `SPEAKER_XX` → "Docente"/"Estudiante" con una muestra de voz al inicio + embeddings (ECAPA-TDNN / coseno). Continúa la diarización. |
| **P5** | **Medir WER y DER** contra metas | pendiente | Metas del PDF: **WER < 20%**, **DER < 15%**. Requiere transcribir a mano ~5-10 min de una clase como *ground truth*. Es el número que la tesis debe reportar. |
| **P6** | **Filtrado semántico (BETO)** | pendiente | Clasificar cada intervención como *pertinente al tema* vs *charla paralela*. **Único módulo con ML original y entrenable** — el aporte investigativo real de la tesis. Requiere corpus etiquetado. |
| **P7** | **Resumen automático** de la sesión | pendiente | Depende de P6. Calidad del modelo en español. |
| **P8** | **Plataforma web definitiva** (React + PostgreSQL) | pendiente | El PDF pide React + PostgreSQL; hoy hay prototipo FastAPI de una página. Alcance grande. |

### Orden propuesto
1. Validar diarización (pendiente inmediato #1).
2. **P4 — Identificar al docente** (cierra el módulo de diarización, requisito del PDF).
3. **P5 — Medir WER/DER** (valida y cuantifica todo el núcleo técnico).
4. **P6 — Filtrado semántico BETO** (el aporte investigativo).
5. **P7 / P8** — resumen y plataforma definitiva (fases finales).

**Ruta crítica:** validar diarización → P4 → P5 cierra y mide el core que el PDF pide
defender. P6–P8 son las fases posteriores del anteproyecto.

---

## Notas de seguridad / operación

- **Token de Hugging Face:** está configurado como variable de entorno de usuario
  (`HUGGINGFACE_TOKEN`), NO en el código. El token que se pegó en el chat durante las
  pruebas quedó expuesto y **debe revocarse/regenerarse** en huggingface.co →
  Settings → Access Tokens.
- **No versionar:** `webdata/` (audios y resultados pesados), `.env`, logs. Ya cubierto
  por `.gitignore`.
- **Métricas objetivo del PDF:** WER < 20%, DER < 15%, precisión de filtrado > 85%.
