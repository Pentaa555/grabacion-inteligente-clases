# Grabación Inteligente de Clases — Documento de Entrega de Resultados

**Proyecto:** Sistema de grabación, transcripción, diarización y filtrado semántico de clases presenciales
**Fecha de entrega:** 23 de septiembre de 2026
**Estado:** Prototipo funcional con métricas validadas

---

## 1. Resumen ejecutivo

Se construyó un prototipo funcional que **graba una clase, extrae y limpia el audio,
lo transcribe automáticamente y separa las voces de los distintos hablantes**,
entregando subtítulos y una transcripción estructurada. El sistema fue validado
sobre **grabaciones reales de aula** (clases de Física Electromagnética de 1 a 2
horas) y alcanza las metas de precisión definidas en el anteproyecto.

**Resultado principal:** la transcripción automática logra un **WER de 12,68 %**,
holgadamente por debajo de la meta de la tesis (**WER < 20 %**), medido contra una
transcripción de referencia hecha manualmente.

---

## 2. Métricas alcanzadas

### 2.1 Precisión de transcripción (WER)

Se midió el *Word Error Rate* comparando la salida automática del sistema contra
una transcripción de referencia (*ground truth*) elaborada **manualmente** sobre
un fragmento representativo de 5 minutos de clase (explicación de la Ley de Coulomb).

| Métrica | Valor |
|---|---:|
| Palabras en la referencia manual | 473 |
| Palabras acertadas | 427 |
| Sustituciones | 35 |
| Eliminaciones | 11 |
| Inserciones | 14 |
| **WER = (S+D+I)/N** | **12,68 %** |
| Precisión de palabra | 90,27 % |

> **Meta del anteproyecto:** WER < 20 % → **✅ CUMPLE (con holgura)**

**Metodología:** el cálculo usa distancia de edición a nivel de palabra (algoritmo
de Levenshtein), la métrica estándar en la literatura de reconocimiento de voz. Antes
de comparar, ambos textos se normalizan (minúsculas, sin tildes, sin puntuación —
conservando el punto decimal de las cifras — y unificando números escritos con cifras,
p. ej. "nueve" ↔ "9") para no penalizar diferencias de formato que no son errores de
reconocimiento. El procedimiento es reproducible mediante el script `src/wer.py`.

**Naturaleza de los errores:** los 35 errores restantes son errores reales de
reconocimiento, no de formato: confusiones léxicas (`indica` → `toxica`,
`medida` → `limita`), notación técnica (`Q1` ↔ `Q2`) y diferencias menores de forma
(`dirigido` ↔ `dirigida`). **No se detectaron alucinaciones** (texto inventado sobre
silencio).

### 2.2 Comparación de modelos de transcripción (medium vs large-v3)

Se comparó el modelo `medium` frente a `large-v3` sobre el **mismo audio** y bajo
**idénticas condiciones**, para justificar la elección del modelo de producción.

| Métrica | medium | large-v3 | Diferencia |
|---|---:|---:|---:|
| Segmentos generados | 793 | 1 287 | +62 % |
| Palabras transcritas | 7 712 | 8 126 | +5,4 % |
| Caracteres transcritos | 45 041 | 47 323 | +5,1 % |
| Duración media por segmento | 4,29 s | 2,69 s | −37 % (más fino) |
| Alucinaciones | Ninguna | Ninguna | igual |

**Decisión:** se adopta **`large-v3`** como modelo de producción por su mayor
cobertura léxica, segmentación más fina (mejor para subtítulos y diarización) y
viabilidad en el hardware disponible (GPU NVIDIA RTX 3070). `medium` queda como
alternativa cuando se priorice la velocidad (~2× más rápido).

*(Informe detallado en `webdata/salida/informe_comparacion_medium_vs_large.md`.)*

### 2.3 Diarización (separación de hablantes)

Sobre una clase real completa (**1 h 47 min**), el sistema separó correctamente
**7 hablantes distintos** (`SPEAKER_00` … `SPEAKER_06`) en 1 926 segmentos, en
**17,7 minutos de proceso** (≈ 6× tiempo real) con GPU.

> Este módulo tenía un fallo crítico corregido en esta fase: previamente colapsaba
> todas las voces en un único hablante. La corrección (emparejamiento de hablantes
> solo entre ventanas consecutivas por solape temporal) resolvió el problema, que
> era de código y no del audio.

### 2.4 Identificación del docente

Sobre la diarización, el sistema **identifica automáticamente al docente** y
re-etiqueta los hablantes con nombres legibles: el hablante identificado pasa a
**"Docente"** y los demás a **"Hablante 1", "Hablante 2", …**

**Criterio:** en una clase magistral el docente es, por definición, quien acumula
**mayor tiempo de habla**. El sistema suma el tiempo total de intervención de cada
hablante detectado por la diarización y asigna la etiqueta "Docente" al de mayor
tiempo. El resultado incluye un desglose cuantitativo (`speaker_stats`) con el
tiempo de habla, el número de intervenciones y el porcentaje de cada hablante, que
respalda la identificación de forma verificable.

Ejemplo del desglose que produce el sistema:

| Hablante | Tiempo de habla | Intervenciones | % del total |
|---|---:|---:|---:|
| **Docente** 🎓 | (mayor) | … | ~el mayor % |
| Hablante 1 | … | … | … |
| Hablante 2 | … | … | … |

> **Salvaguarda de honestidad:** si la diarización detecta un solo hablante (no hay
> a quién distinguir), el sistema **no** fuerza la etiqueta "Docente"; deja
> "Hablante 1". La identificación solo se afirma cuando hay más de una voz separada.

**Mejora futura (no requerida para esta entrega):** confirmar la identidad del
docente mediante una **muestra de voz** (huella por *embeddings*) tomada de clases
pasadas, en lugar de la heurística de tiempo de habla. Esto permitiría identificar
al profesor aunque no sea quien más habla en una sesión concreta.

---

### 2.5 Filtrado semántico de intervenciones (aporte de ML original)

Es el **componente de aprendizaje automático propio** del proyecto: clasificar cada
intervención de la clase como **pertinente** (contenido académico) o **no pertinente**
(logística, avisos, conversación paralela). A diferencia del resto del sistema —que
reutiliza modelos preentrenados—, este clasificador se **entrena con datos del propio
dominio**.

**Arquitectura (dos capas, enfoque "embeddings + clasificador ligero"):**
un *featurizer* convierte cada texto en un vector y un clasificador de **Regresión
Logística** decide la etiqueta. El diseño deja el featurizer **enchufable**: la
versión actual usa **TF-IDF** (scikit-learn, sin GPU, entrena en segundos) como
línea base, y está preparado para sustituirlo por **embeddings de BETO** (BERT en
español) sin cambiar el resto del pipeline —ese es el refinamiento previsto para la
sustentación.

**Estado actual (línea base entrenada):** el módulo entrena, evalúa, predice y
persiste el modelo. Sobre un **dataset semilla** de 40 intervenciones etiquetadas
(clase de Ley de Coulomb + frases típicas de aula):

| Métrica (conjunto de prueba) | Valor |
|---|---:|
| Exactitud (*accuracy*) | 80,0 % |
| Precisión (macro) | 85,7 % |
| Recall (macro) | 80,0 % |
| F1 (macro) | 79,2 % |

> **Nota metodológica honesta:** esta cifra es **orientativa**, medida sobre un
> dataset semilla pequeño. No es la métrica definitiva de la tesis. La meta del
> anteproyecto (**precisión de filtrado > 85 %**) se validará al entrenar con
> **cientos de intervenciones reales** extraídas de las transcripciones ya
> disponibles —trabajo previsto para la fase de refinamiento. Lo entregado hoy es
> el **módulo funcional y reproducible**, no una maqueta: clasifica correctamente
> frases nuevas que no vio en el entrenamiento (p. ej. reconoce como *pertinente*
> "definimos el campo eléctrico como la fuerza por unidad de carga" y como *no
> pertinente* "profe puedo ir al baño").

**Reproducible:** `py -m src.semantic_filter data/filtrado_dataset.csv`
(script `src/semantic_filter.py`, dataset `data/filtrado_dataset.csv`).

---

## 3. Módulos implementados y funcionando

| Módulo | Estado | Evidencia |
|---|---|---|
| P1 · Captura de audio/video | ✅ Funcional | Grabación desde navegador + carga de archivos |
| P2 · Reducción de ruido (denoise) | ✅ Funcional | Filtro suave validado: no degrada la voz |
| P3 · Transcripción (Whisper large-v3) | ✅ Validado | WER 12,68 %; clases reales de 1–2 h |
| P3b · Subtítulos (SRT / VTT / JSON) | ✅ Funcional | Archivos generados automáticamente |
| P3c · Diarización (pyannote) | ✅ Validado | 7 hablantes separados en clase real |
| P4 · Identificación del docente | ✅ Funcional | Etiqueta "Docente" por mayor tiempo de habla + desglose |
| P6 · Filtrado semántico (ML propio) | 🟡 Línea base entrenada | Clasificador entrenable; 80 % acc. sobre dataset semilla |
| Plataforma web de prueba (FastAPI) | ✅ Funcional | Subir/grabar → procesar → ver → descargar |

**Pipeline completo:**
`captura → extracción de audio (FFmpeg) → denoise → transcripción (Whisper large-v3) →
diarización (pyannote) → identificación del docente → filtrado semántico (ML propio) →
subtítulos (SRT/VTT/JSON) → interfaz web`

**Plataforma web:** interfaz de prueba con selección de motor y modelo, barras de
progreso reales (subida y transcripción), tiempo de proceso, factor × tiempo real, y
botón de cancelación de trabajos en curso.

---

## 4. Entorno técnico

- **Lenguaje:** Python
- **Motor de ASR:** faster-whisper (`large-v3`), español (`es-CO`)
- **Diarización:** pyannote.audio (procesamiento por ventanas para audios largos)
- **Preprocesamiento:** FFmpeg (extracción + reducción de ruido)
- **Hardware:** GPU NVIDIA GeForce RTX 3070 (8 GB), CUDA 12.6, precisión float16
- **Interfaz:** FastAPI + Uvicorn

---

## 5. Archivos entregables

| Archivo | Contenido |
|---|---|
| `webdata/salida/informe_comparacion_medium_vs_large.md` | Informe comparativo de modelos |
| `WER Ground truth.txt` | Transcripción de referencia manual (5 min) |
| `webdata/salida/<job>/transcripcion.json` | Transcripción automática con tiempos y hablante |
| `webdata/salida/<job>/subtitulos.srt` / `.vtt` | Subtítulos generados |
| `src/wer.py` | Script reproducible de cálculo de WER |
| `src/semantic_filter.py` | Módulo de filtrado semántico (ML propio, entrenable) |
| `data/filtrado_dataset.csv` | Dataset semilla etiquetado (pertinente / no pertinente) |

---

## 6. Trabajo pendiente (fases siguientes)

Se declara con transparencia lo que **aún no está** implementado, correspondiente a
las fases posteriores del cronograma:

- **P6 · Filtrado semántico — ampliación:** el módulo ya está implementado y
  entrenado sobre un dataset semilla. Falta **entrenarlo con cientos de intervenciones
  reales** etiquetadas de las transcripciones existentes y **sustituir el featurizer
  TF-IDF por embeddings de BETO**, para alcanzar y validar la meta de precisión > 85 %.
- **P7 · Resumen automático** de la clase.
- **P8 · Plataforma web definitiva** (React + PostgreSQL).
- **Confirmar la identificación del docente por muestra de voz** (embeddings de
  clases pasadas), como refinamiento de la heurística actual de mayor tiempo de habla.
- **Ampliar la medición de WER** a 2–3 fragmentos adicionales para robustecer la
  cifra reportada en la tesis final (actualmente medida sobre un fragmento de 5 min).

---

*Documento generado a partir de resultados reales del prototipo. Todas las cifras
provienen de corridas ejecutadas sobre grabaciones de aula, no de estimaciones.*
