"""Motor local: faster-whisper + pyannote.audio (stack offline del PDF).

Requiere:
  - pip install faster-whisper pyannote.audio transformers huggingface_hub
  - Token de Hugging Face para pyannote: HUGGINGFACE_TOKEN
  - GPU NVIDIA recomendada (>= 8 GB VRAM); en CPU es lento pero funcional.

Flujo:
  1. faster-whisper transcribe (palabras + timestamps).
  2. pyannote.audio diariza (segmentos de hablante).
  3. Se asigna cada segmento de transcripción al hablante más cercano en el tiempo.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..types import Segment, TranscriptionResult
from .base import TranscriptionEngine


class LocalEngine(TranscriptionEngine):
    name = "local"

    def is_available(self) -> tuple[bool, str]:
        try:
            import faster_whisper
            import pyannote.audio  # noqa: F401
        except ImportError:
            return False, "Faltan dependencias: pip install faster-whisper pyannote.audio"
        if not os.environ.get("HUGGINGFACE_TOKEN"):
            return False, ("HUGGINGFACE_TOKEN no definido (requerido por pyannote "
                           "para descargar el modelo de diarización).")
        return True, "Motor local disponible."

    def transcribe(self, audio_path, progress_cb=None) -> TranscriptionResult:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(f"Motor local no disponible: {reason}")

        import faster_whisper
        from pyannote.audio import Pipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_size = self.options.get("model", "base")  # base/small/medium/large-v3
        lang = self.language.split("-")[0]  # "es-CO" -> "es"

        print(f"[*] faster-whisper: cargando modelo '{model_size}' ({device}) ...")
        model = faster_whisper.WhisperModel(
            model_size, device=device, compute_type="int8" if device == "cpu" else "float16"
        )

        print(f"[*] Transcribiendo audio ({lang}) por ventanas ...")
        # vad_filter descarta tramos que cree silencio; en audio real con pausas
        # recorta voz de más, así que por defecto va DESACTIVADO.
        use_vad = bool(self.options.get("vad_filter", False))

        # faster-whisper 1.2.x calcula la STFT (espectrograma) de TODO el audio de
        # una vez en NumPy dentro del feature_extractor. Con 69 min eso reserva
        # ~1.24 GiB (array complex128) y lanza MemoryError -- ANTES de tocar la
        # diarización. Solución: trocear el audio y transcribir ventana por
        # ventana, pasando a Whisper un array pequeño (no la ruta completa).
        import soundfile as sf
        import numpy as np

        info_sf = sf.info(str(audio_path))
        sr = info_sf.samplerate
        total = info_sf.frames
        asr_chunk_sec = float(self.options.get("asr_chunk_sec", 300.0))  # 5 min
        chunk = int(asr_chunk_sec * sr)

        transcript_segments = []
        start = 0
        while start < total:
            n = min(chunk, total - start)
            offset_sec = start / sr
            with sf.SoundFile(str(audio_path)) as f:
                f.seek(start)
                wave = f.read(frames=n, dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)  # a mono
            # Whisper espera float32 mono a 16 kHz. El audio ya viene a 16 kHz
            # (extract_audio); si no lo estuviera, habría que remuestrear.
            wave = np.ascontiguousarray(wave, dtype=np.float32)

            segs, _info = model.transcribe(
                wave, language=lang,
                word_timestamps=True, vad_filter=use_vad,
            )
            for seg in segs:
                transcript_segments.append({
                    "start": seg.start + offset_sec,
                    "end": seg.end + offset_sec,
                    "text": seg.text.strip(),
                    "words": [
                        {"start": (w.start + offset_sec), "end": (w.end + offset_sec),
                         "word": w.word}
                        for w in (seg.words or [])
                    ],
                })
                if progress_cb is not None:
                    try:
                        progress_cb(float(seg.end + offset_sec))
                    except Exception:  # noqa: BLE001
                        pass
            print(f"[*]   ASR ventana [{offset_sec:.0f}s..{(start+n)/sr:.0f}s] ok")
            start += chunk

        print(f"[*] Diarización con pyannote.audio (device={device}) ...")
        diar_segments = []
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=os.environ.get("HUGGINGFACE_TOKEN"),
            )
            # Mover TODO el pipeline al dispositivo (GPU si hay).
            pipeline.to(torch.device(device))

            diar_segments = self._diarize_chunked(
                pipeline, audio_path,
                min_speakers=self.options.get("min_speakers", 1),
                max_speakers=self.options.get("max_speakers", 6),
                chunk_sec=float(self.options.get("chunk_sec", 600.0)),   # 10 min
                overlap_sec=float(self.options.get("overlap_sec", 30.0)),
            )
        except MemoryError:
            print("[!] Diarización sin memoria suficiente incluso por fragmentos; "
                  "se conserva la transcripción con un único hablante. "
                  "Prueba a bajar chunk_sec (p. ej. 300).")
            diar_segments = []
        except Exception as e:  # noqa: BLE001
            print(f"[!] Diarización falló ({type(e).__name__}: {e}); "
                  f"asignando 'SPEAKER_00' a todos los segmentos.")
            diar_segments = []

        # Asignar cada segmento de transcripción al hablante más cercano.
        # Si no hubo diarización, todos quedan como SPEAKER_00.
        final_segments = self._assign_speakers(transcript_segments, diar_segments)

        return TranscriptionResult(self.name, self.language, final_segments,
                                   model=model_size)

    def _diarize_chunked(self, pipeline, audio_path, min_speakers, max_speakers,
                         chunk_sec=600.0, overlap_sec=30.0) -> list:
        """Diariza audios largos por fragmentos para no reventar la RAM.

        pyannote calcula internamente una FFT (complex128) sobre TODO el audio;
        en una clase de ~69 min eso reserva >1 GiB y lanza MemoryError. Cortando
        el audio en ventanas de `chunk_sec` con `overlap_sec` de solape, cada FFT
        es pequeña. Los tiempos de cada ventana se desplazan por su offset y las
        etiquetas de hablante se reconcilian entre ventanas usando la zona de
        solape.

        Lee cada ventana con soundfile usando seek/frames, así que NUNCA carga el
        audio completo en memoria.
        """
        import soundfile as sf
        import numpy as np
        import torch

        info = sf.info(str(audio_path))
        sr = info.samplerate
        total = info.frames
        chunk = int(chunk_sec * sr)
        overlap = int(overlap_sec * sr)
        step = max(1, chunk - overlap)

        flat: list[dict] = []
        start = 0
        widx = 0
        while start < total:
            n = min(chunk, total - start)
            offset_sec = start / sr
            with sf.SoundFile(str(audio_path)) as f:
                f.seek(start)
                wave = f.read(frames=n, dtype="float32")
            if wave.ndim == 1:
                wave = wave[None, :]
            else:
                wave = wave.T
            audio_dict = {"waveform": torch.from_numpy(np.ascontiguousarray(wave)),
                          "sample_rate": sr}
            diar = pipeline(audio_dict, min_speakers=min_speakers,
                            max_speakers=max_speakers)
            ann = diar.speaker_diarization
            cnt = 0
            for turn, _, speaker in ann.itertracks(yield_label=True):
                flat.append({
                    "start": turn.start + offset_sec,
                    "end": turn.end + offset_sec,
                    "speaker": f"w{widx}_{speaker}",  # única por ventana
                })
                cnt += 1
            print(f"[*]   ventana {widx} [{offset_sec:.0f}s..{(start+n)/sr:.0f}s]: "
                  f"{cnt} turnos")
            widx += 1
            start += step

        # Reconciliar etiquetas entre ventanas (union-find por solapamiento).
        canon = self._merge_speaker_labels(flat)
        for s in flat:
            s["speaker"] = canon[s["speaker"]]
        flat.sort(key=lambda s: s["start"])
        return flat

    def _merge_speaker_labels(self, segs: list) -> dict:
        """Une etiquetas por-ventana (wI_SPEAKER_XX) en hablantes globales.

        BUG CORREGIDO: la versión anterior unía DOS etiquetas cualesquiera cuyos
        segmentos se solapaban en el tiempo. Como las ventanas comparten 30 s de
        solape a propósito, en esa zona coexisten TODOS los hablantes de ambas
        ventanas, así que el union-find los encadenaba a todos en uno solo
        (SPEAKER_00 único, sin importar cuántas voces hubiera). Solapar en el
        tiempo NO significa ser el mismo hablante — al contrario.

        Enfoque correcto: solo se emparejan hablantes de ventanas CONSECUTIVAS, y
        dentro de la zona de solape se empareja cada hablante de la ventana N con
        el hablante de la ventana N+1 con el que comparte MÁS tiempo (matching
        voraz uno-a-uno). Un hablante ausente en el solape simplemente arranca
        una etiqueta global nueva.
        """
        import re

        def widx(label: str) -> int:
            m = re.match(r"w(\d+)_", label)
            return int(m.group(1)) if m else 0

        labels = sorted({s["speaker"] for s in segs})
        parent = {l: l for l in labels}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Segmentos agrupados por ventana.
        by_win: dict[int, list] = {}
        for s in segs:
            by_win.setdefault(widx(s["speaker"]), []).append(s)
        wins = sorted(by_win)

        def overlap(a, b) -> float:
            return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))

        # Para cada par de ventanas consecutivas, medir cuánto tiempo comparte
        # cada hablante de N con cada hablante de N+1, y emparejar 1-a-1 (voraz).
        for wa, wb in zip(wins, wins[1:]):
            shared: dict[tuple, float] = {}
            for sa in by_win[wa]:
                for sb in by_win[wb]:
                    ov = overlap(sa, sb)
                    if ov > 0:
                        key = (sa["speaker"], sb["speaker"])
                        shared[key] = shared.get(key, 0.0) + ov
            # Emparejar por mayor tiempo compartido, sin reusar etiquetas.
            used_a: set = set()
            used_b: set = set()
            for (la, lb), _ in sorted(shared.items(), key=lambda kv: -kv[1]):
                if la in used_a or lb in used_b:
                    continue
                union(la, lb)
                used_a.add(la)
                used_b.add(lb)

        roots: dict = {}
        canon: dict = {}
        for l in labels:
            r = find(l)
            if r not in roots:
                roots[r] = f"SPEAKER_{len(roots):02d}"
            canon[l] = roots[r]
        return canon

    def _assign_speakers(self, transcript: list, diarization: list) -> list[Segment]:
        """Asigna el hablante más cercano en el tiempo a cada segmento transcrito."""
        result = []
        for t in transcript:
            t_start, t_end = t["start"], t["end"]
            # Buscar el segmento de diarización que más se solapa.
            # Si no hubo diarización, todos quedan como SPEAKER_00.
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0
            for d in diarization:
                d_start, d_end = d["start"], d["end"]
                overlap = max(0.0, min(t_end, d_end) - max(t_start, d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d["speaker"]
            result.append(Segment(
                start=t_start, end=t_end,
                speaker=best_speaker, text=t["text"],
            ))
        return result
