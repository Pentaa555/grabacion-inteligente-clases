"""Cálculo de WER (Word Error Rate) para evaluar la transcripción automática.

Compara una transcripción de referencia hecha a mano (ground truth) contra la
hipótesis producida por el motor de ASR, usando la distancia de edición a nivel
de palabra (algoritmo de Levenshtein), que es la métrica estándar en la
literatura de reconocimiento de voz.

    WER = (S + D + I) / N

donde S = sustituciones, D = eliminaciones, I = inserciones y N = número de
palabras de la referencia.

Uso:
    py -m src.wer "WER Ground truth.txt" webdata/salida/<job_id>/transcripcion.json

El segundo argumento puede ser un .json de salida (se extrae el texto de los
segmentos) o un .txt con la transcripción automática.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

# Mapa de números escritos con palabra -> cifra, para que "nueve" y "9" no
# cuenten como error. Cubre lo que aparece en clases de física.
_PALABRA_A_NUMERO = {
    "cero": "0", "uno": "1", "una": "1", "dos": "2", "tres": "3",
    "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
    "nueve": "9", "diez": "10", "once": "11", "doce": "12",
}


def normalizar(texto: str) -> list[str]:
    """Pasa el texto a una lista de palabras comparables de forma justa.

    - minúsculas
    - quita tildes/acentos (á -> a)
    - quita toda puntuación y símbolos, dejando solo letras/dígitos y espacios
    - unifica coma decimal (8,987 -> 8.987) para no romper los números
    - convierte números escritos a cifra (nueve -> 9)
    """
    texto = texto.lower()
    # Unificar coma decimal entre dígitos ANTES de quitar puntuación.
    salida = []
    for i, ch in enumerate(texto):
        if ch == "," and 0 < i < len(texto) - 1 and texto[i - 1].isdigit() and texto[i + 1].isdigit():
            salida.append(".")
        else:
            salida.append(ch)
    texto = "".join(salida)
    # Quitar acentos.
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    # Dejar solo letras, dígitos, punto decimal y espacios.
    texto = "".join(c if (c.isalnum() or c in ". ") else " " for c in texto)
    # Quitar los puntos que NO son decimales (no están entre dos dígitos):
    # así "cuadrado." -> "cuadrado" pero "8.987" se conserva. Se cuenta como
    # error solo lo que difiere en la PALABRA, no en la puntuación.
    chars = list(texto)
    for k, ch in enumerate(chars):
        if ch == ".":
            prev = chars[k - 1] if k > 0 else ""
            nxt = chars[k + 1] if k + 1 < len(chars) else ""
            if not (prev.isdigit() and nxt.isdigit()):
                chars[k] = " "
    texto = "".join(chars)
    palabras = texto.split()
    # Palabra -> número.
    palabras = [_PALABRA_A_NUMERO.get(p, p) for p in palabras]
    return palabras


def _levenshtein_palabras(ref: list[str], hip: list[str]) -> tuple[int, int, int]:
    """Devuelve (sustituciones, eliminaciones, inserciones) mínimas.

    Programación dinámica clásica de distancia de edición, con backtrace para
    contar cada tipo de error por separado.
    """
    n, m = len(ref), len(hip)
    # d[i][j] = costo mínimo de alinear ref[:i] con hip[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hip[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(
                    d[i - 1][j - 1],  # sustitución
                    d[i - 1][j],      # eliminación
                    d[i][j - 1],      # inserción
                )
    # Backtrace para separar S / D / I.
    i, j = n, m
    s = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hip[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dele, ins


def texto_de_hipotesis(ruta: Path) -> str:
    """Extrae el texto de la hipótesis: si es JSON de salida, une los segmentos."""
    if ruta.suffix.lower() == ".json":
        data = json.loads(ruta.read_text(encoding="utf-8"))
        return " ".join(seg["text"] for seg in data.get("segments", []))
    return ruta.read_text(encoding="utf-8")


def calcular_wer(ref_texto: str, hip_texto: str) -> dict:
    ref = normalizar(ref_texto)
    hip = normalizar(hip_texto)
    s, d, i = _levenshtein_palabras(ref, hip)
    n = len(ref)
    wer = (s + d + i) / n if n else 0.0
    aciertos = n - s - d
    return {
        "palabras_referencia": n,
        "palabras_hipotesis": len(hip),
        "sustituciones": s,
        "eliminaciones": d,
        "inserciones": i,
        "aciertos": aciertos,
        "wer": wer,
        "precision_palabra": aciertos / n if n else 0.0,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ref_path = Path(sys.argv[1])
    hip_path = Path(sys.argv[2])
    ref_texto = ref_path.read_text(encoding="utf-8")
    hip_texto = texto_de_hipotesis(hip_path)
    r = calcular_wer(ref_texto, hip_texto)

    print("=" * 52)
    print("  MEDICIÓN DE WER — Transcripción automática vs manual")
    print("=" * 52)
    print(f"  Referencia (manual):   {ref_path.name}")
    print(f"  Hipótesis (Whisper):   {hip_path.name}")
    print("-" * 52)
    print(f"  Palabras en referencia:   {r['palabras_referencia']:>6}")
    print(f"  Palabras en hipótesis:    {r['palabras_hipotesis']:>6}")
    print(f"  Aciertos:                 {r['aciertos']:>6}")
    print(f"  Sustituciones (S):        {r['sustituciones']:>6}")
    print(f"  Eliminaciones (D):        {r['eliminaciones']:>6}")
    print(f"  Inserciones (I):          {r['inserciones']:>6}")
    print("-" * 52)
    print(f"  WER  = (S+D+I)/N     =    {r['wer'] * 100:>6.2f} %")
    print(f"  Precisión de palabra =    {r['precision_palabra'] * 100:>6.2f} %")
    print("=" * 52)
    meta = 20.0
    if r["wer"] * 100 < meta:
        print(f"  ✅ CUMPLE la meta de la tesis (WER < {meta:.0f} %)")
    else:
        print(f"  ⚠️  Por encima de la meta (WER < {meta:.0f} %)")
    print("=" * 52)


if __name__ == "__main__":
    main()
