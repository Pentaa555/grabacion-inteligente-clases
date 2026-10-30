"""Diagnóstico de errores del WER: alinea referencia e hipótesis y lista cada
sustitución / eliminación / inserción, para ver QUÉ se está contando como error.

Uso:  py wer_detalle.py "WER Ground truth.txt" webdata/salida/<job>/transcripcion.json
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.wer import normalizar, texto_de_hipotesis


def alinear(ref, hip):
    n, m = len(ref), len(hip)
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
                d[i][j] = 1 + min(d[i - 1][j - 1], d[i - 1][j], d[i][j - 1])
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hip[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("SUB", ref[i - 1], hip[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("DEL", ref[i - 1], "—")); i -= 1
        else:
            ops.append(("INS", "—", hip[j - 1])); j -= 1
    return list(reversed(ops))


def main() -> None:
    ref = normalizar(Path(sys.argv[1]).read_text(encoding="utf-8"))
    hip = normalizar(texto_de_hipotesis(Path(sys.argv[2])))
    ops = alinear(ref, hip)
    subs = [(a, b) for k, a, b in ops if k == "SUB"]
    dels = [a for k, a, b in ops if k == "DEL"]
    ins = [b for k, a, b in ops if k == "INS"]
    print(f"\nSUSTITUCIONES ({len(subs)}): referencia -> whisper")
    for a, b in subs:
        print(f"   {a:>18}  ->  {b}")
    print(f"\nELIMINACIONES ({len(dels)}): estaban en la referencia, Whisper las omitió")
    print("   " + ", ".join(dels))
    print(f"\nINSERCIONES ({len(ins)}): Whisper las agregó de más")
    print("   " + ", ".join(ins))


if __name__ == "__main__":
    main()
