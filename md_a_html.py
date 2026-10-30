"""Convierte el DOCUMENTO_DE_ENTREGA.md a un HTML autocontenido e imprimible.

No requiere librerías externas: soporta el subconjunto de Markdown usado en el
documento (encabezados, tablas, listas, negrita, cursiva, citas, código en línea
y reglas horizontales). El HTML resultante se puede abrir en el navegador e
imprimir a PDF con Ctrl+P → "Guardar como PDF".

Uso:  py md_a_html.py DOCUMENTO_DE_ENTREGA.md DOCUMENTO_DE_ENTREGA.html
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

CSS = """
:root{--acc:#2b6a72;--gold:#c8701f;--txt:#1c2733;--mut:#5b6b78;--line:#d9e0e6;--bg:#fff}
*{box-sizing:border-box}
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--txt);
  max-width:820px;margin:0 auto;padding:48px 40px;background:var(--bg)}
h1{font-size:26px;border-bottom:3px solid var(--acc);padding-bottom:10px;margin-top:0}
h2{font-size:20px;color:var(--acc);margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:5px}
h3{font-size:16px;color:var(--txt);margin-top:24px}
p{margin:10px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:8px 11px;text-align:left}
th{background:#eef3f5;color:var(--txt);font-weight:600}
td:last-child,th:last-child{text-align:right}
td:first-child,th:first-child{text-align:left}
blockquote{border-left:4px solid var(--gold);background:#fbf5ee;margin:14px 0;
  padding:10px 16px;color:#5a4632;border-radius:0 6px 6px 0}
blockquote p{margin:4px 0}
code{background:#eef3f5;padding:2px 6px;border-radius:4px;font-size:13px;
  font-family:Consolas,Menlo,monospace}
ul{margin:10px 0;padding-left:26px}
li{margin:5px 0}
hr{border:0;border-top:1px solid var(--line);margin:26px 0}
strong{color:var(--txt)}
em{color:var(--mut)}
@media print{body{padding:0}h2{page-break-after:avoid}table,blockquote{page-break-inside:avoid}}
"""


def inline(text: str) -> str:
    """Formato en línea: negrita, cursiva, código. Escapa HTML primero."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def fila_tabla(linea: str, celda="td") -> str:
    celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
    tag = celda
    return "<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in celdas) + "</tr>"


def convertir(md: str) -> str:
    lineas = md.splitlines()
    out: list[str] = []
    i = 0
    en_lista = False

    def cerrar_lista():
        nonlocal en_lista
        if en_lista:
            out.append("</ul>")
            en_lista = False

    while i < len(lineas):
        ln = lineas[i]
        s = ln.strip()

        # Tabla: fila de encabezado seguida de separador |---|
        if s.startswith("|") and i + 1 < len(lineas) and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lineas[i + 1]):
            cerrar_lista()
            out.append('<table>')
            out.append("<thead>" + fila_tabla(s, "th") + "</thead><tbody>")
            i += 2
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                out.append(fila_tabla(lineas[i]))
                i += 1
            out.append("</tbody></table>")
            continue

        if not s:
            cerrar_lista()
            i += 1
            continue
        if s.startswith("### "):
            cerrar_lista(); out.append(f"<h3>{inline(s[4:])}</h3>")
        elif s.startswith("## "):
            cerrar_lista(); out.append(f"<h2>{inline(s[3:])}</h2>")
        elif s.startswith("# "):
            cerrar_lista(); out.append(f"<h1>{inline(s[2:])}</h1>")
        elif s.startswith("> "):
            cerrar_lista(); out.append(f"<blockquote><p>{inline(s[2:])}</p></blockquote>")
        elif s.startswith("- "):
            if not en_lista:
                out.append("<ul>"); en_lista = True
            out.append(f"<li>{inline(s[2:])}</li>")
        elif re.match(r"^-{3,}$", s):
            cerrar_lista(); out.append("<hr>")
        else:
            cerrar_lista(); out.append(f"<p>{inline(s)}</p>")
        i += 1

    cerrar_lista()
    cuerpo = "\n".join(out)
    return (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Documento de Entrega</title>"
        f"<style>{CSS}</style></head><body>{cuerpo}</body></html>"
    )


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(convertir(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"HTML generado: {dst}")


if __name__ == "__main__":
    main()
