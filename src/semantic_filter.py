"""P6 — Filtrado semántico de intervenciones (aporte de ML original).

Clasifica cada intervención de la clase como:
  - "pertinente"    -> contenido académico (explicación, ejercicio, definición)
  - "no_pertinente" -> conversación paralela, avisos, logística, charla

Arquitectura de DOS CAPAS, pensada para poder crecer sin reescribir el pipeline:

  featurizer (texto -> vector)  +  clasificador ligero (vector -> etiqueta)

  * Featurizer por defecto: TF-IDF (scikit-learn puro). No descarga modelos, no
    usa GPU, entrena en segundos. Es el que corre HOY en un equipo con RAM justa.
  * Featurizer futuro: embeddings de BETO (BERT en español). Es el refinamiento
    previsto para la sustentación: mismo clasificador encima, solo cambia cómo se
    vectoriza el texto. La interfaz `Featurizer` deja ese cambio enchufable.

El clasificador es una Regresión Logística (rápida, interpretable, estándar para
clasificación de texto de tamaño pequeño-medio). El camino "C" del diseño:
embeddings + clasificador ligero, entrenado con datos propios.

Uso:
    from src.semantic_filter import SemanticFilter
    clf = SemanticFilter()
    clf.train("data/filtrado_dataset.csv")     # entrena y evalúa
    clf.save("models/semantic_filter.joblib")
    ...
    clf = SemanticFilter.load("models/semantic_filter.joblib")
    etiqueta, confianza = clf.predict("saquen el cuaderno")  # -> ("no_pertinente", 0.87)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


# ---------------------------------------------------------------------------
# Capa 1: featurizer (texto -> vector). Interfaz enchufable.
# ---------------------------------------------------------------------------
class Featurizer(Protocol):
    """Convierte una lista de textos en una matriz de características."""

    def fit_transform(self, texts: list[str]): ...
    def transform(self, texts: list[str]): ...


def make_tfidf_featurizer():
    """Featurizer por defecto: TF-IDF sobre 1-2 gramas de palabras.

    Ligero, sin descargas ni GPU. Adecuado para el prototipo actual y como
    línea base honesta frente a la versión con embeddings BETO.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )


# NOTA (refinamiento para la sustentación): un BetoFeaturizer implementaría la
# misma interfaz cargando "dccuchile/bert-base-spanish-wwm-uncased" y devolviendo
# el embedding [CLS] o el promedio de tokens por frase. No se activa ahora para
# no cargar torch/BETO en un equipo con RAM ajustada. El resto del módulo (entrenar,
# evaluar, predecir, guardar) NO cambia al sustituir el featurizer.


# ---------------------------------------------------------------------------
# Resultado de evaluación (para reportar en la tesis).
# ---------------------------------------------------------------------------
@dataclass
class EvalReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_train: int
    n_test: int
    labels: list[str]
    report_text: str

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "labels": self.labels,
        }


# ---------------------------------------------------------------------------
# Capa 2: el filtro semántico completo (featurizer + clasificador).
# ---------------------------------------------------------------------------
class SemanticFilter:
    """Clasificador de intervenciones pertinente / no_pertinente."""

    def __init__(self, featurizer=None):
        # Se crea perezosamente en train()/load() para no importar sklearn si no se usa.
        self._featurizer = featurizer
        self._clf = None
        self._labels: list[str] = []

    # ---- datos -----------------------------------------------------------
    @staticmethod
    def load_dataset(csv_path: str | Path) -> tuple[list[str], list[str]]:
        """Lee un CSV con columnas 'texto,label'. Devuelve (textos, etiquetas)."""
        texts: list[str] = []
        labels: list[str] = []
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "texto" not in reader.fieldnames \
                    or "label" not in reader.fieldnames:
                raise ValueError("El CSV debe tener cabeceras 'texto,label'.")
            for row in reader:
                t = (row.get("texto") or "").strip()
                l = (row.get("label") or "").strip()
                if t and l:
                    texts.append(t)
                    labels.append(l)
        if not texts:
            raise ValueError(f"Dataset vacío: {csv_path}")
        return texts, labels

    # ---- entrenamiento + evaluación -------------------------------------
    def train(self, csv_path: str | Path, test_size: float = 0.25,
              random_state: int = 42) -> EvalReport:
        """Entrena el clasificador y devuelve un informe de evaluación.

        Divide el dataset en train/test, entrena sobre train y mide sobre test.
        Con datasets muy pequeños el test es orientativo (se indica n_test).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                      classification_report)

        texts, labels = self.load_dataset(csv_path)
        self._labels = sorted(set(labels))

        # Estratificar si cada clase tiene al menos 2 ejemplos; si no, split simple.
        from collections import Counter
        counts = Counter(labels)
        stratify = labels if all(c >= 2 for c in counts.values()) and len(counts) > 1 else None

        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=test_size,
            random_state=random_state, stratify=stratify,
        )

        feat = self._featurizer or make_tfidf_featurizer()
        self._clf = Pipeline([
            ("feat", feat),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self._clf.fit(X_tr, y_tr)

        y_pred = self._clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0,
        )
        rep = classification_report(y_te, y_pred, zero_division=0)

        return EvalReport(
            accuracy=float(acc), precision=float(p), recall=float(r), f1=float(f1),
            n_train=len(X_tr), n_test=len(X_te), labels=self._labels, report_text=rep,
        )

    # ---- inferencia ------------------------------------------------------
    def predict(self, text: str) -> tuple[str, float]:
        """Clasifica un texto. Devuelve (etiqueta, confianza[0..1])."""
        if self._clf is None:
            raise RuntimeError("El filtro no está entrenado; llama a train() o load().")
        label = self._clf.predict([text])[0]
        try:
            proba = self._clf.predict_proba([text])[0]
            conf = float(max(proba))
        except Exception:  # noqa: BLE001 — algún clasificador podría no dar proba
            conf = 1.0
        return str(label), conf

    def predict_many(self, texts: Iterable[str]) -> list[tuple[str, float]]:
        return [self.predict(t) for t in texts]

    def filter_segments(self, segments: list) -> list:
        """Anota cada Segment con .relevant (bool) y .relevance (confianza).

        Recibe una lista de objetos con atributo .text (p. ej. types.Segment) y
        les asigna dinámicamente si son pertinentes. No borra nada: marca, para
        que la interfaz decida si oculta o resalta las conversaciones paralelas.
        """
        for s in segments:
            label, conf = self.predict(getattr(s, "text", "") or "")
            s.relevant = (label == "pertinente")
            s.relevance = round(conf, 4)
            s.relevance_label = label
        return segments

    # ---- persistencia ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "labels": self._labels}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SemanticFilter":
        import joblib
        data = joblib.load(path)
        obj = cls()
        obj._clf = data["clf"]
        obj._labels = data.get("labels", [])
        return obj


# ---------------------------------------------------------------------------
# CLI mínima: entrenar + evaluar + probar unas frases.
#   py -m src.semantic_filter data/filtrado_dataset.csv
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    dataset = argv[1] if len(argv) > 1 else "data/filtrado_dataset.csv"
    clf = SemanticFilter()
    report = clf.train(dataset)
    print("=== Evaluación del filtrado semántico (P6) ===")
    print(f"Dataset: {dataset}  |  train={report.n_train}  test={report.n_test}")
    print(f"Accuracy : {report.accuracy:.2%}")
    print(f"Precision: {report.precision:.2%}  (macro)")
    print(f"Recall   : {report.recall:.2%}  (macro)")
    print(f"F1       : {report.f1:.2%}  (macro)")
    print()
    print(report.report_text)

    # Guardar el modelo entrenado.
    out = clf.save("models/semantic_filter.joblib")
    print(f"Modelo guardado en: {out}")

    # Prueba rápida con frases nuevas.
    pruebas = [
        "la fuerza entre dos cargas es proporcional al producto de las cargas",
        "oigan ya vieron el partido de ayer",
        "definimos el campo eléctrico como la fuerza por unidad de carga",
        "profe puedo ir al baño",
    ]
    print("\n=== Ejemplos ===")
    for t in pruebas:
        label, conf = clf.predict(t)
        print(f"  [{label:14s} {conf:5.1%}]  {t}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
