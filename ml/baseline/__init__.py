"""Baseline de clasificación con MobileNetV2 + transfer learning (fuera del API FastAPI).

Estructura de artefactos (bajo ``ml/``)::

    artifacts/
      models/     # *.keras (ignorados por git salvo .gitkeep)
      metrics/    # JSON de evaluación
      runs/       # JSON por entrenamiento (historiales)
    data/
      train/<clase_negativa>/   # p. ej. negative/
      train/<clase_positiva>/   # p. ej. positive/
      test/...
"""
