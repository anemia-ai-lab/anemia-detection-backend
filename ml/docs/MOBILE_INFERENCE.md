# Contrato de inferencia móvil (React Native + OpenCV + TFLite)

Documento para el cliente offline. El backend replica la misma lógica de calibración y tiers cuando recibe crops ya recortados.

## Captura

1. Pedir al usuario **anular, medio e índice** de la misma mano.
2. Iluminación uniforme; evitar sombras fuertes sobre la uña.
3. OpenCV detecta y recorta cada uña → tensor **224×224 RGB**.

## Por uña (3 dedos × pipeline)

1. `mobilenet_v2.preprocess_input` sobre float32 [0,255] (mismo que entrenamiento).
2. Ejecutar **3 modelos TFLite** (semillas 42, 123, 456):
   - `p_raw = mean(p_raw_seed42, p_raw_seed123, p_raw_seed456)`
3. Calibración (fuera del grafo TFLite):

   ```text
   logit(p) = log(p / (1-p))
   p_cal = sigmoid(logit(p_raw) / T)
   ```

   `T` y umbrales en [`baseline_mobilenetv2_ghana_ensemble.metadata.json`](../artifacts/models/baseline_mobilenetv2_ghana_ensemble.metadata.json) (o metadatos por miembro tras `export_ensemble_tflite.py`).

## Por mano (agregación)

```text
p_hand = max(p_cal_anular, p_cal_medio, p_cal_indice)
```

Cribado sensible: si una uña eleva el score, la mano sube de tier.

## Riesgo bajo / medio / alto

Comparar `p_hand` con:

| Tier | Regla |
|------|--------|
| **low** | `p_hand <= low_upper` |
| **medium** | `low_upper < p_hand < high_lower` |
| **high** | `p_hand >= high_lower` |

`high_lower` = τ Youden en validación Ghana (umbral operacional). `low_upper` = percentil 90 de negativos en val calibrado.

**No es diagnóstico clínico**; el tier medio es zona gris.

## Latencia orientativa

~3 inferencias TFLite × 3 uñas ≈ **9 forwards** por mano (aceptable offline).

## Sincronización con API

Si la app envía crops al backend, usar los mismos `T`, `low_upper`, `high_lower` y agregación documentada aquí.
