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

## Sincronización con API (offline → backend)

Contrato HTTP detallado: `/docs` → tag **offline-sync**.

El backend **confía en los resultados TFLite del dispositivo** (no re-inferencia al sincronizar). Usar los mismos `T`, `low_upper`, `high_lower` y agregación documentada arriba.

### Cola local

Por cada predicción offline:

1. Generar `client_id` (UUID v4) en el dispositivo.
2. Persistir localmente: imagen en crudo + metadatos (`TFLiteInferenceResult.to_sync_payload()` + `client_id`, `birth_date`, `notes` opcionales).
3. Incluir en `preprocessing` (JSON) lo útil para auditoría: dedos capturados, `p_cal` por uña, agregación `max`, versión del pipeline OpenCV, etc.

### Paso 1 — metadatos en batch

`POST /predictions/sync/metadata` (Bearer JWT)

```json
{
  "items": [
    {
      "client_id": "uuid-del-dispositivo",
      "risk": "low",
      "score": 0.12,
      "raw_probability": 0.15,
      "calibrated_probability": 0.12,
      "threshold_used": 0.168,
      "prediction": 0,
      "model_version": "v2.0",
      "inference_mode": "tflite_offline",
      "client_created_at": "2026-04-30T08:00:00Z",
      "birth_date": "2016-01-15",
      "notes": "opcional",
      "image_sha256": "hex-opcional-para-validar-en-paso-2",
      "preprocessing": { "fingers": ["thumb", "middle", "index"], "aggregation": "max" }
    }
  ]
}
```

- Máximo **50** items por request.
- **Idempotente** por `(usuario, client_id)`: reintentos devuelven el mismo `id` sin duplicar.
- Respuesta: `{ "results": [{ "client_id", "id", "image_pending", "created" }] }`.

### Paso 2 — imagen por predicción

`POST /predictions/{id}/image` (multipart: campo `image`; opcional `image_sha256`)

- Subir cuando haya red estable (una predicción por request o concurrencia limitada).
- Si `image_sha256` no coincide → **409**.
- Si la imagen ya existe → **200** idempotente con URL firmada.
- Marcar la cola local como sincronizada solo tras este paso exitoso.

### Historial y borrado

| Endpoint | Uso |
|----------|-----|
| `GET /predictions?limit=20&cursor=...` | Lista resumida (riesgo, edad, notas); sin imagen |
| `GET /predictions/{id}` | Detalle + `preprocessing` + `image_signed_url` |
| `DELETE /predictions/{id}` | Borra predicción e imagen en servidor; purgar copia local |

El orden cronológico en historial usa `effective_created_at` = momento de captura offline (`client_created_at`) cuando existe.
