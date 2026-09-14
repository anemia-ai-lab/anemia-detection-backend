# Contrato de inferencia móvil (React Native + TFLite offline)

Documento para el cliente móvil. La app **solo captura** una foto con índice, medio y anular en el encuadre guía.

**Alcance (Etapa 1+2, no es el fix definitivo):** la agregación median, el mínimo de 2 uñas, el dedup Ghana y Platt reducen la **amplificación** del sesgo positivo (una uña ruidosa ya no decide la mano vía `max`). **No cierran el domain gap.** Fotos reales de prueba siguieron dando falso positivo en el anular incluso con un crop ~120 px, más apretado que `crop_scale=0.75`. Eso requiere **Etapa 3**: fine-tuning con fotos tipo-producción. `crop_scale` default sigue en **1.0** hasta un smoke de fotos reales 1.0 vs 0.75 (`scripts/compare_nail_crop_scale.py`).

## Modo online (`POST /predict`)

Con conexión, la app envía **una sola imagen** (multipart `image`). El backend hace el 100%:

1. **MediaPipe Hand Landmarker** (Tasks API) localiza índice, medio y anular.
2. Recorta cada uña al eje tip→DIP (rotado, centro hacia el lecho; OpenCV).
3. Por uña: ensemble Keras (3 semillas) → calibración.
4. Agrega `p_hand` con mediana conservadora de `p_cal` (n par = mínimo; n=3 = valor central) y persiste una predicción (`inference_mode: backend`). Hacen falta **≥2** uñas válidas.

Detalle por uña en `preprocessing.nails` de la respuesta. La app no recorta ni infiere en este modo.

**Multipart:** solo el campo `image` es obligatorio. El campo opcional `rois` (JSON de ROIs normalizadas) es **solo debug/ops** en el backend; la app móvil **no debe enviarlo**.

**Límites de subida (defaults):** hasta **20 MB** y **50 MP**; JPEG/PNG/WebP (no HEIC). El backend infiere sobre la imagen decodificada completa y guarda en Storage un PNG reducido (~1024 px de lado).

**Sin detección de uñas:** HTTP **400**, `code`: `no_fingernail_detected`. La app debe mostrar “Vuelve a tomar la foto” (índice, medio, anular y palma visibles) y **no** mostrar riesgo.

## Modo offline (TFLite en dispositivo)

Sin red, la app ejecuta detección, recorte, ensemble TFLite, calibración y tiers en el teléfono. Al sincronizar, el backend **confía en esos resultados** (no re-inferencia) y valida coherencia interna del payload.

### Artefactos embebidos (v2.0)

No es un único `.tflite`: son **3 modelos + 1 JSON de ensemble**. Opcionalmente, el modelo MediaPipe para paridad con el backend online.

| Archivo | Obligatorio | Uso |
|---------|-------------|-----|
| `baseline_mobilenetv2_ghana_augmented_seed42.tflite` | Sí | Miembro 1 del ensemble |
| `baseline_mobilenetv2_ghana_augmented_seed123.tflite` | Sí | Miembro 2 |
| `baseline_mobilenetv2_ghana_augmented_seed456.tflite` | Sí | Miembro 3 |
| `baseline_mobilenetv2_ghana_ensemble.metadata.json` | Sí | Calibración, umbrales y reglas de agregación |
| `hand_landmarker.task` | Recomendado | MediaPipe Hand Landmarker (detección offline) |

Ruta en repo: `ml/artifacts/models/`. Los `.keras` son solo para el servidor; no van en la app.

**Versión en sync:** usar `model_version: "v2.0"` (valor de `MODEL_VERSION` / `GET /health`). El JSON del ensemble puede decir `"v2.0-ensemble"`; eso identifica el bundle ML, no el campo de sync.

### Constantes oficiales (ensemble v2.0)

Valores alineados con [`baseline_mobilenetv2_ghana_ensemble.metadata.json`](../artifacts/models/baseline_mobilenetv2_ghana_ensemble.metadata.json) y `backend/core/config.py`:

| Constante | Valor |
|-----------|-------|
| `temperature` (T) | `0.9443417710165931` (rollback; inferencia default = Platt) |
| `platt_a` / `platt_b` | `1.4673747627390392` / `0.6250000000000003` |
| `operational_threshold` / `high_lower` | `0.5780355600619943` |
| `low_upper` | `0.49133022605269516` |
| `ensemble_aggregation` | `mean_raw_probability` |
| `per_hand_nail_aggregation` | `median_calibrated_probability` |
| `nail_aggregation_even_n` | `lower_central` (n=2 → mínimo; **no** promedio de `statistics.median`) |
| `min_nail_count` | `2` |
| `crop_scale` | `1.0` (candidato 0.75: no fijar sin smoke de fotos reales) |
| `crop` | `tip_to_dip_rotated` (centro 0.35 hacia DIP; eje vertical) |
| `calibration_method` | `platt` |
| `preprocessing` | `mobilenet_v2.preprocess_input` |
| Input del modelo | **224×224 RGB** |

En sync, `threshold_used` debe ser **`high_lower`** (`0.5780355600619943`).

## Captura (ambos modos)

1. Pedir al usuario **anular, medio e índice** de la misma mano.
2. Iluminación uniforme; evitar sombras fuertes sobre la uña.
3. **Online:** enviar la foto completa al backend.
4. **Offline:** detectar landmarks y recortar cada uña en dispositivo → tensor **224×224 RGB**.

## Detección y recorte offline (paridad con backend)

Para resultados comparables con `POST /predict`, replicar la lógica de [`backend/inference/nail_detection.py`](../../backend/inference/nail_detection.py):

1. **MediaPipe Hand Landmarker** (`hand_landmarker.task`, Tasks API, `num_hands=1`).
2. Landmarks por dedo (tip, DIP): índice `(8, 7)`, medio `(12, 11)`, anular `(16, 15)`.
3. Recorte **rotado** (`preprocessing.crop = tip_to_dip_rotated`), no un cuadrado axis-aligned en el tip:
   - Centro hacia el lecho: `center = tip - 0.35 * (tip - DIP)` (entre punta y articulación DIP).
   - Rotar con affine para que DIP→tip quede **vertical** (punta hacia −Y).
   - Cuadrado `side = finger_len * 1.6 * crop_scale`. `crop_scale = 1.0` (default del backend). `0.75` es candidato; el test geométrico (bbox más chico) **no** valida falsos positivos.
   - `bbox` = AABB del cuadrado en coordenadas de la foto original.
4. Redimensionar a 224×224 (bilinear) antes de `mobilenet_v2.preprocess_input`.
5. Si hay **menos de 2** uñas válidas, pedir otra captura (`no_fingernail_detected`).

OpenCV (o equivalente nativo) realiza el crop/resize; **no sustituye** la detección de mano. ROIs fijas del overlay sin landmarks son más frágiles y no están garantizadas contra el backend.

Si no se detecta mano o faltan uñas válidas, la app debe pedir otra captura (análogo a `no_fingernail_detected` en online).

## Por uña (3 dedos × pipeline)

1. `mobilenet_v2.preprocess_input` sobre float32 [0,255] (mismo que entrenamiento).
2. Ejecutar **3 modelos TFLite** (semillas 42, 123, 456):

   ```text
   p_raw = mean(p_raw_seed42, p_raw_seed123, p_raw_seed456)
   ```

3. Calibración (fuera del grafo TFLite). Default actual = **Platt**:

   ```text
   logit(p) = log(p / (1-p))
   p_cal = sigmoid(a * logit(p_raw) + b)
   ```

   Temperature scaling (rollback; `calibration_method = temperature`): `p_cal = sigmoid(logit(p_raw) / T)`.

   Cada `.tflite` devuelve probabilidad sigmoide **sin calibrar** (`raw_output_is_sigmoid_probability: true`).

## Por mano (agregación)

Default: mediana **conservadora en n par** (misma regla que `majority_2_of_3` para 1–3 uñas):

```text
ordenar p_cal ascendente
n impar → valor central          # n=3 → el del medio
n par   → el menor de los dos centrales   # n=2 → mínimo, no el promedio 0.55
p_hand = ese valor
```

No usar el `median()` de la stdlib con 2 uñas: `[0.9, 0.2]` promediaría 0.55 y seguiría por encima del umbral operacional (~0.38).

Rollback: `PREDICT_NAIL_AGGREGATION=max` (una uña alta sube el tier de la mano; es el sesgo que se mitiga aquí).

Decisión binaria y `prediction` en sync:

```text
prediction = 1  si p_hand >= threshold_used
prediction = 0  en caso contrario
```

## Riesgo bajo / medio / alto

Comparar `p_hand` (probabilidad calibrada agregada) con:

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

Al subir imagen (`POST /predictions/{id}/image`), el backend **no** ejecuta `nail_checker` en predicciones `tflite_offline` (la inferencia ya ocurrió en dispositivo).

### Validación del backend (422)

Antes de insertar, el servidor comprueba coherencia interna:

| Regla | Código |
|-------|--------|
| `model_version === "v2.0"` | `sync_model_version_mismatch` |
| `score === calibrated_probability` (±1e-4) | `sync_score_mismatch` |
| `risk` coherente con tiers y `calibrated_probability` | `sync_risk_mismatch` |
| `prediction` coherente con `calibrated_probability` y `threshold_used` | `sync_prediction_mismatch` |
| `inference_mode === "tflite_offline"` | error de validación |

### Cola local

Por cada predicción offline:

1. Generar `client_id` (UUID v4) en el dispositivo.
2. Persistir localmente: imagen en crudo + metadatos (`client_id`, `birth_date`, `notes` opcionales).
3. Incluir en `preprocessing` (JSON) trazabilidad útil: `aggregation`, `crop`, `nails` (por dedo: `finger`, `raw`, `cal`, `bbox`), `winning_finger`, etc.

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
      "threshold_used": 0.5780355600619943,
      "prediction": 0,
      "model_version": "v2.0",
      "inference_mode": "tflite_offline",
      "client_created_at": "2026-04-30T08:00:00Z",
      "birth_date": "2016-01-15",
      "notes": "opcional",
      "image_sha256": "hex-opcional-para-validar-en-paso-2",
      "preprocessing": {
        "aggregation": "median",
        "crop": "tip_to_dip_rotated",
        "crop_scale": 1.0,
        "detector": "mediapipe_hand_landmarker",
        "winning_finger": "index",
        "nails": [
          { "finger": "index", "raw": 0.14, "cal": 0.12, "bbox": [120, 80, 64, 64] },
          { "finger": "middle", "raw": 0.11, "cal": 0.09, "bbox": [200, 75, 62, 62] },
          { "finger": "ring", "raw": 0.10, "cal": 0.08, "bbox": [280, 78, 60, 60] }
        ]
      }
    }
  ]
}
```

- `score` debe ser **igual** a `calibrated_probability`.
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
