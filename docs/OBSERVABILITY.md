# Observabilidad (Prometheus + operación)

Métricas expuestas en `GET /metrics` (protegido con `METRICS_BEARER_TOKEN` fuera de entornos locales).

## Métricas clave

| Métrica | Uso |
|---------|-----|
| `predict_phase_duration_seconds{phase}` | Latencia por fase de `POST /predict` |
| `prediction_errors_total` | Fallos en inferencia online |
| `rate_limit_exceeded_total{path}` | Abuso o tráfico alto (HTTP 429) |
| `sync_metadata_rejected_total{reason}` | Payloads offline incoherentes |
| `sync_completed_total` | Sync offline completado (imagen subida) |
| `model_loaded` | Modelo Keras cargado (0/1) |

## Baseline post-deploy

```bash
export SMOKE_BASE_URL=http://<LoadBalancerDNS>
export METRICS_BEARER_TOKEN=<token>
make metrics-baseline
```

Registrar fecha + p50/p95 por fase antes de cambios de performance.

## Alertas sugeridas (CloudWatch / manual)

| Señal | Umbral orientativo | Acción |
|-------|-------------------|--------|
| p95 `inference` | > 5 s sostenido | Subir vCPU Fargate; `INFERENCE_TTA_ENABLED=false` |
| `prediction_errors_total` | incremento abrupto | Logs ECS; MediaPipe / modelo |
| `rate_limit_exceeded_total` | alto sostenido | Revisar abuso o subir límites |
| `sync_metadata_rejected_total` | > 0 recurrente | App móvil desalineada con tiers/versión |
| Target unhealthy ALB | 2+ checks | Secrets Manager, cold start TF |

No hay Grafana/Alertmanager en el IaC actual; scrape vía script o integración futura.

## Smoke programado

CI smoke ([`docs/RELEASE.md`](RELEASE.md)) incluye sync offline E2E además de `/predict` online.
