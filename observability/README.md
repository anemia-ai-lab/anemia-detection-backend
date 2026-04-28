# Observabilidad local (opcional)

Stack **Prometheus + Grafana** para visualizar las métricas del backend (`GET /metrics`).  
**No forma parte del arranque normal:** la API puede ejecutarse sola (local, Render, Docker propio, etc.). Este `docker-compose` **no incluye** el servicio backend.

---

## Orden de arranque

1. **Arrancar primero el backend** en el host, típicamente **http://localhost:8000** (por ejemplo desde la raíz del repo: `make run`, o `uvicorn backend.main:app --host 127.0.0.1 --port 8000`).
2. **Después** levantar el stack de observabilidad:

```bash
docker compose -f docker-compose.observability.yml up
```

Sin el API en marcha, Prometheus marcará el *target* como caído hasta que el puerto **8000** responda.

---

## Apagado del stack de observabilidad

```bash
docker compose -f docker-compose.observability.yml down
```

El backend sigue siendo un proceso independiente: detenerlo con Ctrl+C en la terminal donde corre `make run`, o según el entorno de despliegue.

---

## URLs

| Servicio    | URL                     | Notas                                      |
|------------|-------------------------|--------------------------------------------|
| Prometheus | http://localhost:9090   | Estado → Targets → `anemia-detection-backend` |
| Grafana    | http://localhost:3000   | Usuario **admin** / contraseña **admin** (solo demo local; cambiar si la máquina es compartida). |

Grafana arranca **sin configuración manual**: el datasource **Prometheus** ya apunta a `http://prometheus:9090` dentro de la red del Compose.

---

## Qué espera Prometheus

Configuración en `observability/prometheus.yml`:

- **Target:** `host.docker.internal:8000` (API en el máquina host).
- **Ruta:** `/metrics`.
- Sin etiquetas extra en `static_configs`; las métricas expuestas por la aplicación siguen las definiciones del backend (sin datos personales en etiquetas).

---

## Resolución de problemas

| Síntoma | Qué comprobar |
|--------|----------------|
| Prometheus muestra el target **DOWN** o no hay series | Que el backend **esté en ejecución** y escuche en el **puerto 8000** del host. |
| Duda sobre si el endpoint responde | Abrir en el navegador **http://localhost:8000/metrics** o ejecutar `curl -sSf http://127.0.0.1:8000/metrics \| head`. |
| Grafana sin datos | Que Prometheus esté **UP** en Status → Targets y que hayas pasado el punto (1) antes que el `docker compose`. |

Si el API corre **solo dentro de otra red Docker** (no en el host), cambie `targets` en `observability/prometheus.yml` al host/puerto que Prometheus pueda alcanzar desde su contenedor.

---

## Demostración (tesis / conferencia)

1. Terminal A: backend en **:8000**.  
2. Terminal B: `docker compose -f docker-compose.observability.yml up`.  
3. Navegador: Grafana → Explore → datasource Prometheus → consultas sobre `http_requests_total`, `model_loaded`, etc.
