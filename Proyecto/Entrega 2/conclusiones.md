## Conclusiones — Entrega Parcial 2

### 1. Resumen ejecutivo
- **Objetivo:** Desplegar en un entorno reproducible el modelo desarrollado en la entrega anterior, orquestar el flujo de datos con Airflow y ofrecer una interfaz para probar inferencia (API y UI con Gradio).
- **Estado actual:** El pipeline principal está implementado (extracción, generación de features, detección básica de drift, reentrenamiento condicional y generación de predicciones). La API en FastAPI responde con el modelo cargado y la UI en Gradio permite probar dos demos (recomendador y asistente conversacional lite). Los artefactos y los datos de prueba están en la estructura del repositorio para reproducir localmente.

### 2. Datos y preparación
- **Conjuntos usados:** `transacciones.parquet`, `clientes.parquet`, `productos.parquet` (ubicados en `airflow/data`).
- **Procesos principales:** limpieza básica (filtrado de valores nulos y normalización de identificadores), agregación temporal para ventanas semanales y construcción de matrices de features para entrenamiento y scoring.
- **Riesgos y mitigaciones:** Se observaron discontinuidades temporales en algunas semanas; se aplicó una ventana de historial controlada y checks para evitar introducir semanas vacías en el entrenamiento. Se documentaron columnas sensibles y no se incluyen datos PII en los artefactos compartidos.

### 3. Metodología y flujo
- **Resumen del flujo:** el DAG ejecuta extracción → construcción de features → análisis de drift → (si aplica) re-entrenamiento → cálculo de explicabilidad (SHAP) → generación y guardado de predicciones.
- **Decisiones clave:** el sistema usa un holdout temporal para validación, y la generación de candidatos se hace con reglas simples (histórico y top-K) para asegurar reproducibilidad rápida en pruebas.

### 4. Modelo y entrenamiento
- **Modelo principal:** LightGBM (por su velocidad y facilidad de uso en tabular). Se usó optimización ligera de hiperparámetros con Optuna para encontrar configuraciones robustas.
- **Validación:** validación temporal (simulando distribución de despliegue) y métricas orientadas a ranking/decisión (se prioriza capacidad de ordenar candidatos relevantes).

### 5. Evaluación
- **Resultados generales:** el modelo muestra mejoras frente a heurísticos simples en pruebas internas (resultados cuantitativos y gráficos están guardados en `airflow/artifacts_entrega1/artifacts_optuna` y en los logs del DAG). En esta entrega priorizamos reproducibilidad y estabilidad sobre la última décima de mejora de métrica.
- **Limitaciones:** en usuarios con pocas interacciones el rendimiento cae (frío). Para estos casos el sistema devuelve recomendaciones globales por popularidad como fallback.

### 6. Interpretabilidad
- **Qué se calculó:** se generó SHAP global para entender las variables más influyentes en la predicción; los resultados apuntan a variables de recencia y frecuencia como las más relevantes.
- **Limitaciones:** el análisis es global y no explora contrafactuales; es suficiente para diagnóstico pero no sustituye un análisis de causalidad.

### 7. Re-entrenamiento y monitoreo
- **Detección de drift:** se implementó una verificación simple sobre cambios en la distribución de las features principales; si se detecta un cambio significativo, el DAG abre la rama de re-entrenamiento.
- **Política:** los artefactos generados (modelos, preprocesadores, metadata) se guardan en `airflow/artifacts_entrega1/` junto con un `metadata.json` que registra parámetros y versión.

### 8. Tracking y reproducibilidad
- **Registro:** los runs y parámetros relevantes se guardan en metadatos JSON en la carpeta de artifacts para poder recuperar la configuración usada en cada ejecución.
- **Recuperar modelo:** para desplegar un modelo ya entrenado, copiar los archivos de `airflow/artifacts_entrega1/artifacts_optuna` al path de modelos que usa la API (la ruta por defecto es relativa al repo; instrucciones completas abajo).

### 9. Despliegue (cómo reproducir localmente)
- Requisitos básicos: Docker + Docker Compose y PowerShell (o terminal UNIX). Python 3.11 para ejecutar scripts locales.
- Comandos rápidos para probar las piezas principales (ejecutar desde la raíz del repo):

```powershell
cd .\app
docker compose build
docker compose up -d
```

- Endpoints útiles:
	- Backend principal: `http://127.0.0.1:8000/health` (o puerto definido en `app/docker-compose.yml`).
	- Bonus recsys: `http://127.0.0.1:8100/health`, recomendaciones en `/recommend/{customer_id}`.
	- Bonus llm (conversational): `http://127.0.0.1:8200/health`, pregunta POST en `/ask`.

### 10. Retos y aprendizajes
- **Lo más difícil:** adaptar rutas y cargas de artefactos para que funcionen igual dentro y fuera de contenedores (problemas con `Path.parents` y rutas relativas). Aprendimos a preferir mounts explicitos (`/data`) y a validar las rutas en arranque.
- **Otras lecciones:** manejar diferencias de encoding/quoting en PowerShell al enviar JSON y documentar comandos exactos para reproducibilidad.

### 11. Trabajo futuro
- Mejorar la pipeline de detección de drift con métricas más robustas y alertas automáticas.
- Añadir tests end-to-end reproducibles que verifiquen todo el DAG y las APIs en CI.
- Incrementar el trabajo de interpretabilidad por usuario y añadir reportes automatizados para stakeholders.


