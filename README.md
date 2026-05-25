# Sistema de Predicción de Tasa de Penetración de Gimnasios por Pais y Año

Diseñado e implementado con una red neuronal de aprendizaje automático (TensorFlow y Keras).

## ¿De qué se trata?

Este sistema predice la **tasa de penetración de gimnasios** (`gym_penetration_rate`) por país y año — es decir, qué proporción de la población total cuenta con una membresía activa en un gimnasio.

El modelo se entrena con datos históricos de tendencias de gimnasios a nivel mundial del año 2000 al 2026, obtenidos del dataset [World Gym and Fitness Trends 2000–2026](https://www.kaggle.com/datasets/aryanmdev/world-gym-and-fitness-trends-20002026).

## ¿Cómo funciona?

1. **EDA** (`EDA.ipynb`): se analizan los datos, se identifican correlaciones, valores atípicos y se seleccionan las variables más relevantes como entrada al modelo.

2. **Carga y preprocesamiento** (`data.py`):
   - Lee el archivo `clean_gym_data.csv`.
   - Aplica codificación one-hot a la variable categórica `region`.
   - Transforma el target con `log1p` para reducir la asimetría de la distribución.
   - Expone funciones: `load_data()`, `load_raw()`, `sample_rows()`.

3. **Modelo** (`model.py`):
   - Construye una red neuronal secuencial con capas densas (64 → 32 → 1).
   - Normaliza las entradas con `MinMaxScaler` al rango [0, 1].
   - Divide los datos en entrenamiento (80%) y validación (20%) con `train_test_split`.
   - Usa función de pérdida Huber (robusta a valores atípicos) y métrica MAE.
   - Guarda el modelo en `gym_model.keras` y el scaler en `gym_scaler.pkl`.
   - Expone funciones: `train_and_save()`, `load_model_and_meta()`, `predict()`, `predict_with_loaded()`, `predict_batch()`.

4. **Predicciones con threads** (`threads.py`):
   - Recibe una lista de tareas `(etiqueta, features)`.
   - Carga el modelo una sola vez y lanza un hilo (`threading.Thread`) por tarea.
   - Usa un `Semaphore` para limitar la concurrencia y un `Lock` para escrituras seguras al diccionario de resultados.
   - Devuelve los resultados en el orden original de las tareas.

5. **Main file** (`main.py`):
   - Si no existe un modelo entrenado, lanza el entrenamiento automáticamente.
   - Muestra predicciones sobre muestras reales del dataset (con error respecto al valor actual).
   - Muestra predicciones de escenarios futuros (año 2030) para países de diferentes regiones.

## Características del dataset

| Variable | Descripción |
|---|---|
| `gdp_per_capita_usd` | PIB per cápita en USD |
| `urban_population_percentage` | Proporción de población urbana (0–1) |
| `fitness_participation_rate` | Proporción que practica alguna actividad física (0–1) |
| `obesity_rate` | Proporción de la población con obesidad (0–1) |
| `average_membership_cost_usd` | Costo promedio de membresía en USD |
| `insufficient_physical_activity_pct` | Proporción sin actividad física suficiente (0–1) |
| `year` | Año de la observación (2000–2026) |
| `region` | Región geográfica (codificada como one-hot) |
| `gym_penetration_rate` | **Target** — proporción de la población con membresía activa |

## Instalación y ejecución desde cero

### Requisitos
- Python 3.11 (requerido por TensorFlow)
- [uv](https://docs.astral.sh/uv/) (recomendado) o pip

### Con uv

```bash
# 1. Instalar dependencias
uv sync

# 2. Ejecutar el sistema
uv run main.py
```

### Con pip

```bash
# 1. Crear entorno virtual con Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el sistema
python main.py
```

### Primera ejecución

Si no existe un modelo entrenado (`gym_model.keras`), el sistema lo entrena automáticamente al arrancar. El entrenamiento tarda aproximadamente 1–2 minutos. Los archivos generados son:

- `gym_model.keras` — pesos y arquitectura de la red neuronal
- `gym_scaler.pkl` — scaler MinMaxScaler serializado con joblib

En ejecuciones posteriores, el modelo se carga directamente sin reentrenar.

## Estructura del proyecto

```
├── data.py                  # Carga y preprocesamiento del dataset
├── model.py                 # Arquitectura, entrenamiento e inferencia
├── threads.py               # Predicciones concurrentes con threading
├── main.py                  # Punto de entrada principal
├── EDA.ipynb                # Análisis exploratorio de datos
├── clean_gym_data.csv       # Dataset limpio
├── requirements.txt         # Dependencias del proyecto
└── pyproject.toml           # Configuración del proyecto
```
