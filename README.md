# Predicción de diabetes con Boosting (XGBoost)

Comparación de XGBoost, árbol de decisión y Random Forest para predecir diabetes a partir de variables clínicas.

**Stack:** Python · pandas · scikit-learn · XGBoost · matplotlib

## Problema
Identificar a pacientes con riesgo de diabetes y comparar si un modelo de *boosting* mejora a los modelos basados en árboles.

## Datos
Dataset *Pima Indians Diabetes*: 768 pacientes, 8 variables clínicas y la variable objetivo `Outcome`. División 80/20: 614 muestras de entrenamiento y 154 de prueba.

## Enfoque
1. Entrenamiento de un `XGBClassifier` base.
2. Evaluación con accuracy y `classification_report`.
3. Búsqueda del número de árboles óptimo (`n_estimators` de 1 a 49) y gráfico de accuracy.
4. Comparación con `DecisionTreeClassifier` y `RandomForestClassifier`.
5. Modelo final guardado en `src/models/best_model.json`.

## Resultados

| Modelo | Accuracy (test) |
|---|---|
| XGBoost (por defecto) | 0.72 |
| Random Forest | 0.73 |
| Árbol de decisión | **0.74** |

XGBoost por defecto: precisión 0.59, recall 0.71 y F1 0.64 en la clase positiva (diabetes).

## Conclusión
Con este dataset pequeño (768 filas), los tres modelos rinden de forma parecida (72–74 %). El recall del 71 % en la clase positiva es un buen punto de partida; con ajuste de hiperparámetros (`learning_rate`, `max_depth`) y validación cruzada se podría obtener una comparación más robusta.

## Estructura
```
├── src/explore.ipynb       # Análisis y modelado
├── src/models/             # Modelo final
└── requirements.txt
```
