# Detección de fraude en transacciones financieras

Comparación de cuatro clasificadores supervisados (**k-NN**, **Árbol de
Decisión**, **SVM lineal** y un **Random Forest propio**) para detectar
fraudes con tarjetas de crédito. El criterio principal es maximizar la
**sensibilidad (recall)** de la clase fraude; la precisión se usa como
desempate.

## Estructura

```
.
├── main.py                         # Orquestador de los 8 experimentos
├── requirements.txt
├── Data/
│   └── creditcard.csv              # Dataset (no incluido)
├── Graficos/                       # Tablas y figuras generadas
└── Modulos/
    ├── config.py                   # Semilla, rutas, tamaños, CV (DRY)
    ├── data_loader.py              # Carga + auditoría + submuestreo
    ├── data_splitter.py            # Split estratificado 70/30 congelado
    ├── feature_selector.py         # SelectKBest + información mutua
    ├── imbalance_handler.py        # RandomUnderSampler / SMOTE
    ├── pipeline_builder.py         # Pipeline anti-data-leakage
    ├── experiments.py              # Matriz de 8 configuraciones
    ├── knn.py                      # Fábrica k-NN
    ├── arbol.py                    # Fábrica Árbol de Decisión
    ├── svm.py                      # Fábrica SVM lineal
    ├── randomForest.py             # CustomRandomForest (Bagging + subespacio)
    └── evaluation/
        ├── cross_validator.py      # CV=5 + búsqueda de hiperparámetros
        ├── metrics_calculator.py   # Recall, especificidad, F1, FNR, ...
        └── reporter.py             # CSV/Markdown + matrices + gráficos
```

## Instalación

```bash
python -m venv venv_ia_p2
source venv_ia_p2/bin/activate
pip install -r requirements.txt
```

Coloca `creditcard.csv` en `Data/`.

## Uso

```bash
# Dataset completo (puede tardar horas según el hardware)
python main.py

# Prueba rápida: submuestra de 15.000 registros y rejillas reducidas
python main.py --quick

# Submuestra personalizada (mínimo 10.000)
python main.py --sample-size 20000

# Ejecutar solo algunos experimentos
python main.py --quick --experiments EXP_1,EXP_2

# Controlar el paralelismo
python main.py --quick --jobs 4 --verbose 0

# Dataset completo con selección rápida de características (recomendado)
python main.py --feature-score f_classif

# Usar SMOTE en lugar de submuestreo
python main.py --quick --resampler smote
```

`--feature-score mutual_info` es el valor del SDD, pero en el dataset completo
cuesta ~15-25 s por fold (y se ejecuta en cada fold de cada candidato de la
rejilla). `f_classif` es prácticamente instantáneo y permite completar las 8
experiencias en un tiempo razonable.

## Anti-data-leakage

El pipeline usa `imblearn.pipeline.Pipeline`, cuyo remuestreador solo actúa
durante `fit`. En cada fold de la validación cruzada:

1. El escalador y `SelectKBest` se ajustan **solo** con el fold de
   entrenamiento.
2. El remuestreo se aplica **solo** a ese fold de entrenamiento.
3. El fold de validación no se transforma.
4. El 30% de prueba se reserva antes de cualquier ajuste y solo se usa en la
   evaluación final.

`GridSearchCV`/`RandomizedSearchCV` reentrena la mejor configuración sobre todo
el 70% de entrenamiento (`refit`), y esa configuración se evalúa una única vez
sobre el 30% de prueba.

## Matriz experimental

| ID | Clasificador | Compensación | Escalado |
| :--- | :--- | :--- | :--- |
| EXP_1 | k-NN | No | Sí |
| EXP_2 | k-NN | Sí (`under` por defecto) | Sí |
| EXP_3 | Árbol de Decisión | No | No |
| EXP_4 | Árbol de Decisión | Sí | No |
| EXP_5 | SVM lineal | No | Sí |
| EXP_6 | SVM lineal | Sí | Sí |
| EXP_7 | Custom Random Forest | No | No |
| EXP_8 | Custom Random Forest | Sí | No |

La compensación por defecto es `RandomUnderSampler` para mantener el coste
computacional bajo. Puede cambiarse a `SMOTE` en `Modulos/config.py`
(`DEFAULT_RESAMPLER = "smote"`).

## Salidas

En `Graficos/`:

- `resultados_experimentos.csv` y `.md`: tabla comparativa final.
- `resultados_validacion_cruzada.csv`: todas las combinaciones evaluadas.
- `matrices_confusion.png`: figura combinada.
- `matrices/matriz_confusion_EXP_*.png`: una figura por experimento.
- `comparacion_metricas.png`: recall, precisión, especificidad y F1.
- `recall_validacion_cruzada.png`: recall medio en CV con desviación estándar.

## Notas de implementación

- **SVM lineal**: se usa `LinearSVC` porque un SVM con kernel RBF sobre
  ~200.000 muestras tendría coste cuadrático. `LinearSVC` es un SVM válido y
  escalable.
- **Random Forest propio**: `CustomRandomForest` implementa bootstrap y
  subespacio aleatorio (`max_features`) con `DecisionTreeClassifier`, sin usar
  `RandomForestClassifier` ni `BaggingClassifier`. Incluye `predict_proba` y
  desempate explícito a favor de la clase positiva.
- **Desbalanceo severo**: con ~0,17% de fraudes, la exactitud es engañosa. Por
  eso el reporte prioriza recall, FNR, precisión y F1.
- **Recall como criterio único**: al maximizar recall sin penalizar falsos
  positivos, un modelo que predice casi todo como fraude puede ganar. La
  precisión solo desempata; revisa la tabla completa antes de elegir el modelo
  final.
