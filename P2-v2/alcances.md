# Software Design Document (SDD)
## Proyecto: Detección de Fraude en Transacciones Financieras mediante Aprendizaje Supervisado

## 1. Visión General del Sistema y Objetivos Técnicos

El propósito de este sistema es evaluar y comparar cuatro clasificadores de aprendizaje supervisado (k-NN, Árbol de Decisión, SVM y una implementación propia de Random Forest) para la detección de fraudes con tarjetas de crédito. 

### Requerimientos Clave:
1. **Prevención estricta de Data Leakage:** El escalado, la selección de características y el remuestreo (compensación de desbalance) deben aplicarse exclusivamente en el conjunto de entrenamiento de cada partición de validación cruzada ($k=5$).
2. **Criterio de Selección:** Maximizar la **Sensibilidad** ($\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$) de la clase positiva (Fraude = 1), equivalente a minimizar el *False Negative Rate* ($\text{FNR}$). Como criterio secundario de desempate se utilizará la **Precisión**.
3. **Random Forest Propietario:** Implementación propia de Bagging + Subespacio Aleatorio (*Random Subspace*) utilizando `DecisionTreeClassifier` de `scikit-learn` como modelo base (sin usar `RandomForestClassifier` ni `BaggingClassifier`).
4. **SOLID/DRY:** Pipelines de preprocesamiento, implementación de algoritmos, validación/búsqueda de hiperparámetros y evaluación/reporte.

---

## 2. Entorno de Desarrollo y Configuración (`venv`)

### 2.1. Configuración del Entorno Virtual
```bash
# Creación del entorno virtual
python -m venv venv_ia_p2

source venv/bin/activate

# guardar los paquetes dentro de requirements.txt
```
---

## 3. Arquitectura y Estructura Modular del Proyecto

Seguir la definida en la raiz del Proyecto

---

## 4. Diseño Modular Detallado y Aplicación de Principios SOLID & DRY

### 4.1. Módulo Configuración
 centraliza constantes y parámetros globales para cumplir con el principio **DRY**.

*   `SEED`: 42 (Garantiza reproducibilidad global).
*   `TEST_SIZE`: 0.3.
*   `N_SPLITS`: 5 (Validación Cruzada Estratificada).
*   `SAMPLE_SIZE`: Opcional para pruebas rápidas (mínimo 10,000 registros según requerimiento).

### 4.2. Módulo de Datos
*   `data_loader.py`: Lee el CSV y realiza auditoría inicial (duplicados, valores nulos, reporte de desbalance).
*   `data_splitter.py`: Aplica `train_test_split(..., stratify=y, random_state=42)`. Devuelve $X_{train}, X_{test}, y_{train}, y_{test}$. El conjunto de prueba se reserva sin tocar.

### 4.3. Implementación Propietaria de Random Forest
Cumple con la especificación de **Open/Closed Principle** (hereda de `BaseEstimator` y `ClassifierMixin` de scikit-learn para integrarse con pipelines nativos).

```python
# Arquitectura conceptual de CustomRandomForest
class CustomRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.tree_features = []  # Almacena qué características usó cada árbol si aplica

    def fit(self, X, y):
        # 1. Derivar semillas independientes por cada árbol a partir de random_state
        # 2. Generar muestras bootstrap con reemplazo (tamaño N = len(X))
        # 3. Para cada muestra bootstrap:
        #    a. Instanciar DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features, random_state=tree_seed)
        #    b. Entrenar el árbol individual
        # 4. Guardar árboles en self.trees
        return self

    def predict(self, X):
        # 1. Obtener predicciones de todos los árboles (matriz N_samples x N_trees)
        # 2. Aplicar Votación Mayoritaria (Majority Voting)
        # 3. Resolver empates si N_trees es par mediante regla explícita (p. ej. priorizar clase positiva)
        return final_predictions
```

### 4.4. Preprocesamiento, Selección y Resampling
Para cumplir el principio de **Single Responsibility**, cada tarea es un componente independiente:
*   `feature_selector.py`: Envuelve transformadores como `SelectKBest(score_func=mutual_info_classif)`.
*   `imbalance_handler.py`: Implementa o configura técnicas como `RandomUnderSampler` o `SMOTE`.
*   `pipeline_builder.py`: Utiliza `imblearn.pipeline.Pipeline` para encadenar:
    1. Scaler (p. ej., `StandardScaler` o `RobustScaler`).
    2. Feature Selection (p. ej., Mutual Information).
    3. Resampling (Opcional según experimento: SMOTE / Undersampling).
    4. Estimador ($k$-NN, Árbol, SVM o CustomRandomForest).

> **Aislamiento Total (Anti-Data Leakage):** Al estar en un `ImbalancePipeline`, las transformaciones (fit_transform) se ejecutan **únicamente** en las particiones de entrenamiento dentro de la Validación Cruzada ($k=5$). El conjunto de validación interna y el de prueba ($30\%$) **jamás** se someten a resampling.

### 4.5. Validación y Optimización (`src/evaluation/cross_validator.py`)
*   Ejecuta `GridSearchCV` / `RandomizedSearchCV` con `StratifiedKFold(n_splits=5)`.
*   Métrica de Optimización Principal (scoring): `make_scorer(recall_score, pos_label=1)`.
*   Métrica Secundaria (desempate): `precision_score(pos_label=1)`.

### 4.6. Cálculo de Métricas y Evaluación (`src/evaluation/metrics_calculator.py`)
Función pura que recibe $(y_{true}, y_{pred})$ y retorna un diccionario estandarizado:
*   Sensibilidad ($\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$)
*   Especificidad ($\text{TNR} = \frac{\text{TN}}{\text{TN} + \text{FP}}$)
*   Precisión ($\frac{\text{TP}}{\text{TP} + \text{FP}}$)
*   F1-Score
*   Exactitud ($\text{Accuracy}$)
*   Tasa de Falsos Negativos ($\text{FNR} = 1 - \text{Sensibilidad}$)

---

## 5. Matriz Experimental (8 Configuraciones)

Se evalúan los 4 clasificadores bajo 2 escenarios obligatorios:

| ID Exp | Clasificador | Compensación Desbalance | Estrategia Pipeline |
| :--- | :--- | :--- | :--- |
| **EXP_1** | $k$-Nearest Neighbors ($k$-NN) | Sin Compensación | Scaler + Feature Selection + $k$-NN |
| **EXP_2** | $k$-Nearest Neighbors ($k$-NN) | Con Compensación (e.g., SMOTE/Under) | Scaler + Feature Selection + Resampler + $k$-NN |
| **EXP_3** | Árbol de Decisión | Sin Compensación | Feature Selection + DecisionTree |
| **EXP_4** | Árbol de Decisión | Con Compensación | Feature Selection + Resampler + DecisionTree |
| **EXP_5** | Support Vector Machine (SVM) | Sin Compensación | Scaler + Feature Selection + SVM |
| **EXP_6** | Support Vector Machine (SVM) | Con Compensación | Scaler + Feature Selection + Resampler + SVM |
| **EXP_7** | Custom Random Forest | Sin Compensación | Feature Selection + CustomRandomForest |
| **EXP_8** | Custom Random Forest | Con Compensación | Feature Selection + Resampler + CustomRandomForest |

---

## 6. Fases del Plan de Desarrollo (Roadmap)

### Fase 1: Setup y Exploración Inicial
*   Creación del entorno virtual `venv_ia_p2` e instalación de dependencias.
*   Creación de la estructura de módulos.
*   Desarrollo de `data_loader.py` y `data_splitter.py`: División estratificada 70/30 congelada con semilla fija.

### Fase 2: Implementación de Custom Random Forest & Pruebas
*   Desarrollo de `CustomRandomForest` en `Modulos/randomForest.py`.
*   Implementación del muestreo Bootstrap, subespacio aleatorio (`max_features`) y regla de votación.

### Fase 3: Pipelines Anti-Data Leakage
*   Implementación de `pipeline_builder.py` integrando escalado, selección de características y técnicas de balanceo mediante `imblearn.pipeline`.
*   Verificación de aislamientos para asegurar que el escalado y el remuestreo no toquen datos fuera de la partición de entrenamiento.

### Fase 4: Validación Cruzada e Hiperparametrización (CV=5)
*   Configuración de `GridSearchCV` / `RandomizedSearchCV` guiado por Sensibilidad de clase Fraude.
*   Ejecución de las 8 experiencias de la Matriz Experimental.
*   Registro de hiperparámetros óptimos, media de sensibilidad y desviación estándar en CV.

### Fase 5: Evaluación Final y Generación de Reportes
*   Re-entrenamiento de las 8 mejores configuraciones sobre el conjunto completo de entrenamiento (70%).
*   Evaluación sobre el conjunto de prueba congelado (30%).
*   Exportación de tablas comparativas de rendimiento y generación de matrices de confusión a través de `reporter.py` en `Graficos`.
```
