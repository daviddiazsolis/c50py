# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:16:16 2025

@author: david
"""

import pandas as pd
import numpy as np
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os # Necesario para os.path.dirname

# Importar los clasificadores de c5py
from c50py import C5Classifier

# Importar clasificadores de scikit-learn para comparación
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- 1. Cargar y Preprocesar los Datos ---
print("Cargando y preprocesando los datos de Titanic...")
try:
    df = pd.read_csv("titanic.csv")
except FileNotFoundError:
    try:
        print("titanic.csv no encontrado en el directorio actual. Intentando desde la ruta de ejemplos de c5py...")
        # Intentar cargar desde la ruta de ejemplos si c5py está instalado
        c5py_root = os.path.dirname(os.path.abspath(c5py.__file__))
        examples_path = os.path.join(c5py_root, '..', 'examples', 'titanic.csv')
        df = pd.read_csv(examples_path)
        print(f"titanic.csv cargado desde: {examples_path}")
    except Exception as e:
        print(f"Error al intentar cargar titanic.csv: {e}")
        print("Asegúrate de que 'titanic.csv' está en el mismo directorio que el script, o proporciona la ruta completa.")
        exit()

# Features y Target
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

X = df[features]
y = df[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Preprocesamiento específico para c5py ---
X_train_c5py = X_train.copy()
X_test_c5py = X_test.copy()

X_train_c5py["pclass"] = X_train_c5py["pclass"].astype(str)
X_test_c5py["pclass"] = X_test_c5py["pclass"].astype(str)

X_train_c5py["sex"] = X_train_c5py["sex"].astype(str)
X_test_c5py["sex"] = X_test_c5py["sex"].astype(str)

X_train_c5py["embarked"] = X_train_c5py["embarked"].astype(str)
X_test_c5py["embarked"] = X_test_c5py["embarked"].astype(str)

X_train_c5py_values = X_train_c5py.values.astype(object)
X_test_c5py_values = X_test_c5py.values.astype(object)

class_names_list = ["Perecido", "Sobrevivió"] # Nombres de las clases para c5py

# --- Preprocesamiento específico para scikit-learn ---
categorical_features_skl = ["pclass", "sex", "embarked"]
numerical_features_skl = ["age", "sibsp", "parch", "fare"]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_skl = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_skl),
        ('cat', categorical_transformer, categorical_features_skl)
    ])

X_train_skl = preprocessor_skl.fit_transform(X_train)
X_test_skl = preprocessor_skl.transform(X_test)


# --- 2. Configuraciones de Hiperparámetros para C5Classifier ---
c5py_configs = {
    "C5Classifier_Default_Adjusted": {
        "trials": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "pruning": True,
        "cf": 0.25,
        "global_pruning": True,
        "random_state": 42,
        "feature_names": list(features), # Pasar feature_names como una lista nueva
        "categorical_features": ["pclass", "sex", "embarked"],
        "infer_categorical": False,
        "int_as_categorical": False,
        "max_depth": 20
    },
    "C5Classifier_Boosted": {
        "trials": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "pruning": True,
        "cf": 0.5,
        "global_pruning": True,
        "random_state": 42,
        "feature_names": list(features),
        "categorical_features": ["pclass", "sex", "embarked"],
        "infer_categorical": False,
        "int_as_categorical": False
    },
    "C5Classifier_NoPruning_Adjusted": {
        "trials": 1,
        "min_samples_split": 60,
        "min_samples_leaf": 20,
        "pruning": False,
        "random_state": 42,
        "feature_names": list(features),
        "categorical_features": ["pclass", "sex", "embarked"],
        "infer_categorical": False,
        "int_as_categorical": False,
        "max_depth": None
    },
    "C5Classifier_CustomThresholds": {
        "trials": 1,
        "min_samples_split": 60,
        "min_samples_leaf": 20,
        "pruning": True,
        "cf": 0.25,
        "global_pruning": True,
        "random_state": 42,
        "feature_names": list(features),
        "categorical_features": ["pclass", "sex", "embarked"],
        "infer_categorical": False,
        "int_as_categorical": False,
        "numeric_threshold_strategy": "quantile",
        "max_numeric_thresholds": 32
    }
}

# Diccionario para almacenar los resultados comparativos
results_comparison = {}

# --- 3. Función de Evaluación (modificada para almacenar todas las métricas) ---
def evaluate_classifier(name, classifier, X_train_data, y_train_data, X_test_data, y_test_data, results_dict):
    print(f"\n--- Evaluando: {name} ---")
    start_time = perf_counter()
    classifier.fit(X_train_data, y_train_data)
    fit_time = perf_counter() - start_time
    print(f"Tiempo de entrenamiento: {fit_time:.3f} segundos")

    # Predicciones en el conjunto de entrenamiento
    y_train_pred = classifier.predict(X_train_data)
    train_accuracy = accuracy_score(y_train_data, y_train_pred)
    train_report_dict = classification_report(y_train_data, y_train_pred, target_names=class_names_list, output_dict=True)
    
    print("\n--- Informe de Clasificación (Entrenamiento) ---")
    print(classification_report(y_train_data, y_train_pred, target_names=class_names_list))
    print(f"Precisión de Entrenamiento: {train_accuracy:.4f}")

    # Predicciones en el conjunto de prueba
    y_test_pred = classifier.predict(X_test_data)
    test_accuracy = accuracy_score(y_test_data, y_test_pred)
    test_report_dict = classification_report(y_test_data, y_test_pred, target_names=class_names_list, output_dict=True)

    print("\n--- Informe de Clasificación (Prueba) ---")
    print(classification_report(y_test_data, y_test_pred, target_names=class_names_list))
    print(f"Precisión de Prueba: {test_accuracy:.4f}")

    # Almacenar resultados detallados
    results_dict[name] = {
        "Tiempo Entrenamiento (s)": fit_time,
        "Precisión Entrenamiento": train_accuracy,
        "Recall Entrenamiento (weighted avg)": train_report_dict['weighted avg']['recall'],
        "F1-Score Entrenamiento (weighted avg)": train_report_dict['weighted avg']['f1-score'],
        "Precisión Prueba": test_accuracy,
        "Recall Prueba (weighted avg)": test_report_dict['weighted avg']['recall'],
        "F1-Score Prueba (weighted avg)": test_report_dict['weighted avg']['f1-score'],
    }
    # Podrías agregar métricas por clase si lo deseas, por ejemplo:
    # f"Precision Clase {class_names_list[0]} (Test)": test_report_dict[class_names_list[0]]['precision']

# --- 4. Evaluar C5Classifier con Diferentes Configuraciones ---
print("\n--- Probando C5Classifier ---")
for config_name, params in c5py_configs.items():
    clf_c5py = C5Classifier(**params)
    evaluate_classifier(config_name, clf_c5py, X_train_c5py_values, y_train, X_test_c5py_values, y_test, results_comparison)

# --- 5. Evaluar Clasificadores de Scikit-learn para Comparación ---
print("\n--- Probando Clasificadores de Scikit-learn ---")

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
evaluate_classifier(
    "Scikit-learn_DecisionTreeClassifier",
    dt_clf,
    X_train_skl, y_train,
    X_test_skl, y_test,
    results_comparison
)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_classifier(
    "Scikit-learn_RandomForestClassifier",
    rf_clf,
    X_train_skl, y_train,
    X_test_skl, y_test,
    results_comparison
)

print("\n--- Pruebas Concluidas ---")

# --- 6. Tabla Comparativa de Rendimiento ---
print("\n--- Tabla Comparativa de Métricas Clave ---")
comparison_df = pd.DataFrame.from_dict(results_comparison, orient='index')

# Formatear las columnas numéricas para una mejor visualización
for col in comparison_df.columns:
    if "Tiempo" in col:
        comparison_df[col] = comparison_df[col].map('{:.3f}'.format)
    else: # Métricas de precisión, recall, f1-score
        comparison_df[col] = comparison_df[col].map('{:.4f}'.format)

print(comparison_df)