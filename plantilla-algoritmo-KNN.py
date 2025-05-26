import numpy as np
# from collections import Counter # Counter no es necesario para regresión (promedio)

# -----------------------------------------------------------------------------
# FUNCIONES PRINCIPALES DEL ALGORITMO KNN
# (distancia_euclidiana y encontrar_vecinos pueden permanecer iguales,
#  pero encontrar_vecinos ahora devolverá los valores numéricos de los vecinos)
# -----------------------------------------------------------------------------

def distancia_euclidiana(punto1, punto2):
    punto1 = np.array(punto1)
    punto2 = np.array(punto2)
    return np.sqrt(np.sum((punto1 - punto2)**2))

def encontrar_vecinos_regresion(X_entrenamiento, y_entrenamiento_valores, punto_prueba, k, funcion_distancia=distancia_euclidiana):
    """
    Encuentra los k vecinos más cercanos y devuelve sus VALORES numéricos (y_entrenamiento_valores).

    Args:
        X_entrenamiento (list of lists or np.array): Características del conjunto de entrenamiento.
        y_entrenamiento_valores (list or np.array): VALORES NUMÉRICOS del conjunto de entrenamiento.
        punto_prueba (list or np.array): El punto para el cual se quieren encontrar los vecinos.
        k (int): El número de vecinos a encontrar.
        funcion_distancia (function): La función a usar para calcular la distancia.

    Returns:
        list: Una lista de los VALORES numéricos de los k vecinos más cercanos.
    """
    distancias = []
    for i, punto_entrenamiento in enumerate(X_entrenamiento):
        dist = funcion_distancia(punto_entrenamiento, punto_prueba)
        # Guardamos el VALOR numérico y la distancia
        distancias.append((y_entrenamiento_valores[i], dist))

    distancias.sort(key=lambda tupla: tupla[1]) # Ordenar por distancia

    # Seleccionar los VALORES de los k vecinos más cercanos
    vecinos_valores = [distancias[i][0] for i in range(min(k, len(distancias)))]
    return vecinos_valores

def predecir_regresion_knn(X_entrenamiento, y_entrenamiento_valores, punto_prueba, k, funcion_distancia=distancia_euclidiana):
    """
    Predice el VALOR numérico de un punto de prueba usando el algoritmo KNN Regresor.

    Args:
        X_entrenamiento (list of lists or np.array): Características del conjunto de entrenamiento.
        y_entrenamiento_valores (list or np.array): VALORES NUMÉRICOS del conjunto de entrenamiento.
        punto_prueba (list or np.array): El punto cuyo valor se quiere predecir.
        k (int): El número de vecinos a considerar.
        funcion_distancia (function): La función a usar para calcular la distancia.

    Returns:
        El valor numérico predicho para el punto de prueba.
    """
    # 1. Encontrar los k vecinos más cercanos y sus valores
    valores_vecinos = encontrar_vecinos_regresion(X_entrenamiento, y_entrenamiento_valores, punto_prueba, k, funcion_distancia)

    if not valores_vecinos:
        return None # O manejar de otra forma

    # 2. Calcular el promedio de los valores de los vecinos
    #    También podrías usar np.median(valores_vecinos) para la mediana
    valor_predicho = np.mean(valores_vecinos)

    return valor_predicho

def predecir_multiples_puntos_regresion_knn(X_entrenamiento, y_entrenamiento_valores, X_prueba, k, funcion_distancia=distancia_euclidiana):
    """
    Predice los valores para un conjunto de puntos de prueba (regresión).
    """
    predicciones = []
    for punto_prueba in X_prueba:
        prediccion = predecir_regresion_knn(X_entrenamiento, y_entrenamiento_valores, punto_prueba, k, funcion_distancia)
        predicciones.append(prediccion)
    return predicciones


# --- (Las funciones de escalado normalizar_min_max y aplicar_normalizacion_min_max pueden ser las mismas) ---
def normalizar_min_max(datos):
    datos_np = np.array(datos, dtype=float)
    min_vals = np.min(datos_np, axis=0)
    max_vals = np.max(datos_np, axis=0)
    rango_vals = max_vals - min_vals
    rango_vals[rango_vals == 0] = 1
    datos_normalizados = (datos_np - min_vals) / rango_vals
    return datos_normalizados, min_vals, max_vals

def aplicar_normalizacion_min_max(datos, min_vals, max_vals):
    datos_np = np.array(datos, dtype=float)
    rango_vals = max_vals - min_vals
    rango_vals[rango_vals == 0] = 1
    return (datos_np - min_vals) / rango_vals
# -----------------------------------------------------------------------------
# PLANTILLA DE EJECUCIÓN PARA REGRESIÓN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 1. DATOS DE EJEMPLO PARA REGRESIÓN (REEMPLAZAR) ---
    # X_entrenamiento: Características.
    # y_entrenamiento_valores: VALORES NUMÉRICOS a predecir.
    # X_prueba: Características de los datos de prueba.

    # Ejemplo: (Característica única: años de experiencia) -> (Valor: salario en miles)
    X_entrenamiento_reg = np.array([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
    ])
    y_entrenamiento_valores_reg = np.array([
        30, 35, 45, 50, 60, 65, 75, 80, 90, 100 # Salarios en miles
    ])
    X_prueba_reg = np.array([
        [2.5],  # Años de experiencia para los que predecir salario
        [5.5],
        [8.8]
    ])

    # --- SELECCIONA QUÉ DATOS DE EJEMPLO USAR ---
    X_entrenamiento_activo = X_entrenamiento_reg
    y_entrenamiento_activo = y_entrenamiento_valores_reg # ¡Asegúrate de usar y_entrenamiento_valores!
    X_prueba_activo = X_prueba_reg

    # --- 2. CONFIGURACIÓN DEL ALGORITMO ---
    K_VECINOS = 3
    USAR_ESCALADO = True # Para regresión, el escalado de X sigue siendo importante.
    # El escalado de Y (la variable objetivo) es un tema aparte y
    # generalmente no se hace de la misma manera para la predicción directa.

    # --- 3. ESCALADO DE CARACTERÍSTICAS (X) ---
    if USAR_ESCALADO:
        print("Aplicando escalado Min-Max a las características X...")
        X_entrenamiento_escalado, min_vals_ent, max_vals_ent = normalizar_min_max(X_entrenamiento_activo)
        X_prueba_escalado = aplicar_normalizacion_min_max(X_prueba_activo, min_vals_ent, max_vals_ent)
    else:
        X_entrenamiento_escalado = X_entrenamiento_activo
        X_prueba_escalado = X_prueba_activo

    # --- 4. REALIZAR PREDICCIONES DE REGRESIÓN ---
    print(f"\nUsando K = {K_VECINOS} para regresión")
    predicciones_regresion = predecir_multiples_puntos_regresion_knn(
        X_entrenamiento_escalado,
        y_entrenamiento_activo, # Se pasan los valores numéricos originales de y
        X_prueba_escalado,
        K_VECINOS
    )

    # --- 5. MOSTRAR RESULTADOS ---
    print("\nResultados de las predicciones de regresión:")
    for i, punto_orig in enumerate(X_prueba_activo):
        print(f"  Para X original: {punto_orig} -> Valor predicho: {predicciones_regresion[i]:.2f}")

    # -----------------------------------------------------------------------------
    # EVALUACIÓN DEL MODELO DE REGRESIÓN (SI TIENES VALORES VERDADEROS PARA PRUEBA)
    # -----------------------------------------------------------------------------
    # Para regresión, se usan métricas como:
    # - Error Absoluto Medio (MAE: Mean Absolute Error)
    # - Error Cuadrático Medio (MSE: Mean Squared Error)
    # - Raíz del Error Cuadrático Medio (RMSE: Root Mean Squared Error)
    # - Coeficiente de Determinación (R^2 Score)
    #
    # y_prueba_valores_ejemplo = np.array([38, 62, 85]) # Ejemplo de valores verdaderos para X_prueba_reg
    #
    # if 'y_prueba_valores_ejemplo' in locals() and len(y_prueba_valores_ejemplo) == len(predicciones_regresion):
    #     mse = np.mean((np.array(predicciones_regresion) - y_prueba_valores_ejemplo)**2)
    #     rmse = np.sqrt(mse)
    #     mae = np.mean(np.abs(np.array(predicciones_regresion) - y_prueba_valores_ejemplo))
    #     print(f"\nEvaluación del modelo de regresión en datos de prueba:")
    #     print(f"  Error Cuadrático Medio (MSE): {mse:.2f}")
    #     print(f"  Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
    #     print(f"  Error Absoluto Medio (MAE): {mae:.2f}")
    # else:
    #     print("\nNo se proporcionaron valores verdaderos de prueba (y_prueba_valores) para la evaluación.")