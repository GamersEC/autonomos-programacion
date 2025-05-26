import numpy as np
from collections import Counter

# -----------------------------------------------------------------------------
# FUNCIONES PRINCIPALES DEL ALGORITMO KNN
# (Estas funciones generalmente no necesitarás modificarlas para diferentes datasets)
# -----------------------------------------------------------------------------

def distancia_euclidiana(punto1, punto2):
    """
    Calcula la distancia euclidiana entre dos puntos.
    Ambos puntos deben ser iterables (listas, tuplas, arrays de NumPy)
    de la misma longitud.
    """
    punto1 = np.array(punto1)
    punto2 = np.array(punto2)
    return np.sqrt(np.sum((punto1 - punto2)**2))

def encontrar_vecinos(X_entrenamiento, y_entrenamiento, punto_prueba, k, funcion_distancia=distancia_euclidiana):
    """
    Encuentra los k vecinos más cercanos a un punto de prueba
    dentro del conjunto de entrenamiento.

    Args:
        X_entrenamiento (list of lists or np.array): Características del conjunto de entrenamiento.
        y_entrenamiento (list or np.array): Etiquetas de clase del conjunto de entrenamiento.
        punto_prueba (list or np.array): El punto para el cual se quieren encontrar los vecinos.
        k (int): El número de vecinos a encontrar.
        funcion_distancia (function): La función a usar para calcular la distancia.
                                      Por defecto, es distancia_euclidiana.

    Returns:
        list: Una lista de las etiquetas de clase de los k vecinos más cercanos.
    """
    distancias = []
    for i, punto_entrenamiento in enumerate(X_entrenamiento):
        dist = funcion_distancia(punto_entrenamiento, punto_prueba)
        distancias.append((y_entrenamiento[i], dist)) # (etiqueta, distancia)

    distancias.sort(key=lambda tupla: tupla[1]) # Ordenar por distancia

    vecinos_etiquetas = [distancias[i][0] for i in range(min(k, len(distancias)))]
    return vecinos_etiquetas

def predecir_clasificacion_knn(X_entrenamiento, y_entrenamiento, punto_prueba, k, funcion_distancia=distancia_euclidiana):
    """
    Predice la clase de un punto de prueba usando el algoritmo KNN.

    Args:
        X_entrenamiento (list of lists or np.array): Características del conjunto de entrenamiento.
        y_entrenamiento (list or np.array): Etiquetas de clase del conjunto de entrenamiento.
        punto_prueba (list or np.array): El punto cuya clase se quiere predecir.
        k (int): El número de vecinos a considerar.
        funcion_distancia (function): La función a usar para calcular la distancia.

    Returns:
        La clase predicha para el punto de prueba.
    """
    etiquetas_vecinos = encontrar_vecinos(X_entrenamiento, y_entrenamiento, punto_prueba, k, funcion_distancia)
    conteo_clases = Counter(etiquetas_vecinos)
    if not conteo_clases: # Si no hay vecinos (k > len(X_entrenamiento) o X_entrenamiento vacío)
        return None # O manejar de otra forma, e.g., devolver una clase por defecto
    clase_predicha = conteo_clases.most_common(1)[0][0]
    return clase_predicha

def predecir_multiples_puntos_knn(X_entrenamiento, y_entrenamiento, X_prueba, k, funcion_distancia=distancia_euclidiana):
    """
    Predice las clases para un conjunto de puntos de prueba.
    """
    predicciones = []
    for punto_prueba in X_prueba:
        prediccion = predecir_clasificacion_knn(X_entrenamiento, y_entrenamiento, punto_prueba, k, funcion_distancia)
        predicciones.append(prediccion)
    return predicciones

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES PARA ESCALADO (OPCIONAL PERO RECOMENDADO)
# -----------------------------------------------------------------------------

def normalizar_min_max(datos):
    """
    Normaliza los datos usando la escala Min-Max (escala a [0, 1]).
    Aplica la normalización columna por columna.
    Devuelve los datos normalizados y los valores min y max por columna
    para poder aplicar la misma transformación a nuevos datos.
    """
    datos_np = np.array(datos, dtype=float) # Asegurar que es float para divisiones
    min_vals = np.min(datos_np, axis=0)
    max_vals = np.max(datos_np, axis=0)
    # Evitar división por cero si todos los valores en una columna son iguales
    rango_vals = max_vals - min_vals
    rango_vals[rango_vals == 0] = 1 # Si el rango es 0, la normalización no cambiará el valor (será 0 o no aplicable)

    datos_normalizados = (datos_np - min_vals) / rango_vals
    return datos_normalizados, min_vals, max_vals

def aplicar_normalizacion_min_max(datos, min_vals, max_vals):
    """
    Aplica una normalización Min-Max precalculada a nuevos datos.
    """
    datos_np = np.array(datos, dtype=float)
    rango_vals = max_vals - min_vals
    rango_vals[rango_vals == 0] = 1
    return (datos_np - min_vals) / rango_vals

# -----------------------------------------------------------------------------
# PLANTILLA DE EJECUCIÓN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 1. DATOS DE EJEMPLO (REEMPLAZAR CON TUS DATOS) ---
    # X_entrenamiento: Características de los datos de entrenamiento.
    #                  Debe ser una lista de listas o un array de NumPy 2D.
    #                  Cada sublista/fila es una instancia, cada elemento es una característica.
    # y_entrenamiento: Etiquetas de clase de los datos de entrenamiento.
    #                  Debe ser una lista o un array de NumPy 1D.
    #                  El i-ésimo elemento es la clase de la i-ésima instancia en X_entrenamiento.
    # X_prueba:        Características de los datos de prueba para los que se quiere predecir.
    #                  Mismo formato que X_entrenamiento.

    # Ejemplo 1: Datos simples (altura, peso -> clase A o B)
    X_entrenamiento_ej1 = np.array([
        [160, 60], [165, 65], [158, 58],  # Clase 0
        [175, 70], [180, 75], [178, 72]   # Clase 1
    ])
    y_entrenamiento_ej1 = np.array([0, 0, 0, 1, 1, 1])
    X_prueba_ej1 = np.array([
        [170, 68],
        [155, 55],
        [182, 65] # Un punto un poco más ambiguo
    ])

    # Ejemplo 2: Otro dataset (inventado - características florales -> especie)
    # Característica 1, Característica 2 -> Especie X, Especie Y, Especie Z
    # X_entrenamiento_ej2 = np.array([
    #     [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], # Especie X
    #     [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], # Especie Y
    #     [6.3, 3.3], [5.8, 2.7], [7.1, 3.0]  # Especie Z
    # ])
    # y_entrenamiento_ej2 = np.array(['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z'])
    # X_prueba_ej2 = np.array([
    #     [5.0, 3.1], # ¿Qué especie es?
    #     [6.5, 3.0]
    # ])

    # --- SELECCIONA QUÉ DATOS DE EJEMPLO USAR ---
    X_entrenamiento_activo = X_entrenamiento_ej1
    y_entrenamiento_activo = y_entrenamiento_ej1
    X_prueba_activo = X_prueba_ej1

    # --- 2. CONFIGURACIÓN DEL ALGORITMO ---
    K_VECINOS = 3
    USAR_ESCALADO = True # Cambiar a False si no se quiere usar escalado

    # --- 3. (OPCIONAL PERO RECOMENDADO) ESCALADO DE CARACTERÍSTICAS ---
    if USAR_ESCALADO:
        print("Aplicando escalado Min-Max...")
        X_entrenamiento_escalado, min_vals_ent, max_vals_ent = normalizar_min_max(X_entrenamiento_activo)
        X_prueba_escalado = aplicar_normalizacion_min_max(X_prueba_activo, min_vals_ent, max_vals_ent)
        print("Datos de entrenamiento escalados (primeras filas):\n", X_entrenamiento_escalado[:2])
        print("Datos de prueba escalados (primeras filas):\n", X_prueba_escalado[:2])
    else:
        X_entrenamiento_escalado = X_entrenamiento_activo # Usar datos originales si no se escala
        X_prueba_escalado = X_prueba_activo

    # --- 4. REALIZAR PREDICCIONES ---
    print(f"\nUsando K = {K_VECINOS}")
    predicciones_finales = predecir_multiples_puntos_knn(
        X_entrenamiento_escalado, # Usar datos escalados
        y_entrenamiento_activo,
        X_prueba_escalado,        # Usar datos escalados
        K_VECINOS
    )

    # --- 5. MOSTRAR RESULTADOS ---
    print("\nResultados de las predicciones:")
    for i, punto_orig in enumerate(X_prueba_activo): # Mostrar el punto original para referencia
        print(f"  Punto de prueba original: {punto_orig} -> Clase predicha: {predicciones_finales[i]}")

    # --- Para un solo punto de prueba (ejemplo) ---
    # punto_unico_original = np.array([168, 62])
    # if USAR_ESCALADO:
    #     punto_unico_escalado = aplicar_normalizacion_min_max(np.array([punto_unico_original]), min_vals_ent, max_vals_ent)[0]
    # else:
    #     punto_unico_escalado = punto_unico_original

    # prediccion_unica = predecir_clasificacion_knn(
    #     X_entrenamiento_escalado,
    #     y_entrenamiento_activo,
    #     punto_unico_escalado,
    #     K_VECINOS
    # )
    # print(f"\nPredicción para un punto único {punto_unico_original}: {prediccion_unica}")

    # -----------------------------------------------------------------------------
    # EVALUACIÓN DEL MODELO (SI TIENES ETIQUETAS PARA TUS DATOS DE PRUEBA)
    # -----------------------------------------------------------------------------
    # Si tienes y_prueba (las etiquetas verdaderas para X_prueba_activo), puedes calcular la precisión:
    #
    # y_prueba_ejemplo = np.array([1, 0, 1]) # Ejemplo de etiquetas verdaderas para X_prueba_ej1
    #
    # if 'y_prueba_ejemplo' in locals() and len(y_prueba_ejemplo) == len(predicciones_finales):
    #     correctas = np.sum(predicciones_finales == y_prueba_ejemplo)
    #     precision = correctas / len(y_prueba_ejemplo)
    #     print(f"\nPrecisión del modelo en datos de prueba: {precision * 100:.2f}% ({correctas}/{len(y_prueba_ejemplo)} correctas)")
    # else:
    #     print("\nNo se proporcionaron etiquetas de prueba (y_prueba) para calcular la precisión.")