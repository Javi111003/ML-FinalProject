import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def run_dbscan_dynamic(pca_df: pd.DataFrame):
    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_df)

    # Número de dimensiones
    n_dims = X_scaled.shape[1]

    # min_samples dinámico
    min_samples = max(2 * n_dims, 5)  # al menos 5

    # Calcular distancias k-vecinos
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    # Usar la distancia al último vecino
    distances = np.sort(distances[:, -1])

    # Heurística: percentil 90 como eps
    eps = np.percentile(distances, 90)

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Añadir etiquetas
    result = pca_df.copy()
    result['cluster'] = labels

    # Outliers
    outliers = result[result['cluster'] == -1]
    
    print(f"Se detectaron {len(outliers)} outliers")

    # Visualización si hay al menos 2 PCs
    if n_dims >= 2:
        plt.scatter(result.iloc[:,0], result.iloc[:,1], c=labels, cmap='plasma', s=30)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Clusters y outliers con DBSCAN")
        plt.savefig("clusters_outliers_PC1_PC2.png", dpi=300, bbox_inches="tight")
        plt.show()

    return result, outliers

def describe_outliers(df, outlier_label=-1):
    """
    df: DataFrame con columnas originales + 'cluster' (resultado de DBSCAN)
    outlier_label: etiqueta usada por DBSCAN para outliers (por defecto -1)
    """

    # Separar normales y outliers
    normales = df[df['cluster'] != outlier_label]
    outliers = df[df['cluster'] == outlier_label]

    descripcion = {}

    for col in df.columns:
        if col == 'cluster':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Estadísticas comparativas
            normales_stats = normales[col].describe()
            outliers_stats = outliers[col].describe()

            descripcion[col] = {
                "media_normal": normales_stats['mean'],
                "media_outlier": outliers_stats['mean'],
                "min_outlier": outliers_stats['min'],
                "max_outlier": outliers_stats['max']
            }

    return descripcion

def save_descriptions(descriptions, filename="outlier_descriptions.txt"):
    """
    Guarda las descripciones de outliers en un archivo TXT.
    
    descriptions: diccionario generado por describir_outliers()
    filename: nombre del archivo de salida
    """
    with open(filename, "w", encoding="utf-8") as f:
        for var, info in descriptions.items():
            f.write(f"Variable: {var}\n")
            for k, v in info.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

if __name__ == "__main__":
    pca_df = pd.read_csv("../correlation-study/dataset_reducido_pca.csv")
    result, outliers = run_dbscan_dynamic(pca_df)
    pca_df['cluster'] = result['cluster'].values # DataFrame con variables originales + columna 'cluster'
    descriptions = describe_outliers(pca_df)
    save_descriptions(descriptions)
