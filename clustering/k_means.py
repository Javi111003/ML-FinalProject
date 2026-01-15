import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def run_kmeans(df_numericas: pd.DataFrame, features=None):
    # Usar todas las columnas numéricas si no se pasan features
    if features is None:
        features = df_numericas.columns.tolist()

    # Selección de variables
    X = df_numericas[features]

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Probar distintos k
    # sil_scores = []
    # for k in range(2, 8):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     labels = kmeans.fit_predict(X_scaled)
    #     sil = silhouette_score(X_scaled, labels)
    #     sil_scores.append((k, sil))

    # # Elegir mejor k
    # best_k = max(sil_scores, key=lambda x: x[1])[0]
    # print("Mejor número de clusters:", best_k)
    best_k = 2
    # Entrenar KMeans final
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df_numericas['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

    # Crear carpeta results si no existe
    os.makedirs("results", exist_ok=True)

    # Visualización (si hay al menos 2 variables)
    if X.shape[1] >= 2:
        # <-- cambiado: usar X crudo para el scatter
        plt.figure(figsize=(8,6))
        plt.scatter(
            X.iloc[:,0], X.iloc[:,1],
            c=df_numericas['cluster_kmeans'],
            cmap='plasma', s=30
        )
        plt.xlabel(features[0])   # total_usage, total_duration
        plt.ylabel(features[1])
        plt.title("Clusters de usuarios (sin outliers)")
        plt.ylim(bottom=0)       # <-- agregado: evita mostrar negativos en duración
        plt.xlim(left=0)         # <-- agregado: evita negativos en uso
        plt.savefig("results/clusters_consumo.png", dpi=300, bbox_inches="tight")
        plt.show()


    return df_numericas

if __name__ == "__main__":
   # 1) Cargar dataset
    pca_df = pd.read_csv("../data/synthetic_data/datos_a_completar.csv")

    # 2) Convertir fechas a datetime
    pca_df['START_DATE'] = pd.to_datetime(pca_df['START_DATE'])
    pca_df['END_DATE'] = pd.to_datetime(pca_df['END_DATE'])

    # 3) Crear columna de duración en horas
    pca_df['DURATION_HOURS'] = (pca_df['END_DATE'] - pca_df['START_DATE']).dt.total_seconds() / 3600

    # 4) Cargar outliers detectados por DBSCAN
    outliers = pd.read_csv("../outliers-analysis/outliers.csv")

    # 5) Quitar outliers del dataset
    df_no_outliers = pca_df[~pca_df.index.isin(outliers.index)].copy()

    # 6) Ejecutar KMeans con consumo y duración
    result = run_kmeans(df_no_outliers, features=['ACTUAL_USAGE', 'DURATION_HOURS'])

    # 7) Estadísticas por cluster
    with open("results/cluster_report.txt", "w", encoding="utf-8") as f:
        for cid in sorted(result['cluster_kmeans'].unique()):
            grupo = result[result['cluster_kmeans'] == cid]
            f.write(f"Cluster {cid} — usuarios: {len(grupo)}\n")
            f.write(grupo[['ACTUAL_USAGE','DURATION_HOURS']].describe().to_string())
            f.write("\n\n")

    # 8) Boxplot para ver distribución de consumo por cluster
    sns.boxplot(x='cluster_kmeans', y='ACTUAL_USAGE', data=result)
    plt.title("Distribución de consumo por cluster")
    plt.savefig("results/boxplot_consumo.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 9) Boxplot para ver distribución de duración por cluster
    sns.boxplot(x='cluster_kmeans', y='DURATION_HOURS', data=result)
    plt.title("Distribución de duración por cluster")
    plt.savefig("results/boxplot_duracion.png", dpi=300, bbox_inches="tight")
    plt.close()


