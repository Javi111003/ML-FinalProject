import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def run_kmeans(df_numericas: pd.DataFrame, k_clusters=3, features=None):
    # Usar todas las columnas numéricas si no se pasan features
    if features is None:
        features = df_numericas.columns.tolist()

    # Selección de variables
    X = df_numericas[features]

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
  
    # Entrenar KMeans con K especificado
    print(f"\n🔍 Entrenando K-Means con K = {k_clusters} clusters...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df_numericas['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

    # Crear carpeta results si no existe
    os.makedirs("results", exist_ok=True)

    # Visualización (si hay al menos 2 variables)
    if X.shape[1] >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter LINEAL
        scatter1 = axes[0].scatter(
            X.iloc[:,0], X.iloc[:,1],
            c=df_numericas['cluster_kmeans'],
            cmap='plasma', s=50, alpha=0.7, edgecolors='k', linewidth=0.5
        )
        axes[0].set_xlabel(features[0], fontsize=12, fontweight='bold')
        axes[0].set_ylabel(features[1], fontsize=12, fontweight='bold')
        axes[0].set_title("Clusters - Escala LINEAL", fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Scatter LOGARÍTMICA
        scatter2 = axes[1].scatter(
            X.iloc[:,0], X.iloc[:,1],
            c=df_numericas['cluster_kmeans'],
            cmap='plasma', s=50, alpha=0.7, edgecolors='k', linewidth=0.5
        )
        axes[1].set_xscale('log')  # ESCALA LOGARÍTMICA en X
        axes[1].set_xlabel(features[0] + " (escala logarítmica)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel(features[1], fontsize=12, fontweight='bold')
        axes[1].set_title("Clusters - Escala LOGARÍTMICA", fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Cluster')
        
        fig.tight_layout()
        plt.savefig(f"results/clusters_{k_clusters}_consumo.png", dpi=300, bbox_inches="tight")
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
    outliers = pd.read_csv("../outliers-analysis/results/datos/DATOS_outliers.csv")

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


