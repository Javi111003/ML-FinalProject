import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np

def run_kmeans(df_numericas: pd.DataFrame, k_clusters=3, features=None, graphic="2D") -> pd.DataFrame:
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

    # Visualización 2D
    if graphic.upper() == "2D":
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
        else:
            print("⚠️ Se necesitan al menos 2 variables para visualizar en 2D")
    
    # Visualización 3D interactiva
    elif graphic.upper() == "3D":
        if X.shape[1] >= 3:
            clusters = df_numericas['cluster_kmeans'].values
            
            # Crear colormaps numéricos para plotly
            colors_map = {i: f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' 
                         for i, c in zip(sorted(df_numericas['cluster_kmeans'].unique()), 
                                        plt.cm.plasma(np.linspace(0, 1, len(df_numericas['cluster_kmeans'].unique()))))}
            
            # Gráfico 3D LINEAL (interactivo)
            fig1 = go.Figure()
            
            for cluster_id in sorted(df_numericas['cluster_kmeans'].unique()):
                mask = clusters == cluster_id
                fig1.add_trace(go.Scatter3d(
                    x=X.iloc[mask, 0],
                    y=X.iloc[mask, 1],
                    z=X.iloc[mask, 2],
                    mode='markers',
                    marker=dict(size=5, opacity=0.7),
                    name=f'Cluster {cluster_id}',
                    text=[f"Cluster {cluster_id}"] * mask.sum(),
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'{features[0]}: %{{x:.2f}}<br>' +
                                 f'{features[1]}: %{{y:.2f}}<br>' +
                                 f'{features[2]}: %{{z:.2f}}<extra></extra>'
                ))
            
            fig1.update_layout(
                title=f"Clusters 3D - Escala LINEAL (K = {k_clusters})",
                scene=dict(
                    xaxis_title=features[0],
                    yaxis_title=features[1],
                    zaxis_title=features[2],
                ),
                width=1000,
                height=800,
                hovermode='closest'
            )
            fig1.write_html(f"results/clusters_{k_clusters}_3d_lineal.html")
            fig1.show()
            
            # Gráfico 3D LOGARÍTMICO (interactivo)
            fig2 = go.Figure()
            
            for cluster_id in sorted(df_numericas['cluster_kmeans'].unique()):
                mask = clusters == cluster_id
                fig2.add_trace(go.Scatter3d(
                    x=np.log10(X.iloc[mask, 0] + 1e-10),  # Evitar log(0)
                    y=np.log10(X.iloc[mask, 1] + 1e-10),
                    z=np.log10(X.iloc[mask, 2] + 1e-10),
                    mode='markers',
                    marker=dict(size=5, opacity=0.7),
                    name=f'Cluster {cluster_id}',
                    text=[f"Cluster {cluster_id}"] * mask.sum(),
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'{features[0]} (log): %{{x:.2f}}<br>' +
                                 f'{features[1]} (log): %{{y:.2f}}<br>' +
                                 f'{features[2]} (log): %{{z:.2f}}<extra></extra>'
                ))
            
            fig2.update_layout(
                title=f"Clusters 3D - Escala LOGARÍTMICA (K = {k_clusters})",
                scene=dict(
                    xaxis_title=features[0] + " (log10)",
                    yaxis_title=features[1] + " (log10)",
                    zaxis_title=features[2] + " (log10)",
                ),
                width=1000,
                height=800,
                hovermode='closest'
            )
            fig2.write_html(f"results/clusters_{k_clusters}_3d_logaritmico.html")
            fig2.show()
        else:
            print("⚠️ Se necesitan al menos 3 variables para visualizar en 3D")
    else:
        print(f"⚠️ Tipo de gráfico no válido: {graphic}. Use '2D' o '3D'")


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


