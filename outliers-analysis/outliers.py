import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


class OutlierDetection:
    def __init__(self, df: pd.DataFrame, service: str, results_root="results"):
        self.preprocess_dataframe(df)
        self.service = service
        self.root = f"{results_root}/{self.service.lower()}"

    def preprocess_dataframe(self, df: pd.DataFrame):
        self.df = df.select_dtypes(include=['number'])
    
    def run_dbscan(self):
        # 1) Eliminar columnas completamente vacías
        # self.df = self.df.dropna(axis=1, how="all")

        # 2) Imputación de NaN con 0
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        X_imputed = imputer.fit_transform(self.df)

        # 3) Normalización
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # 4) min_samples dinámico
        n_dims = X_scaled.shape[1]
        min_samples = max(2 * n_dims, 5)

        # 5) k-vecinos para eps
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, _ = neighbors_fit.kneighbors(X_scaled)
        distances = np.sort(distances[:, -1])
        eps = np.percentile(distances, 90)

        # ⚠️ Asegurar que eps sea > 0
        if eps <= 0:
            eps = 0.1  # valor mínimo por defecto
            
        # 6) DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        result = self.df.copy()
        result['cluster'] = labels
        outliers = result[result['cluster'] == -1]

        print(f"Se detectaron {len(outliers)} outliers")
        self.outliers = outliers
        self.df['cluster'] = result['cluster'].values

        return result, outliers


    def describe_outliers(self, outlier_label=-1):
        """
        df: DataFrame con columnas originales + 'cluster' (resultado de DBSCAN)
        outlier_label: etiqueta usada por DBSCAN para outliers (por defecto -1)
        """

        # Separar normales y outliers
        normales = self.df[self.df['cluster'] != outlier_label]
        outliers = self.df[self.df['cluster'] == outlier_label]

        descriptions: dict[str, dict] = {}

        for col in self.df.columns:
            if col == 'cluster':
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Estadísticas comparativas
                normales_stats = normales[col].describe()
                outliers_stats = outliers[col].describe()

                descriptions[col] = {
                    "media_normal": normales_stats['mean'],
                    "media_outlier": outliers_stats['mean'],
                    "min_outlier": outliers_stats['min'],
                    "max_outlier": outliers_stats['max']
                }
        self.descriptions = descriptions
        return descriptions

    def save_descriptions(self):
        """
        Guarda las descripciones de outliers en un archivo TXT.
        """
        
        filename = f"{self.root}/{self.service}_outlier_descriptions.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for var, info in self.descriptions.items():
                f.write(f"Variable: {var}\n")
                for k, v in info.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

    def save_outliers(self):
        """
        Guarda los outliers en un archivo .csv y .xlsx.
        """
        self.outliers.to_csv(f"{self.root}/{self.service}_outliers.csv", index=False, encoding="utf-8")
        self.outliers.to_excel(f"{self.root}/{self.service}_outliers.xlsx", index=False, engine="openpyxl")

    def visualize_clusters(self):
        """
        Visualiza clusters:
        - Si hay más de una columna numérica: reduce con PCA a 2D.
        - Si solo hay una columna: grafica esa columna vs índice.
        """
        if 'cluster' not in self.df.columns:
            raise ValueError("Primero debes ejecutar run_dbscan() para obtener los clusters.")

        features = self.df.drop(columns=['cluster'])
        n_features = features.shape[1]

        if n_features > 1:
            # PCA a 2 componentes
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(features.values)

            reduced_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            reduced_df['cluster'] = self.df['cluster'].values

            normales = reduced_df[reduced_df['cluster'] != -1]
            outliers = reduced_df[reduced_df['cluster'] == -1]

            plt.figure(figsize=(10, 6))
            plt.scatter(normales['PC1'], normales['PC2'], c="skyblue", s=30, label="Normales", alpha=0.7)
            plt.scatter(outliers['PC1'], outliers['PC2'], c="red", s=40, label="Outliers", alpha=0.9)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"Clusters en espacio PCA 2D ({self.service})")
            plt.legend()
            plt.tight_layout()

        else:
            # Solo una columna → gráfico 1D
            col = features.columns[0]
            normales = self.df[self.df['cluster'] != -1]
            outliers = self.df[self.df['cluster'] == -1]

            plt.figure(figsize=(10, 6))
            plt.scatter(normales.index, normales[col], c="skyblue", s=30, label="Normales", alpha=0.7)
            plt.scatter(outliers.index, outliers[col], c="red", s=40, label="Outliers", alpha=0.9)
            plt.xlabel("Índice")
            plt.ylabel(col)
            plt.title(f"{col} con outliers ({self.service})")
            plt.legend()
            plt.tight_layout()

        # Guardar la figura
        filename = f"{self.root}/{self.service}_clusters.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.show()
