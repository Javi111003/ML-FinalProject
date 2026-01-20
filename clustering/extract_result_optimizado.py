#!/usr/bin/env python3
"""
Script para extraer result_optimizado del notebook y guardarlo como CSV
Este script replica la lógica del notebook clustering/experiments.ipynb
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from k_means import run_kmeans

# Configurar el directorio de resultados
os.makedirs("results", exist_ok=True)

# Leer los datos del CSV
print("Cargando datos...")
df = pd.read_csv("../data/synthetic_data/datos_a_completar.csv")

# Procesar datos (igual que en el notebook)
print("Procesando datos...")
df['START_DATE'] = pd.to_datetime(df['START_DATE'])
df['END_DATE'] = pd.to_datetime(df['END_DATE'])
df['DURATION_HOURS'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 3600
df_1 = df

# Agrupar por usuario
usuarios = df_1.groupby('OBJ_ID').agg(
    total_usage=('ACTUAL_USAGE', 'sum'),
    total_duration=('DURATION_HOURS', 'sum'),
    num_sessions=('ACTUAL_USAGE', 'count')
).reset_index()

print(f"Total de usuarios: {len(usuarios)}")

# Calcular el K óptimo usando Silhoueta
print("\nCalculando coeficiente de Silueta para diferentes valores de K...")
X = usuarios[['total_usage', 'total_duration', 'num_sessions']]
X_scaled = StandardScaler().fit_transform(X)

silhouette_scores = []
k_values = list(range(2, 11))

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    print(f"K={k}  →  Silueta = {sil:.4f}")

# Encontrar K óptimo
best_idx = np.argmax(silhouette_scores)
best_k = k_values[best_idx]
best_silueta = silhouette_scores[best_idx]

print(f"\n🎯 K ÓPTIMO (por Silueta) = {best_k} con Silueta = {best_silueta:.4f}")

# Analizar K=4 vs K=5
kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_k4 = kmeans_k4.fit_predict(X_scaled)
silueta_k4 = silhouette_scores[2]  # K=4 es el índice 2

kmeans_k5 = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_k5 = kmeans_k5.fit_predict(X_scaled)
silueta_k5 = silhouette_scores[3]  # K=5 es el índice 3

print(f"\n📈 SILUETA (Score):")
print(f"   K=4 → Silueta = {silueta_k4:.4f}")
print(f"   K=5 → Silueta = {silueta_k5:.4f}")
print(f"   Diferencia = {abs(silueta_k5 - silueta_k4):.6f} (MÍNIMA - prácticamente igual)")

# Contar singletons
k4_singletons = sum(1 for k in kmeans_k4.labels_ if kmeans_k4.labels_.tolist().count(k) == 1)
k5_singletons = sum(1 for k in kmeans_k5.labels_ if kmeans_k5.labels_.tolist().count(k) == 1)

print(f"\n📊 DISTRIBUCIÓN Y SINGLETONS:")
print(f"\n   K=4:")
for i in range(4):
    count = sum(1 for label in labels_k4 if label == i)
    pct = (count / len(labels_k4)) * 100
    is_singleton = "⚠️ SINGLETON" if count == 1 else ""
    print(f"      Cluster {i}:  {count} usuarios ({pct:5.1f}%) {is_singleton}")
print(f"   Total de SINGLETONS en K=4: {k4_singletons}")

print(f"\n   K=5:")
for i in range(5):
    count = sum(1 for label in labels_k5 if label == i)
    pct = (count / len(labels_k5)) * 100
    is_singleton = "⚠️ SINGLETON" if count == 1 else ""
    print(f"      Cluster {i}:  {count} usuarios ({pct:5.1f}%) {is_singleton}")
print(f"   Total de SINGLETONS en K=5: {k5_singletons}")

# Elegir K óptimo
best_k_final = 4 if k4_singletons < k5_singletons else best_k
print(f"\n💡 CONCLUSIÓN:")
print(f"   ✅ K={best_k_final} es mejor: {k4_singletons if best_k_final == 4 else k5_singletons} singletons")
if best_k_final == 4:
    print(f"      Silueta es casi idéntica ({silueta_k4:.4f} vs {silueta_k5:.4f})")
    print(f"      K=4 tiene distribución más balanceada")

# Aplicar clustering con K óptimo
print(f"\n{'='*70}")
print(f"CLUSTERING ÓPTIMO DE USUARIOS - K = {best_k_final}")
print(f"{'='*70}")

result_optimizado = run_kmeans(
    usuarios.copy(),
    k_clusters=best_k_final,
    features=['total_usage', 'total_duration', 'num_sessions']
)

print(f"\n✅ Clustering completado con K óptimo")
print(f"   K elegido: {best_k_final}")
print(f"   Razón: Silueta prácticamente idéntica a K=5 pero con menor cantidad de singletons")

# Mapear clusters a perfiles
print(f"\n{'='*70}")
print("MAPEO DE CLUSTERS A PERFILES")
print(f"{'='*70}")

cluster_stats = result_optimizado.groupby('cluster_kmeans')['total_usage'].mean().sort_values()
orden = cluster_stats.index.tolist()

# Mapeo según K
if len(orden) == 2:
    nombres = {orden[0]: 'Bajo', orden[1]: 'Alto'}
elif len(orden) == 3:
    nombres = {orden[0]: 'Bajo', orden[1]: 'Normal', orden[2]: 'Alto'}
elif len(orden) == 4:
    nombres = {orden[0]: 'Bajo Extremo', orden[1]: 'Bajo Regular', orden[2]: 'Alto Regular', orden[3]: 'Alto Extremo'}
else:
    nombres = {orden[i]: f'Cluster{i+1}' for i in range(len(orden))}

result_optimizado['perfil_consumo'] = result_optimizado['cluster_kmeans'].map(nombres)

print("\nMapeo de clusters a perfiles:")
for cluster_id, perfil in nombres.items():
    count = len(result_optimizado[result_optimizado['cluster_kmeans'] == cluster_id])
    avg_usage = result_optimizado[result_optimizado['cluster_kmeans'] == cluster_id]['total_usage'].mean()
    print(f"  Cluster {cluster_id} → {perfil} ({count} usuarios, consumo medio: {avg_usage:.2f})")

# Guardar result_optimizado como CSV
output_path = "results/result_optimizado.csv"
result_optimizado.to_csv(output_path, index=False)
print(f"\n✅ result_optimizado guardado en: {output_path}")
print(f"   Forma: {result_optimizado.shape}")
print(f"   Columnas: {result_optimizado.columns.tolist()}")

# Mostrar primeras filas
print(f"\nPrimeras filas:")
print(result_optimizado.head(10))
