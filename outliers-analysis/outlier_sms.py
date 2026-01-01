import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def run_dbscan_dynamic(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    n_dims = X_scaled.shape[1]
    min_samples = max(2 * n_dims, 5)

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    distances, _ = neighbors.fit(X_scaled).kneighbors(X_scaled)

    distances = np.sort(distances[:, -1])
    eps = np.percentile(distances, 90)

    dbscan = DBSCAN(eps=max(eps, 0.1), min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    result = df.copy()
    result["cluster"] = labels
    outliers = result[result["cluster"] == -1]

    print(f"[VOICE] Outliers detectados: {len(outliers)}")
    return result, outliers


def describe_outliers(df):
    normales = df[df["cluster"] != -1]
    outliers = df[df["cluster"] == -1]

    desc = {}

    for col in df.columns:
        if col == "cluster":
            continue

        # 👇 SOLO columnas numéricas
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        desc[col] = {
            "mean_normal": normales[col].mean(),
            "mean_outlier": outliers[col].mean(),
            "min_outlier": outliers[col].min(),
            "max_outlier": outliers[col].max(),
        }

    return desc


def save_descriptions(desc, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for k, v in desc.items():
            f.write(f"{k}\n")
            for kk, vv in v.items():
                f.write(f"  {kk}: {vv}\n")
            f.write("\n")


def save_outliers(df, prefix):
    df.to_csv(f"{prefix}.csv", index=False)
    df.to_excel(f"{prefix}.xlsx", index=False)

def outliers_frequent_receivers(voice_df):

    receivers = (
        voice_df.groupby("OTHER_NUMBER")
                .size()
                .reset_index(name="received_sms")
    )

    # Transformación logarítmica
    receivers["log_received_sms"] = np.log1p(receivers["received_sms"])

    result, outliers = run_dbscan_dynamic(
        receivers[["log_received_sms"]]
    )

    receivers["cluster"] = result["cluster"].values

    desc = describe_outliers(receivers)
    save_descriptions(desc, "outliers_frequent_receivers.txt")
    save_outliers(
        receivers[receivers["cluster"] == -1],
        "outliers_frequent_receivers"
    )

    print(f"[VOICE] Usuarios que reciben demasiadas llamadas: {len(outliers)}")


if __name__ == "__main__":

    df = pd.read_csv("/home/miguel/Escritorio/Escuela/ML-FinalProject/data/muestra.csv")

    voice = df[df["SERVICE_CATEGORY"] == 2]

    # 🔹 Outliers por duración (YA FUNCIONA)
    duration_df = voice[["ACTUAL_USAGE"]]
    result, outliers = run_dbscan_dynamic(duration_df)
    save_descriptions(describe_outliers(result), "outliers_message.txt")
    save_outliers(outliers, "outliers_message")

    # 🔹 Outliers por recepción de llamadas
    outliers_frequent_receivers(voice)

# Para este outlier se escogio la cantidad de mensajes recibidos solamente. Es analisis que se hizo
# utilizando la columna OTHER_NUMBER

# received_sms
#   mean_normal: 2.5391304347826087
#   mean_outlier: 25.96153846153846
#   min_outlier: 7
#   max_outlier: 168

# log_received_sms
#   mean_normal: 1.0865505165839957
#   mean_outlier: 2.8561597273585693
#   min_outlier: 2.0794415416798357
#   max_outlier: 5.1298987149230735

