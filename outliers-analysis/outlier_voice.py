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
        voice_df.groupby("OWNER_CUST_ID")
                .size()
                .reset_index(name="calls")
    )

    # Transformación logarítmica
    receivers["log_calls"] = np.log1p(receivers["calls"])

    result, outliers = run_dbscan_dynamic(
        receivers[["log_calls"]]
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

    voice = df[df["SERVICE_CATEGORY"] == 1]

    # 🔹 Outliers por duración (YA FUNCIONA)
    duration_df = voice[["ACTUAL_USAGE"]]
    result, outliers = run_dbscan_dynamic(duration_df)
    save_descriptions(describe_outliers(result), "outliers_voice_duration.txt")
    save_outliers(outliers, "outliers_voice_duration")

    # 🔹 Outliers por recepción de llamadas
    outliers_frequent_receivers(voice)


# El outlier de llamada que se utilizo fue ver que llamadas eran mas "largas" que la media
# Usando la columna "ACTUAL_USAGE" asumiendo que cada unidad de ACTUAL_USAGE es un segundo.
# Ademas se analizo el ACTUAL_CHARGE, el cual presenta una irregularidad abismal, con la mayoria
# de valores en 0 y alguno que otro superior incluso a 1000. Por otro lado se analizo la 
# cantidad de llamadas recibidas por los usuarios.
# El resultado muestra lo siguiente: 

# ACTUAL_USAGE
#   mean_normal: 36.52054794520548
#   mean_outlier: 310.5
#   min_outlier: 97.0
#   max_outlier: 831.0

# ACTUAL_CHARGE
#   mean_normal: 0.0
#   mean_outlier: 18888.88888888889
#   min_outlier: 2500.0
#   max_outlier: 80500.0

# received_calls
#   mean_normal: 1.1967213114754098
#   mean_outlier: 5.0
#   min_outlier: 4
#   max_outlier: 6