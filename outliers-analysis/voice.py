import numpy as np
import pandas as pd
from outliers import OutlierDetection


def outliers_frequent_receivers(voice_df: pd.DataFrame):
    """
    Detecta usuarios que reciben demasiadas llamadas.
    Se agrupa por OWNER_CUST_ID y se calcula log_calls.
    """
    receivers = (
        voice_df.groupby("OWNER_CUST_ID")
                .size()
                .reset_index(name="calls")
    )
    receivers["log_calls"] = np.log1p(receivers["calls"])

    # Detectar outliers sobre calls y log_calls
    outlier_detection = OutlierDetection(receivers[["calls", "log_calls"]], "VOZ_FREQ")
    outlier_detection.run_dbscan()
    outlier_detection.describe_outliers()
    outlier_detection.save_descriptions()
    outlier_detection.save_outliers()
    outlier_detection.visualize_clusters()

    print(f"[VOICE] Usuarios que reciben demasiadas llamadas: {len(outlier_detection.outliers)}")


def detect_voice_outliers():
    """
    Detecta outliers en llamadas de voz por duración, cargo y recepción de llamadas.
    """
    df = pd.read_excel("../data/muestra.xlsx")
    voice = df[df["SERVICE_CATEGORY"] == 1]

    # 🔹 Outliers por duración (ACTUAL_USAGE)
    usage_df = voice[["ACTUAL_USAGE"]].copy()
    det_usage = OutlierDetection(usage_df, "VOZ_USAGE")
    det_usage.run_dbscan()
    det_usage.describe_outliers()
    det_usage.save_descriptions()
    det_usage.save_outliers()
    det_usage.visualize_clusters()

    # 🔹 Outliers por cargo (ACTUAL_CHARGE)
    charge_df = voice[["ACTUAL_CHARGE"]].copy()
    det_charge = OutlierDetection(charge_df, "VOZ_CHARGE")
    det_charge.run_dbscan()
    det_charge.describe_outliers()
    det_charge.save_descriptions()
    det_charge.save_outliers()
    det_charge.visualize_clusters()

    # 🔹 Outliers por recepción de llamadas
    outliers_frequent_receivers(voice)


if __name__ == "__main__":
    detect_voice_outliers()

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

# Variable: calls
#   media_normal: 1.4320987654320987
#   media_outlier: 5.0
#   min_outlier: 4.0
#   max_outlier: 8.0