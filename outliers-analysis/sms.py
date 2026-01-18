import numpy as np
import pandas as pd
from outliers import OutlierDetection


def outliers_frequent_receivers(message_df: pd.DataFrame):
    """
    Detecta usuarios que reciben demasiados SMS.
    Se agrupa por OTHER_NUMBER y se calcula log_received_sms.
    """
    receivers = (
        message_df.groupby("OTHER_NUMBER")
                  .size()
                  .reset_index(name="received_sms")
    )
    receivers["log_received_sms"] = np.log1p(receivers["received_sms"])

    # Detectar outliers sobre received_sms y log_received_sms
    outlier_detection = OutlierDetection(receivers[["received_sms", "log_received_sms"]], "SMS_FREQ")
    outlier_detection.run_dbscan()
    outlier_detection.describe_outliers()
    outlier_detection.save_descriptions()
    outlier_detection.save_outliers()
    outlier_detection.visualize_clusters()

    print(f"[SMS] Usuarios que reciben demasiados mensajes: {len(outlier_detection.outliers)}")


def detect_message_outliers():
    """
    Detecta outliers en mensajes por uso y por recepción de SMS.
    """
    df = pd.read_excel("../data/muestra.xlsx")
    messages = df[df["SERVICE_CATEGORY"] == 2]

    # 🔹 Outliers por duración/uso de mensajes (ACTUAL_USAGE)
    usage_df = messages[["ACTUAL_USAGE"]].copy()
    det_usage = OutlierDetection(usage_df, "SMS_USAGE")
    det_usage.run_dbscan()
    det_usage.describe_outliers()
    det_usage.save_descriptions()
    det_usage.save_outliers()
    det_usage.visualize_clusters()

    # 🔹 Outliers por recepción de SMS
    outliers_frequent_receivers(messages)


if __name__ == "__main__":
    detect_message_outliers()

# Para este outlier se escogio la cantidad de mensajes recibidos solamente. Es analisis que se hizo
# utilizando la columna OTHER_NUMBER

'''
Variable: received_sms
  media_normal: 2.0
  media_outlier: 22.743589743589745
  min_outlier: 7.0
  max_outlier: 168.0

Variable: log_received_sms
  media_normal: 1.0177498155565665
  media_outlier: 2.8519780094364426
  min_outlier: 2.0794415416798357
  max_outlier: 5.1298987149230735
'''