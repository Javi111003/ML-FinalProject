import pandas as pd
from outliers import OutlierDetection


def detect_data_outliers():
    df = pd.read_csv("../data/synthetic_data/datos_completados.csv")
    
    outlier_detection = OutlierDetection(df, "DATOS")
    outlier_detection.run_dbscan()
    outlier_detection.describe_outliers()
    outlier_detection.save_descriptions()
    outlier_detection.save_outliers()
    outlier_detection.visualize_clusters()
    
    # el dato más significativo para la detección de outliers fue 
    # ACTUAL_USAGE que representa el uso del servicio (datos) en
    # el intervalo de START_DATE a END_DATE

if __name__ == "__main__":
     detect_data_outliers()