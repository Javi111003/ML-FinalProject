import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

def auto_arima_like_with_csv(serie, p_range=(0,3), d_range=(0,2), q_range=(0,3)):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(p_range[0], p_range[1]+1):
        for d in range(d_range[0], d_range[1]+1):
            for q in range(q_range[0], q_range[1]+1):
                try:
                    model = ARIMA(serie, order=(p,d,q))
                    fit = model.fit()

                    # Guardar el summary en un CSV por cada combinación
                    summary_str = fit.summary().as_text()
                    summary_lines = summary_str.splitlines()
                    df_summary = pd.DataFrame(summary_lines, columns=["summary"])
                    df_summary.to_csv(f"./results/arima_summary_p{p}_d{d}_q{q}.csv", index=False)

                    # Selección del mejor modelo
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p,d,q)
                        best_model = fit
                except Exception:
                    continue

    print(f"Mejor modelo ARIMA{best_order} con AIC={best_aic:.2f}")
    return best_model