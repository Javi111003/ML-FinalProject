import pandas as pd
import matplotlib.pyplot as plt

# Directorio donde están los CSV
path = "./results/"

# Leer cada archivo (asegúrate de que la primera columna se llame 'fecha')
df_datos = pd.read_csv(path + "prediccion_datos.csv", parse_dates=["fecha"])
df_voz   = pd.read_csv(path + "prediccion_voz.csv", parse_dates=["fecha"])
df_sms   = pd.read_csv(path + "prediccion_sms.csv", parse_dates=["fecha"])

# --- Gráfico de Consumo de Datos ---
plt.figure(figsize=(10,5))
plt.plot(df_datos["fecha"], df_datos["consumo"] / (1024**2), label="Datos", color="blue")
plt.xlabel("fecha")
plt.ylabel("Consumo (MB)")
plt.title("Consumo predicho de datos móviles")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(path + "consumo_datos.png")   # guarda la gráfica

# --- Gráfico de Consumo de Voz ---
plt.figure(figsize=(10,5))
plt.plot(df_voz["fecha"], df_voz["consumo"], label="Voz", color="green")
plt.xlabel("fecha")
plt.ylabel("Minutos")
plt.title("Consumo predicho de voz")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(path + "consumo_voz.png")     # guarda la gráfica

# --- Gráfico de Consumo de SMS ---
plt.figure(figsize=(10,5))
plt.plot(df_sms["fecha"], df_sms["consumo"], label="SMS", color="red")
plt.xlabel("fecha")
plt.ylabel("Cantidad")
plt.title("Cantidad predicha de SMS")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(path + "consumo_sms.png")     # guarda la gráfica
