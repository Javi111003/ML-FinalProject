"""
Visualización y comparación de consumo VOZ vs SMS
Universidad de La Habana - ML Final Project 2026

Genera gráficos comparativos entre servicios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

STUDY_DIR = Path(__file__).parent
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_all_series(freq: str = '1T') -> dict:
    """Carga todas las series temporales disponibles"""
    
    series_dict = {}
    services = ['voz', 'sms', 'datos']
    
    for service in services:
        file_path = STUDY_DIR / f"serie_temporal_{service}_{freq}.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=['fecha'])
            series = pd.Series(df['consumo'].values, index=df['fecha'], name=service.upper())
            series_dict[service] = series
            print(f"✅ Cargada: {service.upper()} ({len(series)} períodos)")
        else:
            print(f"⚠️ No encontrada: {service}")
    
    return series_dict


def plot_comparison(series_dict: dict, normalize: bool = True):
    """Genera gráfico comparativo de series"""
    
    print("\n📊 Generando gráfico comparativo...")
    
    n_services = len(series_dict)
    
    if n_services == 0:
        print("❌ No hay series para comparar")
        return
    
    # Crear figura
    fig, axes = plt.subplots(n_services + 1, 1, figsize=(14, 4 * (n_services + 1)))
    
    if n_services == 1:
        axes = [axes, axes]  # Para mantener consistencia
    
    colors = {'voz': '#2ecc71', 'sms': '#e74c3c', 'datos': '#3498db'}
    
    # Gráficos individuales
    for i, (service, series) in enumerate(series_dict.items()):
        ax = axes[i]
        color = colors.get(service, 'gray')
        
        ax.plot(series.index, series.values, color=color, linewidth=1.5, label=service.upper())
        ax.fill_between(series.index, series.values, alpha=0.3, color=color)
        
        ax.set_title(f'Consumo de {service.upper()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Consumo')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Añadir estadísticas
        mean_val = series.mean()
        ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.7, label=f'Media: {mean_val:.1f}')
    
    # Gráfico combinado (normalizado)
    ax_combined = axes[-1]
    
    if normalize:
        scaler = MinMaxScaler()
        for service, series in series_dict.items():
            color = colors.get(service, 'gray')
            normalized = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            ax_combined.plot(series.index, normalized, color=color, 
                           linewidth=1.5, label=f'{service.upper()} (normalizado)')
    else:
        for service, series in series_dict.items():
            color = colors.get(service, 'gray')
            ax_combined.plot(series.index, series.values, color=color, 
                           linewidth=1.5, label=service.upper())
    
    ax_combined.set_title('Comparación de Todos los Servicios', fontsize=14, fontweight='bold')
    ax_combined.set_xlabel('Fecha')
    ax_combined.set_ylabel('Consumo (normalizado)' if normalize else 'Consumo')
    ax_combined.legend(loc='upper right')
    ax_combined.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = STUDY_DIR / "comparacion_servicios.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {output_file}")
    
    plt.close()


def plot_hourly_patterns(series_dict: dict):
    """Analiza patrones horarios (promedio por hora del día)"""
    
    print("\n📊 Analizando patrones horarios...")
    
    fig, axes = plt.subplots(1, len(series_dict), figsize=(6 * len(series_dict), 5))
    
    if len(series_dict) == 1:
        axes = [axes]
    
    colors = {'voz': '#2ecc71', 'sms': '#e74c3c', 'datos': '#3498db'}
    
    for i, (service, series) in enumerate(series_dict.items()):
        ax = axes[i]
        color = colors.get(service, 'gray')
        
        # Agrupar por hora del día (0-23)
        hourly = series.groupby(series.index.hour).mean()
        
        # Rellenar horas faltantes con 0 para mantener el eje x consistente (0-23)
        full_range = range(24)
        hourly = hourly.reindex(full_range, fill_value=0)
        
        ax.bar(hourly.index, hourly.values, color=color, alpha=0.7, width=0.8)
        ax.set_xticks(range(0, 24, 2)) # Etiquetas cada 2 horas
        ax.set_xlim(-0.5, 23.5)
        
        ax.set_title(f'Patrón Horario - {service.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hora del día')
        ax.set_ylabel('Consumo Promedio')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = STUDY_DIR / "patrones_horarios.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {output_file}")
    
    plt.close()


def plot_distribution(series_dict: dict):
    """Analiza distribución de consumo"""
    
    print("\n📊 Analizando distribución de consumo...")
    
    fig, axes = plt.subplots(1, len(series_dict), figsize=(5 * len(series_dict), 5))
    
    if len(series_dict) == 1:
        axes = [axes]
    
    colors = {'voz': '#2ecc71', 'sms': '#e74c3c', 'datos': '#3498db'}
    
    for i, (service, series) in enumerate(series_dict.items()):
        ax = axes[i]
        color = colors.get(service, 'gray')
        
        # Histograma con KDE
        ax.hist(series.values, bins=30, color=color, alpha=0.7, density=True, edgecolor='white')
        #series.plot.kde(ax=ax, color='black', linewidth=2, label='KDE')
        
        # Ajuste de distribución exponencial
        try:
            loc, scale = stats.expon.fit(series.values)
            x_range = np.linspace(series.min(), series.max(), 200)
            pdf_expon = stats.expon.pdf(x_range, loc, scale)
            
            ax.plot(x_range, pdf_expon, color='blue', linestyle='--', linewidth=2, label='Ajuste Exponencial')
        except Exception as e:
            print(f"No se pudo ajustar exponencial para {service}: {e}")

        
        ax.set_title(f'Distribución - {service.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Consumo')
        ax.set_ylabel('Densidad')
        
        # Añadir estadísticas
        mean_val = series.mean()
        std_val = series.std()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.1f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    
    output_file = STUDY_DIR / "distribucion_consumo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {output_file}")
    
    plt.close()


def plot_correlation_matrix(series_dict: dict):
    """Genera matriz de correlación entre servicios"""
    
    if len(series_dict) < 2:
        print("⚠️ Se necesitan al menos 2 servicios para correlación")
        return
    
    print("\n📊 Calculando correlaciones...")
    
    df = pd.DataFrame(series_dict)
    
    df = df.dropna()
    
    if len(df) < 10:
        print("⚠️ Muy pocos datos para calcular correlación")
        return
    
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, 
                square=True, linewidths=1, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_title('Correlación entre Servicios', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = STUDY_DIR / "correlacion_servicios.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {output_file}")
    
    plt.close()
    
    print("\n📈 Correlaciones:")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            col1, col2 = corr.columns[i], corr.columns[j]
            val = corr.iloc[i, j]
            print(f"   {col1} ↔ {col2}: {val:.4f}")


def generate_summary_stats(series_dict: dict):
    """Genera resumen estadístico"""
    
    print("\n📊 Resumen Estadístico:")
    print("=" * 60)
    
    stats_data = []
    
    for service, series in series_dict.items():
        stats = {
            'Servicio': service.upper(),
            'Períodos': len(series),
            'Media': series.mean(),
            'Mediana': series.median(),
            'Desv. Std': series.std(),
            'Mínimo': series.min(),
            'Máximo': series.max(),
            'Total': series.sum()
        }
        stats_data.append(stats)
        
        print(f"\n📌 {service.upper()}:")
        print(f"   Períodos: {stats['Períodos']}")
        print(f"   Media: {stats['Media']:.2f}")
        print(f"   Mediana: {stats['Mediana']:.2f}")
        print(f"   Desv. Std: {stats['Desv. Std']:.2f}")
        print(f"   Rango: [{stats['Mínimo']:.2f}, {stats['Máximo']:.2f}]")
        print(f"   Total: {stats['Total']:.2f}")
    
    # Guardar resumen
    df_stats = pd.DataFrame(stats_data)
    output_file = STUDY_DIR / "resumen_estadistico.csv"
    df_stats.to_csv(output_file, index=False)
    print(f"\n💾 Resumen guardado: {output_file}")


def main():
    """Función principal"""
    
    print("=" * 80)
    print("📊 VISUALIZACIÓN Y COMPARACIÓN DE SERVICIOS")
    print("=" * 80)
    
    series_dict = load_all_series(freq='1T')
    
    if not series_dict:
        print("\n❌ No se encontraron series temporales")
        print("   Ejecuta primero: python create_time_series.py")
        return
    
    generate_summary_stats(series_dict)
    
    plot_comparison(series_dict)
    plot_hourly_patterns(series_dict)
    plot_distribution(series_dict)
    plot_correlation_matrix(series_dict)
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n📁 Gráficos generados en: {STUDY_DIR}")


if __name__ == "__main__":
    main()
