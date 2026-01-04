"""
Script para crear series temporales de consumo a partir de los CSV filtrados
Universidad de La Habana - ML Final Project 2026

Genera series temporales agregadas por intervalos de tiempo para VOZ y SMS
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def load_service_data(service: str) -> pd.DataFrame:
    """Carga los datos filtrados de un servicio"""
    
    file_path = DATA_DIR / f"{service}_completados.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"❌ No se encontró: {file_path}\n"
                               f"   Ejecuta primero: python filter_services.py")
    
    df = pd.read_csv(file_path)
    print(f"✅ Cargado {service}: {len(df):,} registros")
    
    return df


def prepare_datetime(df: pd.DataFrame, date_col: str = 'START_DATE') -> pd.DataFrame:
    """Prepara la columna de fecha como datetime"""
    
    if date_col not in df.columns:
        # Buscar alternativas
        date_alternatives = ['END_DATE', 'EVENT_TIME', 'CHARGE_TIME']
        for alt in date_alternatives:
            if alt in df.columns:
                date_col = alt
                break
        else:
            raise ValueError(f"❌ No se encontró columna de fecha. "
                           f"Columnas disponibles: {df.columns.tolist()}")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Eliminar registros sin fecha válida
    invalid_dates = df[date_col].isna().sum()
    if invalid_dates > 0:
        print(f"   ⚠️ {invalid_dates} registros sin fecha válida eliminados")
        df = df.dropna(subset=[date_col])
    
    return df, date_col


def create_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str = 'ACTUAL_USAGE',
    freq: str = 'H',
    agg_func: str = 'sum'
) -> pd.Series:
    """
    Crea una serie temporal agregada
    
    Parameters:
    -----------
    df : DataFrame con los datos
    date_col : Columna de fecha
    value_col : Columna a agregar (ACTUAL_USAGE por defecto)
    freq : Frecuencia de agregación ('H'=hora, 'D'=día, 'W'=semana, 'M'=mes)
    agg_func : Función de agregación ('sum', 'mean', 'count', 'max', 'min')
    
    Returns:
    --------
    pd.Series con índice temporal
    """
    
    if value_col not in df.columns:
        print(f"   ⚠️ Columna '{value_col}' no encontrada, usando conteo de registros")
        value_col = None
    
    df_temp = df.copy()   
    df_temp[date_col] = pd.to_datetime(df_temp[date_col]) 
    df_temp = df_temp.set_index(date_col)
    
    if value_col:
        if agg_func == 'sum':
            series = df_temp[value_col].resample(freq).sum()
        elif agg_func == 'mean':
            series = df_temp[value_col].resample(freq).mean()
        elif agg_func == 'count':
            series = df_temp[value_col].resample(freq).count()
        elif agg_func == 'max':
            series = df_temp[value_col].resample(freq).max()
        elif agg_func == 'min':
            series = df_temp[value_col].resample(freq).min()
        else:
            series = df_temp[value_col].resample(freq).sum()
    else:
        # Si no hay columna de valor, contar registros
        series = df_temp.resample(freq).size()
    
    # Rellenar valores faltantes con 0
    series = series.fillna(0)
    
    return series


def analyze_time_series(series: pd.Series, name: str) -> dict:
    """Analiza estadísticas básicas de la serie temporal"""
    
    stats = {
        'name': name,
        'start_date': series.index.min(),
        'end_date': series.index.max(),
        'periods': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'zeros': (series == 0).sum(),
        'total': series.sum()
    }
    
    return stats


def print_series_stats(stats: dict):
    """Imprime estadísticas de la serie temporal"""
    
    print(f"\n📊 Estadísticas de {stats['name']}:")
    print(f"   📅 Rango: {stats['start_date'].date()} → {stats['end_date'].date()}")
    print(f"   📈 Períodos: {stats['periods']}")
    print(f"   📉 Media: {stats['mean']:.2f}")
    print(f"   📊 Desv. Std: {stats['std']:.2f}")
    print(f"   🔻 Mínimo: {stats['min']:.2f}")
    print(f"   🔺 Máximo: {stats['max']:.2f}")
    print(f"   ⭕ Períodos en cero: {stats['zeros']}")
    print(f"   📦 Total acumulado: {stats['total']:.2f}")


def process_service(
    service: str,
    freq: str = 'H',
    value_col: str = 'ACTUAL_USAGE',
    agg_func: str = 'sum'
) -> Tuple[pd.Series, dict]:
    """
    Procesa un servicio completo: carga, prepara y crea serie temporal
    
    Parameters:
    -----------
    service : 'voz' o 'sms'
    freq : Frecuencia de agregación
    value_col : Columna a agregar
    agg_func : Función de agregación
    
    Returns:
    --------
    (serie_temporal, estadísticas)
    """
    
    print(f"\n{'='*60}")
    print(f"📌 PROCESANDO: {service.upper()}")
    print(f"{'='*60}")
    
    df = load_service_data(service)
    
    df, date_col = prepare_datetime(df)
    
    print(f"\n🔄 Creando serie temporal...")
    print(f"   Frecuencia: {freq}")
    print(f"   Columna valor: {value_col}")
    print(f"   Agregación: {agg_func}")
    
    series = create_time_series(df, date_col, value_col, freq, agg_func)
    series.name = f"Consumo_{service.upper()}"
    
    stats = analyze_time_series(series, service.upper())
    print_series_stats(stats)
    
    return series, stats


def save_time_series(series: pd.Series, service: str, freq: str, remove_zeros: bool = False) -> Path:
    """Guarda la serie temporal en CSV"""
    
    if remove_zeros:
        original_len = len(series)
        series = series[series > 0]
        removed = original_len - len(series)
        print(f"   🗑️ Períodos con cero eliminados: {removed}")    
        
    output_file = OUTPUT_DIR / f"serie_temporal_{service}_{freq}.csv"
    
    df_output = pd.DataFrame({
        'fecha': series.index,
        'consumo': series.values
    })
    
    df_output.to_csv(output_file, index=False)
    print(f"   💾 Guardado: {output_file}")
    
    return output_file


def main():
    """Función principal"""
    
    print("=" * 80)
    print("📊 CREACIÓN DE SERIES TEMPORALES - VOZ y SMS")
    print("=" * 80)
    
    FREQ = '1T'  # 'H'=hora, 'D'=día, 'W'=semana, 'M'=mes
    VALUE_COL = 'ACTUAL_USAGE'  # Columna a agregar
    AGG_FUNC = 'sum'  # 'sum', 'mean', 'count'
    
    results = {}
    
    # Procesar VOZ
    try:
        series_voz, stats_voz = process_service('voz', FREQ, VALUE_COL, AGG_FUNC)
        save_time_series(series_voz, 'voz', FREQ, remove_zeros=True)
        results['voz'] = {'series': series_voz, 'stats': stats_voz}
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
    except Exception as e:
        print(f"\n❌ Error procesando VOZ: {e}")
    
    # Procesar SMS
    try:
        series_sms, stats_sms = process_service('sms', FREQ, VALUE_COL, AGG_FUNC)
        save_time_series(series_sms, 'sms', FREQ, remove_zeros=True)
        results['sms'] = {'series': series_sms, 'stats': stats_sms}
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
    except Exception as e:
        print(f"\n❌ Error procesando SMS: {e}")
    
    # Procesar DATOS    
    try:
        series_datos, stats_datos = process_service('datos', FREQ, VALUE_COL, AGG_FUNC)
        save_time_series(series_datos, 'datos', FREQ, remove_zeros=True)
        results['datos'] = {'series': series_datos, 'stats': stats_datos}
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
    except Exception as e:
        print(f"\n❌ Error procesando DATOS: {e}")
    
    
    # Resumen final
    print("\n" + "=" * 80)
    print("✅ SERIES TEMPORALES CREADAS")
    print("=" * 80)
    
    print(f"\n📁 Archivos generados en: {OUTPUT_DIR}")
    for service in results:
        print(f"   • serie_temporal_{service}_{FREQ}.csv")
    
    return results


if __name__ == "__main__":
    main()
