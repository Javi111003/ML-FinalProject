"""
Script para filtrar muestra.xlsx y generar CSVs por tipo de servicio
Universidad de La Habana - ML Final Project 2026

Genera:
- voz_completados.csv (registros de LLAMADAS)
- sms_completados.csv (registros de SMS/Mensajes)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


INPUT_FILE = Path(__file__).parent.parent / "data" / "muestraok" / "muestra.xlsx"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

SERVICE_COLUMN = 'SERVICE_CATEGORY'  

VOZ_VALUES = [1]     
SMS_VALUES = [2]   
DATOS_VALUES = [5]   

DATE_COLUMN = 'START_DATE'     
USAGE_COLUMN = 'ACTUAL_USAGE'  

def analyze_service_columns(df: pd.DataFrame) -> dict:
    """Analiza las columnas que identifican tipo de servicio"""
    
    print("\n" + "=" * 80)
    print("TODAS LAS COLUMNAS DEL DATASET")
    print("=" * 80)
    
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        unique = df[col].nunique()
        print(f"  {i:2d}. {col:<35} | {str(dtype):<10} | Únicos: {unique:<6} | Nulos: {nulls}")
    
    print("\n" + "=" * 80)
    print("🔍 COLUMNAS CANDIDATAS PARA FILTRAR TIPO DE SERVICIO")
    print("=" * 80)
    
    # Columnas candidatas para filtrado
    service_cols = [col for col in df.columns if any(keyword in str(col).upper() 
                    for keyword in ['SERVICE', 'TYPE', 'CATEGORY', 'USAGE', 'EVENT', 'EVT'])]
    
    analysis = {}
    
    for col in service_cols:
        if col in df.columns:
            unique_vals = df[col].unique()
            value_counts = df[col].value_counts()
            
            print(f"\n📌 Columna: '{col}'")
            print(f"   Tipo: {df[col].dtype}")
            print(f"   Valores únicos: {len(unique_vals)}")
            print(f"\n   Distribución:")
            for val, count in value_counts.items():
                pct = count / len(df) * 100
                bar = "█" * int(pct / 3)
                print(f"      {str(val):<15} : {count:>6,} ({pct:>5.1f}%) {bar}")
            
            analysis[col] = {
                'unique_values': unique_vals,
                'value_counts': value_counts
            }
    
    # Columnas de fecha
    print("\n" + "=" * 80)
    print("📅 COLUMNAS DE FECHA")
    print("=" * 80)
    
    date_cols = [col for col in df.columns 
                 if any(kw in str(col).upper() for kw in ['DATE', 'TIME', 'FECHA'])]
    for col in date_cols:
        sample = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
        print(f"   - {col:<30} | Ejemplo: {sample}")
    
    # Columnas de uso
    print("\n" + "=" * 80)
    print("📈 COLUMNAS DE USO/CONSUMO")
    print("=" * 80)
    
    usage_cols = [col for col in df.columns 
                  if any(kw in str(col).upper() for kw in ['USAGE', 'CHARGE', 'FLUX', 'AMOUNT'])]
    for col in usage_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            print(f"   - {col}")
            print(f"     Min: {stats['min']:.2f} | Max: {stats['max']:.2f} | Media: {stats['mean']:.2f}")
    
    return analysis


def preprocess_dataframe(df: pd.DataFrame, service_type: str) -> pd.DataFrame:
    """Limpieza y preparación del dataframe"""
    
    print(f"\n  🔧 Procesando {service_type}...")
    
    # Eliminar duplicados
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"    - Duplicados eliminados: {removed}")
    
    # Convertir fechas
    date_columns = ['START_DATE', 'END_DATE', 'EVENT_TIME', 'CHARGE_TIME']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"    - {col} convertida a datetime")
            except Exception as e:
                print(f"    - Error en {col}: {e}")
    
    # Rellenar valores nulos en columnas numéricas clave
    key_numeric_cols = ['ACTUAL_USAGE', 'RATING_USAGE', 'ACTUAL_CHARGE', 'TOTAL_TAX_AMOUNT']
    for col in key_numeric_cols:
        if col in df.columns:
            nulls = df[col].isna().sum()
            if nulls > 0:
                df[col] = df[col].fillna(0)
                print(f"    - {col}: {nulls} nulos → 0")
    
    return df


def filter_and_save(df: pd.DataFrame):
    """Filtra por tipo de servicio y guarda los CSV"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📦 GENERANDO ARCHIVOS CSV")
    print("=" * 80)
    
    results = {}
    
    if SERVICE_COLUMN not in df.columns:
        print(f"\n❌ ERROR: Columna '{SERVICE_COLUMN}' no encontrada")
        print(f"   Columnas disponibles:")
        for col in df.columns:
            print(f"      - {col}")
        print(f"\n   Edita SERVICE_COLUMN en la sección CONFIGURACIÓN")
        return results
    
    print(f"\n✓ Columna de filtrado: {SERVICE_COLUMN}")
    print(f"   Valores en dataset: {sorted(df[SERVICE_COLUMN].unique())}")
    
    # --- FILTRAR VOZ ---
    print(f"\n📞 Filtrando VOZ...")
    print(f"   Condición: {SERVICE_COLUMN} in {VOZ_VALUES}")
    
    df_voz = df[df[SERVICE_COLUMN].isin(VOZ_VALUES)].copy()
    
    if len(df_voz) == 0:
        print(f"   ⚠️ ADVERTENCIA: No se encontraron registros")
        print(f"   Verifica que VOZ_VALUES={VOZ_VALUES} sea correcto")
    else:
        df_voz = preprocess_dataframe(df_voz, "VOZ")
        output_voz = OUTPUT_DIR / "voz_completados.csv"
        df_voz.to_csv(output_voz, index=False, encoding='utf-8')
        
        print(f"   ✅ Guardado: {output_voz}")
        print(f"   📊 Registros: {len(df_voz):,}")
        
        results['voz'] = {
            'path': output_voz,
            'records': len(df_voz),
            'filter_values': VOZ_VALUES
        }
    
    # --- FILTRAR SMS ---
    print(f"\n💬 Filtrando SMS...")
    print(f"   Condición: {SERVICE_COLUMN} in {SMS_VALUES}")
    
    df_sms = df[df[SERVICE_COLUMN].isin(SMS_VALUES)].copy()
    
    if len(df_sms) == 0:
        print(f"   ⚠️ ADVERTENCIA: No se encontraron registros")
        print(f"   Verifica que SMS_VALUES={SMS_VALUES} sea correcto")
    else:
        df_sms = preprocess_dataframe(df_sms, "SMS")
        output_sms = OUTPUT_DIR / "sms_completados.csv"
        df_sms.to_csv(output_sms, index=False, encoding='utf-8')
        
        print(f"   ✅ Guardado: {output_sms}")
        print(f"   📊 Registros: {len(df_sms):,}")
        
        results['sms'] = {
            'path': output_sms,
            'records': len(df_sms),
            'filter_values': SMS_VALUES
        }
        
    print(f"\n🌐 Filtrando DATOS...")
    print(f"   Condición: {SERVICE_COLUMN} in {DATOS_VALUES}")
    
    df_datos = df[df[SERVICE_COLUMN].isin(DATOS_VALUES)].copy()
    
    if len(df_datos) == 0:
        print(f"   ⚠️ ADVERTENCIA: No se encontraron registros de DATOS")
    else:
        df_datos = preprocess_dataframe(df_datos, "DATOS")
        output_datos = OUTPUT_DIR / "datos_completados.csv"
        df_datos.to_csv(output_datos, index=False, encoding='utf-8')
        
        print(f"   ✅ Guardado: {output_datos}")
        print(f"   📊 Registros: {len(df_datos):,}")
        
        results['datos'] = {
            'path': output_datos,
            'records': len(df_datos),
            'filter_values': DATOS_VALUES
        }
    
    return results


def main():
    """Función principal"""
    
    filter_mode = len(sys.argv) > 1 and sys.argv[1] == "--filter"
    
    print("\n" + "=" * 80)
    if filter_mode:
        print("🔧 MODO: FILTRADO Y GENERACIÓN DE CSV")
    else:
        print("🔍 MODO: ANÁLISIS DEL DATASET")
    print("=" * 80)
    
    # 1. Cargar datos
    print(f"\n📥 Cargando: {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        print(f"\n❌ ERROR: No se encontró el archivo")
        print(f"   Ruta: {INPUT_FILE}")
        return
    
    df = pd.read_excel(INPUT_FILE)
    print(f"✅ Cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    
    analysis = analyze_service_columns(df)
    
    if not filter_mode:
        # Solo análisis - mostrar instrucciones
        print("\n" + "=" * 80)
        print("📝 PRÓXIMOS PASOS")
        print("=" * 80)
        print(f"""
    1. Revisa la información anterior
    
    2. Edita este archivo (filter_services.py) y ajusta:
    
       SERVICE_COLUMN = '???'   # Columna para filtrar (ej: 'SERVICE_CATEGORY')
       VOZ_VALUES = [???]       # Valores que identifican VOZ
       SMS_VALUES = [???]       # Valores que identifican SMS
       
    3. Guarda el archivo
    
    4. Ejecuta: python filter_services.py --filter
        """)
        return
    
    # Filtrar y guardar (solo en modo --filter)
    print("\n" + "=" * 80)
    print("🎯 CONFIGURACIÓN ACTUAL")
    print("=" * 80)
    print(f"   SERVICE_COLUMN = '{SERVICE_COLUMN}'")
    print(f"   VOZ_VALUES = {VOZ_VALUES}")
    print(f"   SMS_VALUES = {SMS_VALUES}")
    
    results = filter_and_save(df)
    
    if results:
        print("\n" + "=" * 80)
        print("✅ PROCESO COMPLETADO")
        print("=" * 80)
        
        total_original = len(df)
        total_procesado = sum(r['records'] for r in results.values())
        
        print(f"\n📊 Resumen:")
        print(f"   Total original: {total_original:,}")
        print(f"   VOZ + SMS: {total_procesado:,}")
        print(f"   Otros: {total_original - total_procesado:,}")
        
        print(f"\n📁 Archivos en: {OUTPUT_DIR}")
        for service, info in results.items():
            print(f"   • {info['path'].name}: {info['records']:,} registros")
    
    return results


if __name__ == "__main__":
    main()
