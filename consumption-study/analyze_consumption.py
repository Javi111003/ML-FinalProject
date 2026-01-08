"""
Análisis de consumo - Versión Final con AutoML y Optuna
Universidad de La Habana - ML Final Project 2026

"""

import sys
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "automl"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from automl.config import DEFAULT_MODEL_CONFIGS
from automl.models.features import FeatureEngineering
from automl.models.factory import ModelFactory
from automl.models.base import ModelResult
from automl.models.automl import AutoMLTimeSeries

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

STUDY_DIR = Path(__file__).parent / "time-series"
OUTPUT_DIR = Path(__file__).parent / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Modelos ML a evaluar
MODELS_TO_EVALUATE = [
    'Linear Regression',
    'Ridge Regression', 
    'Random Forest',
    'Gradient Boosting'
]

# Configuración de features adaptada para series de 1 minuto
FEATURE_CONFIG = {
    'lag_features': {
        'max_lags': 7,
        'windows': [1, 3, 5, 7]
    },
    'rolling_features': {
        'windows': [3, 5, 7],
        'stats': ['mean', 'std', 'min', 'max']
    },
    'seasonal_features': {
        'include_fourier': False,
        'fourier_periods': [7],
        'fourier_terms': 2
    }
}

AUTOML_CONFIG = {
    'strategy': 'optuna_bayesian',
    'metric': 'rmse',
    'cv_splits': 5,
    'n_trials_per_model': 5  # Número de trials de Optuna por modelo
}


def load_time_series(service: str, freq: str = '1T') -> pd.Series:
    """Carga una serie temporal previamente creada"""
    
    file_path = STUDY_DIR / f"serie_temporal_{service}_{freq}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"❌ No se encontró: {file_path}\n"
            f"   Ejecuta primero: python create_time_series.py"
        )
    
    df = pd.read_csv(file_path, parse_dates=['fecha'])
    series = pd.Series(df['consumo'].values, index=df['fecha'], name=f'Consumo_{service.upper()}')
    
    print(f"✅ Serie temporal cargada: {service.upper()}")
    print(f"   📊 Períodos: {len(series)}")
    print(f"   📅 Rango: {series.index.min()} → {series.index.max()}")
    
    return series


def create_ml_features(series: pd.Series, config: dict) -> pd.DataFrame:
    """
    Crea features usando FeatureEngineering de automl
    """
    print("\n🔧 Creando features con FeatureEngineering...")
    
    fe = FeatureEngineering(config)
    
    # Determinar tipos de features a crear
    feature_types = []
    if config.get('lag_features'):
        feature_types.append('lag')
    if config.get('rolling_features'):
        feature_types.append('rolling')
    if config.get('seasonal_features'):
        feature_types.append('seasonal')
    
    # Crear features usando la infraestructura
    X = fe.create_features(
        series=series,
        dates=series.index,
        feature_types=feature_types
    )
    
    print(f"   ✅ Features creados: {X.shape[1]} columnas")
    
    return X


def prepare_data(series: pd.Series, X: pd.DataFrame) -> tuple:
    """Prepara los datos eliminando NaN y alineando índices"""
    valid_idx = ~X.isna().any(axis=1)
    X_clean = X[valid_idx]
    y_clean = series[valid_idx]
    
    print(f"\n📊 Datos preparados:")
    print(f"   Total períodos: {len(series)}")
    print(f"   Períodos válidos: {len(X_clean)} (sin NaN)")
    
    return X_clean, y_clean


def train_with_automl(
    X: pd.DataFrame, 
    y: pd.Series, 
    models_to_evaluate: list,
    n_trials: int = 5
) -> list:
    """
    Entrena modelos usando AutoMLTimeSeries de automl
    """
    
    print("\n" + "=" * 70)
    print("🤖 ENTRENAMIENTO CON AutoMLTimeSeries (Optuna)")
    print("=" * 70)

    selected_configs = {
        name: config 
        for name, config in DEFAULT_MODEL_CONFIGS.items() 
        if name in models_to_evaluate
    }    
    automl = AutoMLTimeSeries(
        model_configs=selected_configs,
        strategy=AUTOML_CONFIG['strategy'],
        metric=AUTOML_CONFIG['metric'],
        cv_splits=AUTOML_CONFIG['cv_splits']
    )
    
    results = []
    
    for model_name in models_to_evaluate:
        if model_name not in selected_configs:
            print(f"\n⚠️ {model_name} no está configurado en DEFAULT_MODEL_CONFIGS")
            continue
        
        print(f"\n{'─' * 50}")
        print(f"📌 Optimizando: {model_name}")
        print(f"{'─' * 50}")
        
        try:
            best_params, best_score = automl.optimize_model(
                model_name=model_name,
                X=X,
                y=y,
                n_trials=n_trials
            )
            
            print(f"   ✅ Mejores hiperparámetros: {best_params}")
            print(f"   📊 Mejor score (CV): {best_score:.4f}")
            
            model_config = selected_configs[model_name].copy()
            model_config['name'] = model_name
            model_config['params'] = best_params            
            final_model = ModelFactory.create_model(model_config)
            
            if final_model.model_type == 'ml':
                final_model.fit(series=y, X=X, y=y)
                y_pred = final_model.predict(X=X)
            else:
                final_model.fit(series=y)
                y_pred = final_model.predict(steps=len(y)) # Esto es in-sample prediction para estadísticos
            
            in_sample_rmse = np.sqrt(mean_squared_error(y, y_pred))
            in_sample_r2 = r2_score(y, y_pred)
            
            print(f"   📈 RMSE (In-Sample): {in_sample_rmse:.4f}")
            
            result = ModelResult(
                model_name=model_name,
                model_type=final_model.model_type,
                model_instance=final_model,
                params=best_params,
                score=best_score,
                features_shape=X.shape
            )            
            result.in_sample_rmse = in_sample_rmse
            result.in_sample_r2 = in_sample_r2
            result.y_pred_in_sample = y_pred
            
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    results = sorted(results, key=lambda x: x.score)
    
    return results


def evaluate_holdout(
    results: list,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> list:
    """
    Evaluación adicional con holdout set para validación final
    """
    print("\n" + "=" * 70)
    print("📊 EVALUACIÓN EN HOLDOUT SET (Test Set)")
    print("=" * 70)
    
    print(f"   Test:  {len(X_test)} períodos")
    
    for result in results:
        try:
            # Usar el modelo ya entrenado (que fue entrenado con X_train)
            model = result.model_instance
            
            # Predecir en test
            if model.model_type == 'ml':
                y_pred_test = model.predict(X=X_test)
            else:
                # Para estadísticos, predict suele requerir steps o horizonte
                y_pred_test = model.predict(steps=len(y_test))
                
                # Alinear explícitamente con el índice de test si es necesario
                if not isinstance(y_pred_test, pd.Series):
                    y_pred_test = pd.Series(y_pred_test, index=y_test.index)
            
            # Métricas en holdout
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            
            result.holdout_rmse = rmse
            result.holdout_r2 = r2
            result.y_pred_test = y_pred_test
            result.y_test = y_test
            
            print(f"\n   {result.model_name}:")
            print(f"      RMSE (holdout): {rmse:.4f}")
            print(f"      R² (holdout):   {r2:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error evaluando {result.model_name}: {e}")
    
    return results


def plot_results(results: list, service: str, output_dir: Path):
    """Genera gráficos de resultados"""
    
    u_m = {
        "voz": "Minutos",
        "sms": "Cantidad",
        "datos": "MB"
    }
    
    y_label = u_m[service]
    
    if not results:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1 = axes[0]
    best = results[0]
    
    if hasattr(best, 'y_test') and hasattr(best, 'y_pred_test'):     
        y_test_vals = best.y_test.values
        y_pred_vals = best.y_pred_test 
        
        if service == 'datos':
            y_test_vals = y_test_vals / 1048576 
            y_pred_vals = np.array(y_pred_vals) / 1048576
            
        ax1.plot(best.y_test.index, y_test_vals, 'k-', label='Real', alpha=0.7)
        ax1.plot(best.y_test.index, y_pred_vals, 'r--', label=f'Predicción ({best.model_name})', linewidth=2)
        ax1.set_title(f'Mejor Modelo: {best.model_name} (RMSE: {best.holdout_rmse:.2f})')
        ax1.set_ylabel(y_label)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False)) 
        ax1.ticklabel_format(style='plain', axis='y')
    
    ax2 = axes[1]
    names = [r.model_name for r in results]
    cv_scores = [r.score for r in results]
    holdout_scores = [getattr(r, 'holdout_rmse', 0) for r in results]
    
    if service == 'datos':
        cv_scores = [s / 1048576 for s in cv_scores]
        holdout_scores = [s / 1048576 for s in holdout_scores]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax2.bar(x - width/2, cv_scores, width, label='CV RMSE (Optuna)', color='skyblue')
    ax2.bar(x + width/2, holdout_scores, width, label='Holdout RMSE', color='salmon')
    
    ax2.set_ylabel('RMSE ' + (y_label))
    ax2.set_title('Comparación de Error por Modelo')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False)) 
    ax2.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    output_file = output_dir / f"resultados_{service}.png"
    plt.savefig(output_file)
    print(f"\n📊 Gráfico guardado: {output_file}")
    plt.close()


def generate_forecast(
    best_result: ModelResult,
    series: pd.Series,
    config: dict,
    steps: int,
    freq: str = '1T'
) -> pd.Series:
    """Genera predicciones futuras"""
    
    print(f"\n🔮 Generando predicción de {steps} períodos...")
    
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(freq), periods=steps, freq=freq)
    
    # Si es modelo estadístico, es directo
    if best_result.model_type == 'statistical':
        # Re-entrenar con todo
        best_result.model_instance.fit(series)
        preds = best_result.model_instance.predict(steps=steps)
        return pd.Series(preds, index=future_dates, name='Prediccion')
    
    # Si es ML, features iterativos
    fe = FeatureEngineering(config)
    extended_series = series.copy()
    predictions = []
    
    for date in future_dates:
        temp_series = extended_series.copy()
        temp_series[date] = 0
        
        X_future = fe.create_features(temp_series, temp_series.index, ['lag', 'rolling', 'seasonal'])
        X_pred = X_future.iloc[[-1]].fillna(0)
        
        pred = best_result.model_instance.predict(X=X_pred)[0]
        pred = max(0, pred)
        
        predictions.append(pred)
        extended_series[date] = pred
    
    forecast = pd.Series(predictions, index=future_dates, name='Prediccion')
    print(f"   ✅ Predicción generada")
    return forecast


def save_results(results: list, forecast: pd.Series, service: str, output_dir: Path):
    """Guarda resultados en CSV"""
    
    summary = []
    for r in results:
        summary.append({
            'Modelo': r.model_name,
            'CV_RMSE': r.score,
            'Holdout_RMSE': getattr(r, 'holdout_rmse', None),
            'Params': str(r.params)
        })
    
    pd.DataFrame(summary).to_csv(output_dir / f"resumen_modelos_{service}.csv", index=False)
    forecast_df = forecast.reset_index()
    forecast_df.columns = ['fecha', 'consumo']
    forecast_df.to_csv(output_dir / f"prediccion_{service}.csv", index=False)

    print(f"💾 Resultados guardados en CSV")


def analyze_service(service: str, freq: str = '1T', forecast_periods: int = 30):
    """Pipeline principal"""
    
    print("\n" + "=" * 80)
    print(f"🎯 ANÁLISIS: {service.upper()}")
    print("=" * 80)
    
    series = load_time_series(service, freq)
    X = create_ml_features(series, FEATURE_CONFIG)
    X_clean, y_clean = prepare_data(series, X)
    
    test_size = 0.2
    split_idx = int(len(X_clean) * (1 - test_size))
    
    X_train = X_clean.iloc[:split_idx]
    y_train = y_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_test = y_clean.iloc[split_idx:]
    
    results = train_with_automl(
        X=X_train, 
        y=y_train, 
        models_to_evaluate=MODELS_TO_EVALUATE,
        n_trials=AUTOML_CONFIG['n_trials_per_model']
    )
    
    if not results: return None
    
    results = evaluate_holdout(results, X_test, y_test)
    
    plot_results(results, service, OUTPUT_DIR)
    
    best_result = results[0]
    
    # Re-entrenar el mejor modelo con TODOS los datos para el forecast futuro
    print(f"\n🔄 Re-entrenando mejor modelo ({best_result.model_name}) con todos los datos para forecast...")
    
    model_config = DEFAULT_MODEL_CONFIGS[best_result.model_name].copy()
    model_config['name'] = best_result.model_name
    model_config['params'] = best_result.params
    
    final_model = ModelFactory.create_model(model_config)
    if final_model.model_type == 'ml':
        final_model.fit(series=y_clean, X=X_clean, y=y_clean)
    else:
        final_model.fit(series=y_clean)
        
    best_result.model_instance = final_model
    
    forecast = generate_forecast(best_result, series, FEATURE_CONFIG, forecast_periods, freq)
    
    save_results(results, forecast, service, OUTPUT_DIR)
    
    return {'best_model': best_result}


def main():
    print("=" * 80)
    print("🚀 ANÁLISIS DE CONSUMO - AUTOML COMPLETO")
    print("=" * 80)
    
    for service in ['voz', 'sms', 'datos']:
        try:
            analyze_service(service)
        except Exception as e:
            print(f"❌ Error en {service}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
