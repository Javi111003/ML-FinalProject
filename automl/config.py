DEFAULT_MODEL_CONFIGS = {
    "ARIMA": {
        "class": "statsmodels.tsa.arima.model.ARIMA",
        "params": {"p": [0, 1, 2, 3], "d": [0, 1, 2], "q": [0, 1, 2, 3]},
        "type": "statistical",
        "needs_frequency": False,
    },
    "SARIMA": {
        "class": "statsmodels.tsa.statespace.sarimax.SARIMAX",
        "params": {
            "order": [(1, 1, 1), (2, 1, 2), (1, 0, 0)],
            "seasonal_order": [(0, 0, 0, 12), (1, 1, 1, 12)],
        },
        "type": "statistical",
        "needs_frequency": True,
    },
    "Exponential Smoothing": {
        "class": "statsmodels.tsa.holtwinters.ExponentialSmoothing",
        "params": {
            "trend": ["add", "mul", None],
            "seasonal": ["add", "mul", None],
            "seasonal_periods": [4, 7, 12, 24],
        },
        "type": "statistical",
        "needs_frequency": True,
    },
    "Linear Regression": {
        "class": "sklearn.linear_model.LinearRegression",
        "params": {"fit_intercept": [True, False]},
        "type": "ml",
        "needs_frequency": False,
    },
    "Ridge Regression": {
        "class": "sklearn.linear_model.Ridge",
        "params": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "type": "ml",
        "needs_frequency": False,
    },
    "Lasso Regression": {
        "class": "sklearn.linear_model.Lasso",
        "params": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "type": "ml",
        "needs_frequency": False,
    },
    "ElasticNet": {
        "class": "sklearn.linear_model.ElasticNet",
        "params": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9]
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "Decision Tree": {
        "class": "sklearn.tree.DecisionTreeRegressor",
        "params": {
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": [None, "sqrt", "log2"],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "Gradient Boosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "XGBoost": {
        "class": "xgboost.XGBRegressor",
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "LightGBM": {
        "class": "lightgbm.LGBMRegressor",
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [-1, 5, 10, 20],
            "num_leaves": [31, 50, 100],
            "subsample": [0.8, 0.9, 1.0],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "CatBoost": {
        "class": "catboost.CatBoostRegressor",
        "params": {
            "iterations": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8, 10],
            "l2_leaf_reg": [1, 3, 5, 10],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "SVR": {
        "class": "sklearn.svm.SVR",
        "params": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "epsilon": [0.01, 0.1, 0.5],
            "kernel": ["rbf", "linear", "poly"],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "K-Neighbors": {
        "class": "sklearn.neighbors.KNeighborsRegressor",
        "params": {
            "n_neighbors": [3, 5, 7, 10, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],  # 1: Manhattan, 2: Euclidean
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "Neural Network": {
        "class": "sklearn.neural_network.MLPRegressor",
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "max_iter": [500, 1000],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "Bayesian Ridge": {
        "class": "sklearn.linear_model.BayesianRidge",
        "params": {
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "alpha_2": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4],
            "lambda_2": [1e-6, 1e-5, 1e-4],
        },
        "type": "ml",
        "needs_frequency": False,
    },
    "Huber Regressor": {
        "class": "sklearn.linear_model.HuberRegressor",
        "params": {
            "epsilon": [1.1, 1.35, 1.5, 2.0],
            "alpha": [0.0001, 0.001, 0.01],
        },
        "type": "ml",
        "needs_frequency": False,
    },
}

# Feature engineering defaults
FEATURE_ENGINEERING_CONFIG = {
    "lag_features": {"enabled": True, "max_lags": 12, "windows": [1, 3, 6, 12]},
    "rolling_features": {
        "enabled": True,
        "windows": [7, 14, 30],
        "stats": ["mean", "std", "min", "max"],
    },
    "seasonal_features": {
        "enabled": True,
        "include_fourier": True,
        "fourier_periods": [7, 365],
        "fourier_terms": 2,
    },
}

# Page configuration
PAGE_CONFIG = {
    "page_title": "AutoML Time Series Explorer",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# AutoML defaults
AUTOML_DEFAULTS = {
    "strategy": "optuna_bayesian",
    "metric": "rmse",
    "cv_splits": 5,
    "n_trials_per_model": 30,
    "timeout": 3600,  # seconds
}
