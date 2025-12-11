import warnings
import joblib
import inspect


def save_model(model, filename):
    """Save trained model to file"""
    joblib.dump(model, filename)


def load_model(filename):
    """Load model from file"""
    return joblib.load(filename)


def get_function_parameters(func):
    """Get parameters of a function"""
    return inspect.signature(func).parameters


def setup_warnings():
    """Configure warning settings"""
    warnings.filterwarnings("ignore")
