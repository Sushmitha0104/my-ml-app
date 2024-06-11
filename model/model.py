import pickle
from pathlib import Path

def load_model():
    # Set the model path relative to the current file
    model_path = Path(__file__).parent / "model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
