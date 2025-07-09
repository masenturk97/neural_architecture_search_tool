import pandas as pd
import keras
import os

dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "sample_regression_dataset.csv")
dataset_path = os.path.abspath(dataset_path)

df = pd.read_csv(dataset_path)
input_heads = [c for c in df.columns if c.startswith('x')]
inputs = df[input_heads].to_numpy()
output_heads = [c for c in df.columns if c.startswith('y')]
outputs = df[output_heads].to_numpy()

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "saved_best_model.keras")
model_path = os.path.abspath(model_path)

model = keras.models.load_model(model_path)
results = model.predict(inputs)
print(f"Target Value: {outputs[0]} and Predicted Value: {results[0]}")