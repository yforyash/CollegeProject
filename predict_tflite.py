# predict_tflite.py
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

def load_and_prepare(csv_path, timesteps=36, channels=6):
    df = pd.read_csv(csv_path, header=None)  # no header assumed
    data = df.values.astype(np.float32)
    if data.shape[1] >= channels:
        data = data[:, :channels]
    else:
        pad_cols = channels - data.shape[1]
        data = np.pad(data, ((0,0),(0,pad_cols)), 'constant', constant_values=0.0)
    if data.shape[0] >= timesteps:
        data = data[:timesteps, :]
    else:
        pad_rows = timesteps - data.shape[0]
        data = np.pad(data, ((0,pad_rows),(0,0)), 'constant', constant_values=0.0)
    data = data.reshape(1, timesteps, channels, 1).astype(np.float32)
    return data

def predict(csv_path, model_path="model_quantized.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x = load_and_prepare(csv_path, timesteps=input_details[0]['shape'][1], channels=input_details[0]['shape'][2])
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    probs = output.flatten()
    predicted = int(np.argmax(probs))
    print(f"Input file: {csv_path}")
    print("Model output scores:", probs.tolist())
    print("Predicted class (argmax):", predicted)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_tflite.py <path_to_csv>")
        sys.exit(1)
    csv_file = sys.argv[1]
    predict(csv_file)
