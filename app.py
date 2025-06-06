import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flasgger import Swagger
from prometheus_flask_exporter import PrometheusMetrics
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import subprocess
import sys


# Definindo a aplicação Flask
app = Flask(__name__)
swagger = Swagger(app, template_file=os.path.join("static", "swagger.yaml"))
metrics = PrometheusMetrics(app, defaults=None)

# Definição do modelo LSTM igual ao do treinamento
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Parâmetros do modelo (iguais ao treinamento)
input_size = 5
hidden_size = 10
num_layers = 2
output_size = 10

# Inicializa modelo e carrega pesos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
app.model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
app.model.eval()

# Carrega scaler treinado (ajuste conforme necessário)
df = pd.read_csv('data/GOOG.csv')
df = df[['Close', 'High', 'Low', 'Open', 'Volume']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df.values)

# Rotas da API
@app.route('/')
def home():
    return "LSTM API"

@metrics.summary('predict_processing_seconds', 'Tempo processando o precict', labels={'endpoint': 'predict'})
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Espera um array de 10 objetos, cada um com as 5 features
    features = data.get('features')
    if not features or len(features) != 10:
        return jsonify({'error': 'Forneça exatamente 10 registros em features.'}), 400

    # Cada registro deve conter as 5 features na ordem: Close, High, Low, Open, Volume
    try:
        seq = np.array([[float(f['Close']), float(f['High']), float(f['Low']), float(f['Open']), float(f['Volume'])] for f in features])
    except Exception:
        return jsonify({'error': 'Cada registro deve conter as chaves Close, High, Low, Open e Volume.'}), 400

    # Normaliza usando o scaler treinado
    seq_scaled = scaler.transform(seq)
    seq_scaled = seq_scaled.reshape(1, 10, 5)  # (batch, seq_len, features)

    # Ajusta para o tamanho de sequência esperado (20)
    ultimos_20 = df.values[-20:].copy()
    ultimos_20[-10:, :] = seq  # substitui os últimos 10 registros pelas features fornecidas
    seq_full = scaler.transform(ultimos_20).reshape(1, 20, 5)

    input_tensor = torch.tensor(seq_full, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = app.model(input_tensor).cpu().numpy()[0]

    # Desnormaliza as previsões
    predictions = []
    for pred in output:
        inv = scaler.inverse_transform([[pred, 0, 0, 0, 0]])[0, 0]
        predictions.append(float(inv))

    return jsonify({'predictions': predictions})

@metrics.summary('retrain_processing_seconds', 'Tempo processando o retreino', labels={'endpoint': 'retrain'})
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        result = subprocess.run(
            [sys.executable, "model_train.py"],
            capture_output=True,
            text=True,
            check=True
        )
        # Recarrega o modelo treinado após o retreino
        app.model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
        app.model.eval()
        return jsonify({
            "status": "success",
            "output": result.stdout
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "output": e.stdout,
            "error": e.stderr
        }), 500

def main():
    app.run(debug=False, use_reloader=False)

if __name__ == '__main__':
    main()