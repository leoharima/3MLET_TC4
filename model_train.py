import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.pytorch

# Define o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros do modelo
input_size = 5
hidden_size = 10
num_layers = 2
num_epochs = 30
batch_size = 64
learning_rate = 0.001
sequence_length = 20
output_size = 10

# Carrega e prepara os dados
df = pd.read_csv('data/GOOG.csv')
df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

# Normaliza os dados
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df.values)

def cria_sequencia(df, sequence_length, output_size):
    """
    Cria sequências de entrada e saída para o modelo LSTM.
    Cada sequência de entrada tem tamanho sequence_length e a saída são os próximos output_size valores de 'Close'.
    """
    sequences = []
    labels = []
    for i in range(len(df) - sequence_length - output_size + 1):
        seq = df[i:i + sequence_length]
        label = df[i + sequence_length:i + sequence_length + output_size, 0]  # Predicting the 'Close' price for future_days
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Gera os dados de entrada e saída
X, y = cria_sequencia(df_scaled, sequence_length, output_size)

# Divide em treino e teste (sem embaralhar para séries temporais)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# Converte para tensores do PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Cria DataLoaders para treino e teste
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# LSTM Model
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


# Training the model
def train_model():
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("LSTM")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Save the model
        mlflow.pytorch.log_model(model, "lstm_model")

        # Evaluate the model
        evaluate_model(model, criterion)


def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_preds.append(outputs.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())  

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = math.sqrt(mse)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    mlflow.log_metric("test_loss", average_test_loss)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)

    # Avaliação nos dados de treino
    train_loss = 0.0
    train_preds = []
    train_labels = []
    with torch.no_grad():
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_preds.append(outputs.cpu().detach().numpy())
            train_labels.append(labels.cpu().detach().numpy())
    average_train_loss = train_loss / len(train_loader)
    train_preds = np.concatenate(train_preds, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_mae = mean_absolute_error(train_labels, train_preds)
    train_mse = mean_squared_error(train_labels, train_preds)
    train_rmse = math.sqrt(train_mse)
    print(f"Train Loss: {average_train_loss:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")

    mlflow.log_metric("train_loss", average_train_loss)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    
    # Predict future days' closing prices
    next_day_input = X_test[-1].unsqueeze(0)  # Get the last sequence from test set
    future_predictions = model(next_day_input).cpu().detach().numpy()  

    # Denormalize the predictions
    future_predictions_denormalized = []
    for i in range(output_size):
        prediction = scaler.inverse_transform([[0, 0, 0, future_predictions[0, i], 0]])[0, 3]  # Reverse scaling
        future_predictions_denormalized.append(prediction)
        print(f"Day {i + 1} predicted 'Close' price: {prediction:.4f}")
    
    # Exemplo de reversão da normalização do MAE
    mae_original_scale = scaler.inverse_transform([[0, 0, 0, 0.0266, 0]])[0, 3]
    print(f"MAE na escala original: {mae_original_scale:.4f}")

    # Save the model to a file
    torch.save(model.state_dict(), 'lstm_model.pth')
    print("Model saved to lstm_model.pth")


# Run the training and evaluation
train_model()