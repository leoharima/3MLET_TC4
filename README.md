# 3MLET-TC4 - LSTM API
Este projeto implementa uma API Flask para previsão dos próximos 10 fechamentos da ação GOOG utilizando um modelo LSTM treinado em PyTorch. A API oferece endpoints para predição e retreino do modelo, além de documentação Swagger e métricas Prometheus.

## Estrutura do Projeto

```
.
├── data/
│   └── GOOG.csv
├── static/
│   └── swagger.yaml
├── app.py
├── model_train.py
├── data_prep.py
├── lstm_model.pth
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── ...
```

## Funcionalidades

- **/predict**: Recebe os últimos 10 registros de features (Close, High, Low, Open, Volume) e retorna a previsão dos próximos 10 fechamentos.
- **/retrain**: Executa o retreino do modelo com os dados atuais e recarrega o modelo na API.
- **/apidocs**: Documentação Swagger interativa.
- **/metrics**: Métricas Prometheus para monitoramento.

## Como rodar localmente

1. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

2. Certifique-se de que o arquivo `lstm_model.pth` e o dataset `data/GOOG.csv` existem. Para baixar os dados:
    ```sh
    python data_prep.py
    ```

3. Treine o modelo (opcional, caso queira atualizar o modelo):
    ```sh
    python model_train.py
    ```

4. Inicie a API:
    ```sh
    python app.py
    ```

5. Acesse a documentação em [http://localhost:5000/apidocs](http://localhost:5000/apidocs)

## Como rodar com Docker

1. Build da imagem:
    ```sh
    docker build -t lstm-api .
    ```

2. Execute o container:
    ```sh
    docker run -p 5000:5000 lstm-api
    ```

3. Acesse a API em [http://localhost:5000](http://localhost:5000)

## Exemplo de uso do endpoint `/predict`

**Request:**
```
POST /predict
Content-Type: application/json

{
  "features": [
    {"Close": 123.4, "High": 124.0, "Low": 122.5, "Open": 123.0, "Volume": 1000000},
    ...
    // total de 10 registros
  ]
}
```

**Response:**
```
{
  "predictions": [125.1, 125.8, ..., 130.2]
}
```

## Retreino do modelo

Para retreinar o modelo com os dados atuais:
```sh
curl -X POST http://localhost:5000/retrain
```

## Métricas

As métricas estarão disponibilizadas em http://localhost:5000/metrics

## Observações

- O arquivo `static/swagger.yaml` contém a especificação Swagger da API.
- O modelo é salvo e carregado do arquivo `lstm_model.pth`.
- Os dados históricos são salvos em `data/GOOG.csv`.

## Autoria

**Grupo 19** - Leonardo André Ferreira - RM359721