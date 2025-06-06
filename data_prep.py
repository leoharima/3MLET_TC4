import yfinance as yf


# Especifique o símbolo da empresa que você vai trabalhar
# Configure data de início e fim da sua base
symbol = 'GOOG'
start_date = '2018-01-01'
end_date = '2024-07-20'


# Use a função download para obter os dados
df = yf.download(symbol, start=start_date, end=end_date)


df = df.stack(level=1, future_stack=True).reset_index()
df.columns = ['Date', 'Symbol', 'Close', 'High', 'Low', 'Open', 'Volume']
df.drop(columns=['Symbol'], inplace=True)

df.dropna(inplace=True)

df.to_csv('data/GOOG.csv', index=False)