# Imagem base oficial do Python
FROM python:3.11-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código para dentro do container
COPY . .

# Comando para rodar a aplicação
CMD ["python", "src/main.py"]