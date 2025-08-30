FROM python:3.12-slim

WORKDIR /app

# Instale dependências do sistema primeiro
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY mine-tracker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Use shell para mudar de diretório e rodar o Kedro
CMD cd mine-tracker && kedro run