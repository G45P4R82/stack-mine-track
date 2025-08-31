FROM python:3.12-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY mine-tracker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código completo
COPY . .

# Define o diretório de trabalho no projeto Kedro
WORKDIR /app/mine-tracker

# Expõe porta para kedro-viz
EXPOSE 8000

# Define entrypoint padrão, permitindo passar comandos no `docker run`
ENTRYPOINT ["kedro"]
CMD ["run"]
