# Stack Mine Track

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## 📋 Visão Geral

O **Stack Mine Track** é um projeto de machine learning baseado no framework Kedro que implementa um sistema de rastreamento e monitoramento de mineração. O projeto utiliza MLflow para gerenciamento de modelos e é containerizado com Docker para facilitar o deploy e execução.

## 🏗️ Arquitetura

### Estrutura do Projeto
```
stack-mine-track/
├── mine-tracker/          # Projeto Kedro principal
│   ├── src/               # Código fonte
│   ├── data/              # Dados e datasets
│   ├── notebooks/         # Jupyter notebooks
│   ├── tests/             # Testes automatizados
│   └── conf/              # Configurações
├── Dockerfile             # Containerização
├── Makefile               # Automação de tarefas
└── README.md              # Documentação
```

### Tecnologias Utilizadas
- **Python 3.12+** - Linguagem principal
- **Kedro 1.0.0** - Framework de data science
- **MLflow** - Gerenciamento de modelos
- **Docker** - Containerização
- **Scikit-learn** - Machine learning

## 🚀 Como Executar

### Pré-requisitos
- Docker instalado
- Make (opcional, mas recomendado)

### Comandos Principais

#### Build da Imagem Docker
```bash
make build
```

#### Executar Pipeline
```bash
make run
```

#### Visualizar com Kedro Viz
```bash
make viz
```
Acesse: http://localhost:8000

#### Parar Container
```bash
make stop
```

#### Ver Logs
```bash
make logs
```

### Comandos Docker Diretos
```bash
# Build
docker build -t stack-mine-track:latest .

# Executar
docker run --rm stack-mine-track:latest run

# Kedro Viz
docker run --rm -p 8000:8000 stack-mine-track:latest viz --host 0.0.0.0 --port 8000
```

## 🔧 Desenvolvimento

### Instalação Local
```bash
cd mine-tracker
pip install -r requirements.txt
```

### Executar Testes
```bash
cd mine-tracker
pytest
```

### Jupyter Notebooks
```bash
cd mine-tracker
kedro jupyter notebook
# ou
kedro jupyter lab
```

## 📦 Deploy

### Registry Harbor
O projeto está configurado para fazer push para um registry Harbor privado:

```bash
# Login no registry
make login

# Publicar imagem
make publish
```

### Configurações
- **Registry**: serverlab.lonk-chinstrap.ts.net
- **Usuário**: admin
- **Porta**: 8000 (Kedro Viz)

## 🏭 Pipeline MLflow

O projeto inclui um pipeline de deploy que:
- Publica modelos no MLflow usando API clássica
- Compatível com servidores MLflow antigos
- Salva modelos como artifacts simples
- Registra parâmetros e metadados

## 📁 Estrutura de Dados

- **Dados**: Armazenados em `mine-tracker/data/`
- **Modelos**: Gerenciados via MLflow
- **Configurações**: Centralizadas em `mine-tracker/conf/`

## 🧪 Testes

- **Cobertura**: Configurada via pytest-cov
- **Threshold**: 0% (configurável em pyproject.toml)
- **Ferramentas**: pytest, pytest-mock, pytest-cov

## 📚 Documentação

- [Documentação Kedro](https://docs.kedro.org)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto é privado e proprietário.

---

**Desenvolvido com ❤️ usando Kedro e MLflow**