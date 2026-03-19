# Pipelines do Mine-Tracker 🛠️

O projeto é construído sobre o framework **Kedro**, organizando o fluxo de dados em três pipelines modulares.

---

## 🏗️ 1. Pipeline: `mine` (Coleta e Engenharia)
Responsável por buscar dados externos e transformá-los em um formato pronto para Machine Learning.

- **`carregar_dados`**:
    - Baixa CSVs históricos mensais do `minetrack.me`.
    - Realiza merge linear em disco para economizar RAM em datasets massivos.
- **`gerar_features`**:
    - **Temporal**: Hora, dia, final de semana, ciclos de demanda (noite, tarde, etc).
    - **Tendência**: Média móvel de 10 e 30 janelas, desvio padrão móvel.
    - **Variação**: Quantidade de novos jogadores versus período anterior e variação percentual.
    - **Contextual**: `total_jogadores` na rede e `proporcao_rede` do servidor atual.
    - **Eventos**: Identifica picos (`flag_pico`) e quedas abruptas (`queda_abrupta`).

---

## 🧠 2. Pipeline: `model` (Modelagem Preditiva)
Processo de treinamento e seleção do melhor modelo para predição de carga.

- **`load_data` & `preprocess_data`**: Filtra o servidor mais relevante e limpa outliers (Winsorização de 1% e 99%).
- **`treinar_modelos_incremental`**:
    - O grande diferencial: treina modelos em **fases (chunks)** de 50.000 linhas.
    - **SGDRegressor**: Usa `partial_fit` para aprendizado contínuo.
    - **RandomForest**: Usa `warm_start` para adicionar árvores progressivamente, mantendo o consumo de memória controlado.
- **`avaliar_modelos`**:
    - Compara resultados via $R^2$ score e Erro Absoluto Médio (MAE).
    - Seleciona automaticamente o melhor entre os dois modelos para a fase de inferência.

---

## 📈 3. Pipeline: `inference` (Operação e Resultados)
Transforma os modelos treinados em inteligência acionável no dia a dia.

- **`inferencia`**: Carrega o melhor modelo e realiza predições para novos conjuntos de dados (normalmente as últimas 24h).
- **`generate_report`**: Agrupa as predições e gera um relatório JSON detalhado contendo:
    - **Legenda**: Explicação de cada atributo usado.
    - **Classificação de Carga**: Categoriza como "Baixo", "Médio", "Alto" ou "Crítico".
    - **Ações Sugeridas**: Orientações automáticas para a equipe de infraestrutura.
    - **Ranking**: Quais instâncias de servidores estão sob maior pressão.

---

## 📐 Glossário de Features (Variáveis)

| Atributo | Descrição |
| :--- | :--- |
| `hora` | Hora do dia (0–23) |
| `final_de_semana` | Indica se é sábado ou domingo |
| `media_movel_10` | Média de jogadores nas últimas 10 observações |
| `proporcao_rede` | % de jogadores no servidor comparado ao total da rede Minecraft |
| `pct_var_jogadores`| Variação percentual de jogadores frente ao período anterior |
| `server_id` | Identificador único numérico do servidor |
