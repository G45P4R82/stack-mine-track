# Arquitetura Técnica do Mine-Tracker 🏗️

O sistema segue uma arquitetura de pipeline de dados desacoplada, utilizando o **Kedro** como espinha dorsal. Esse design permite escalabilidade, separação de preocupações e fácil manutenção.

## 🧱 Componentes Principais

### 1. Camada de Ingestão e Processamento (`mine`)
- **Extração**: Coleta assíncrona/linear de arquivos CSV do `minetrack.me`.
- **Transformação**: Feature Engineering voltada para séries temporais.
- **Armazenamento Intermediário**: Os dados são salvos em `data/04_feature/` (Catálogo Kedro).

### 2. Camada de Inteligência (`model`)
- **Desenvolvido com Scikit-Learn**.
- **Treinamento Incremental**: Solução de carregamento parcial (`partial_fit` e `warm_start`) para lidar com grandes volumes de dados sem estourar a memória RAM (limite de ~4GB configurado).
- **Seleção de Modelos**: Automática via métricas de desempenho ($R^2$ e MAE).

### 3. Camada de Aplicação e Entrega (`inference`)
- **Inferência Offline/Batch**: Gera previsões diárias ou semanais.
- **Relatórios Automatizados**: Exporta o resultado em `data/08_reporting/report_inference.json`.
- **Lógica de Negócio**: Transforma números em classes de carga (`baixo`, `médio`, `alto` e `crítico`).

---

## 🗺️ Diagrama de Fluxo de Dados (Data Flow)

```mermaid
graph LR
    subgraph Coleta
        A1[minetrack API] --> A2[CSV Brutos]
    end
    
    subgraph Processamento
        A2 --> B1[Feature Engineering]
        B1 --> B2[Matriz de Atributos]
    end
    
    subgraph Treinamento
        B2 --> C1[SGD Incremental]
        B2 --> C2[RandomForest Incr.]
        C1 & C2 --> D1[Avaliador]
        D1 --> D2[Melhor Modelo (.pkl)]
    end
    
    subgraph Inferência
        D2 --> E1[Inferencia de Ontem]
        E1 --> E2[Dashboard / Report JSON]
    end
```

## 🔐 Decisões de Design (Trade-offs)

- **Por que SGDRegressor?** Baixo consumo de memória e capacidade de `partial_fit` para datasets massivos.
- **Por que RandomForest com warm_start?** Melhor captura de relacionamentos não lineares, mas exige controle rigoroso do número de árvores para não exceder a RAM.
- **Por que Kedro?** Garante reprodutibilidade, separação clara entre código e configuração (`conf/`), e uma estrutura de dados robusta (`data/01_raw`, `data/04_feature`, etc).
