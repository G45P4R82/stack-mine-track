# Mine Tracker - Dashboard de Monitoramento

Sistema completo de visualização e análise de dados de mineração com interface web moderna e responsiva.

## 🚀 Funcionalidades

### Dashboard Completo
- **Visualização em Tempo Real**: Métricas e estatísticas dos clusters de mineração
- **Gráficos Interativos**: 6 tipos diferentes de visualizações usando Chart.js
- **Tabelas Detalhadas**: Dados completos dos clusters e instâncias
- **Ranking de Performance**: Classificação dos clusters por performance
- **Animações Suaves**: Interface animada com anime.js
- **Design Responsivo**: Funciona perfeitamente em desktop, tablet e mobile

### Dados Visualizados
- **Clusters**: 4 clusters com diferentes níveis (alto/médio)
- **Previsões**: Baseline predictions para cada cluster
- **Instâncias**: Dados detalhados por instância incluindo:
  - Hora do dia (0-23)
  - Indicador de fim de semana
  - Média móvel de jogadores
  - Proporção de rede
  - Variação percentual de jogadores

### Gráficos Disponíveis
1. **Distribuição por Nível**: Gráfico de pizza mostrando clusters alto/médio
2. **Previsões por Cluster**: Gráfico de barras com previsões ordenadas
3. **Distribuição por Horário**: Análise temporal das instâncias
4. **Correlação Hora vs Previsão**: Scatter plot de correlação
5. **Análise Temporal**: Linha do tempo das previsões
6. **Proporção de Rede**: Distribuição da proporção de rede por cluster

### Recursos Adicionais
- **Exportação**: CSV e JSON dos dados
- **Impressão**: Função de imprimir o dashboard
- **Legenda Completa**: Descrição de todos os campos
- **API REST**: Endpoint `/api/data` para acesso aos dados

## 🛠️ Tecnologias Utilizadas

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Framework CSS**: Bootstrap 4
- **Gráficos**: Chart.js
- **Animações**: Anime.js
- **Ícones**: Font Awesome 6

## 📦 Instalação e Execução

### Pré-requisitos
- Python 3.7+
- Flask

### Instalação
```bash
# Instalar Flask
pip install flask

# Navegar para o diretório da aplicação
cd app

# Executar a aplicação
python app.py
```

### Acesso
- **URL**: http://localhost:8901
- **API**: http://localhost:8901/api/data

## 📊 Estrutura dos Dados

O sistema lê dados do arquivo `report_inference.json` que contém:

```json
{
  "legend": {
    "hora": "Hora do dia (0–23)",
    "final_de_semana": "Indicador se é fim de semana (0=Não, 1=Sim)",
    "media_movel_10": "Média móvel de jogadores nas últimas 10 janelas",
    "proporcao_rede": "Proporção de jogadores no cluster em relação à rede total (0–1)",
    "pct_var_jogadores": "Variação percentual de jogadores em relação ao período anterior"
  },
  "clusters": [...],
  "ranking": [...]
}
```

## 🎨 Interface

### Design Moderno
- **Cores**: Gradientes modernos e paleta de cores profissional
- **Tipografia**: Fontes limpas e legíveis
- **Espaçamento**: Layout bem organizado com espaçamentos consistentes
- **Cards**: Elementos visuais em formato de cards com sombras suaves

### Responsividade
- **Mobile First**: Design otimizado para dispositivos móveis
- **Breakpoints**: Adaptação para diferentes tamanhos de tela
- **Touch Friendly**: Interface otimizada para toque

### Animações
- **Entrada**: Elementos aparecem com animações suaves
- **Hover**: Efeitos visuais ao passar o mouse
- **Transições**: Animações fluidas entre estados
- **Loading**: Indicadores de carregamento animados

## 🔧 Personalização

### Cores
As cores podem ser personalizadas no arquivo `base.html` na seção `:root`:

```css
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --accent-color: #e74c3c;
    --success-color: #27ae60;
    --warning-color: #f39c12;
}
```

### Gráficos
Os gráficos podem ser personalizados modificando as configurações do Chart.js no arquivo `dashboard.html`.

## 📱 Compatibilidade

- **Navegadores**: Chrome, Firefox, Safari, Edge (versões recentes)
- **Dispositivos**: Desktop, Tablet, Mobile
- **Resoluções**: 320px até 4K

## 🚀 Performance

- **Carregamento Rápido**: Otimizado para carregamento rápido
- **Animações Suaves**: 60fps nas animações
- **Responsivo**: Interface fluida em todos os dispositivos
- **Leve**: Sem dependências desnecessárias

## 📄 Licença

Este projeto é de uso interno para monitoramento de dados de mineração.

---

**Desenvolvido com ❤️ para análise de dados de mineração**
