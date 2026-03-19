# Problemas Resolvidos pelo Mine-Tracker 🛠️

Este sistema foi projetado para resolver três problemas fundamentais no gerenciamento de redes de servidores Minecraft:

## 1. Monitoramento Ineficiente e Reativo ⚠️
Tradicionalmente, administradores reagem a picos de carga *depois* que o servidor começa a apresentar lag ou queda de desempenho. O **Mine-Tracker** muda essa abordagem de **reativa** para **proativa**.

- **Problema**: Incapacidade de prever quando a carga será alta.
- **Solução**: Modelagem preditiva que antecipa a quantidade de jogadores em janelas específicas.

## 2. Desafios de Custo e Recursos (Autoscaling) 💸
Manter servidores com alta capacidade 24/7 é caro, e servidores com baixa capacidade durante picos afastam jogadores.

- **Problema**: Desperdício de recursos em horários de baixa (madrugada) e sobrecarga em horários de pico (noite/fim de semana).
- **Solução**: Predição de horários de pico para orientar o **autoscaling** de infraestrutura (aumentar/reduzir hardware conforme a demanda prevista).

## 3. Planejamento de Manutenção sem Interrupção 🛠️
Realizar manutenções em horários de pico causa perda de jogadores e impacto negativo na reputação.

- **Problema**: Falta de visibilidade sobre os melhores momentos para janelas de manutenção.
- **Solução**: Identificação de janelas de "baixa carga" (Ex: 새벽/Madrugada) com base em padrões históricos e tendências semanais.

## 4. Análise de Popularidade Relativa 📊
Entender como o seu servidor se compara ao resto da rede (Java vs Bedrock).

- **Problema**: Dificuldade em saber se uma queda de jogadores é um problema interno ou uma tendência geral de mercado.
- **Solução**: Feature de `proporcao_rede` que normaliza a contagem de jogadores em relação ao tráfego total da plataforma, ajudando a identificar se o servidor está ganhando ou perdendo fatia de mercado (market share).
