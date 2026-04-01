# Arquitetura do Projeto: OCR + Tradução Multilingue

## Visão Geral
Este projecto permite receber **imagens, PDFs ou texto** em até **48 línguas** e produzir a respetiva tradução para **Português Europeu (PT-PT)**, funcionando **100% offline**.

Disponibiliza duas interfaces:
- **CLI (terminal)** - para utilização via linha de comandos  
- **GUI (Streamlit)** - interface web acessível no browser  

## Pipeline de Processamento
Entrada -> Pré-processamento -> OCR -> Limpeza -> Detecção de língua -> Routing -> Tradução -> Pós-processamento -> Saída

## Execução

### Interface gráfica
python -m streamlit run app.py

### Interface CLI
python cli.py --help
