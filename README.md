# cox-models-sudden-death
Modelos de Aprendizado de Máquina para Previsão de Morte Súbita Cardíaca
# Modelos de Aprendizado de Máquina para Previsão de Morte Súbita Cardíaca

Este repositório reúne o trabalho da minha linha de pesquisa de mestrado, focada no desenvolvimento e avaliação de modelos preditivos para ** morte súbita cardíaca(Sudden Cardiac Death - SCD) ** a partir de dados clínicos e sinais fisiológicos, como os do dataset MUSIC.

# 🎯 Objetivo

Investigar e adaptar modelos de análise de sobrevivência, especialmente a ** Regressão de Cox ** (tradicional e variantes modernas), para prever o risco individual de SCD em pacientes com histórico de parada cardíaca ou alto risco cardiovascular.

# 🔍 Contexto
Tem quatro pastas principais:
    - 01_Dataset: que não esta disponivel no repositório, pelo o grande volume de dados mas é necessário para rodar os scripts. 
    - 02_Prepocessamento_filtro: Onde vou tratar de limpar os sinais ECG, e aplicar filtros para remover ruídos. passar por uma etapa de validação e depois padronização dos dados.
    - 03_Modelos: Onde vou aplicar os modelos de regressão de Cox, e outros modelos de aprendizado de máquina para prever a morte súbita cardíaca.
    - 04_Painel_e_logs: Onde vou gerar os logs dos modelos, e os painéis de visualização dos resultados. Esta pasta é mais para o uso pessoal e não é necessário para rodar os scripts. a ideia é guardar o log dos modelos e os resultados dos experimentos.

## 📦 Dados

Os experimentos foram conduzidos com o **MUSIC Dataset**, um conjunto de dados clínicos robusto voltado para estudos de SCD. Por razões de tamanho e licença, o dataset não está incluído neste repositório.

 Link do dataset 
 - [Physionet music](https://physionet.org/content/music-sudden-cardiac-death/1.0.1/)
 
 
## 🧠 Metodologia

Foram explorados:
- Modelos clássicos de regressão de Cox
- Modelos adaptativos baseados em **deep learning**, utilizando a biblioteca `pycox`
- Estratégias de avaliação com **curvas de sobrevivência**, **concordância (C-index)** e **análise de risco cumulativo**

## 🛠️ Tecnologias Utilizadas
- Neurokit
- Python
- PyTorch
- Pycox
- Lifelines
- Scikit-learn
- Pandas / Numpy / Matplotlib / Seaborn

## 📚 Referências

- [PyCox Documentation](https://github.com/havakv/pycox)
- Goldberger et al., 2000. *PhysioNet*
- Harrell, F. E., 2015. *Regression Modeling Strategies*

## ⚠️ Observações

O repositório não inclui o dataset original por limitações de tamanho e licenciamento. Para fins de reprodutibilidade, scripts e pipelines estão estruturados para receber os dados no formato `.csv` compatível com o MUSIC.

---

🔬 *Este projeto está em andamento como parte da minha pesquisa de mestrado. Feedbacks e colaborações são muito bem-vindos!*

## 📧 gmail
Para entrar em contato, envie um e-mail para: [danieloliveirafff@gmail.com]

