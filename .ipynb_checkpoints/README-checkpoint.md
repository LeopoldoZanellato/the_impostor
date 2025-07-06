# 🕵️‍♂️ Fake or Real - The Impostor Hunt

Este projeto participa da competição do Kaggle **"Fake or Real: The Impostor Hunt"**, com o objetivo de detectar qual dos dois textos é o verdadeiro em cada par de artigos.

---

## 📁 Estrutura dos Dados

- Cada pasta `article_xxxx` contém dois arquivos:
  - `file_1.txt` e `file_2.txt`
- Apenas um dos dois textos é verdadeiro (`label=1`), o outro é falso (`label=0`)

---

## 📊 Abordagem

### 🔧 1. Pré-processamento

- Leitura e extração dos textos `.txt`
- Criação de features manuais:
  - `texto_len`
  - `stopword_ratio`
  - `entropia`
  - `avg_sentence_len`
  - (testadas: `long_words`, `non_latin_chars`, `num_linhas`, etc.)

### 🧪 2. Vetorização

- Vetorização do texto usando **TF-IDF** (`max_features=500`)
- Normalização das features com `StandardScaler`
- Combinação das features com `scipy.sparse.hstack`

### 🧠 3. Modelos Testados

- `RandomForestClassifier`
- `XGBClassifier`
- `SVC` (com `GridSearchCV`)
- `LogisticRegression`

> ✅ Melhor desempenho obtido com **SVM com kernel RBF** ajustado via `GridSearchCV`.

### 🔁 4. Avaliação

- Validação cruzada com `StratifiedKFold (n_splits=5)`
- Métricas:
  - `Accuracy`
  - `Precision`
  - `Recall`
  - `F1-score`
- Também foi calculada a **Pairwise Accuracy**: acerto dentro de cada par de arquivos.

---

## 📤 Submissão

- Usa a saída do `predict_proba()` do melhor modelo
- Para cada par (`article`), seleciona o `file_id` com maior probabilidade
- Gera arquivo `submission.csv` com:
  - `id`: índice da submissão
  - `real_text_id`: texto escolhido como verdadeiro

---

## 📦 Estrutura do Projeto
the_impostor/
├── data/ # Contém os dados de treino e teste
├── notebooks/ # Notebooks utilizados durante o desenvolvimento
├── submission.ipynb # ✅ Notebook principal com o pipeline final
├── submission.csv # Arquivo de submissão final
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo

---

## ✅ Requisitos
- Python 3.8+
- Bibliotecas:
  - `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `nltk`, `scipy`

```bash
pip install -r requirements.txt

