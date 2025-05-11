# IMDb Review Classifier

## Descriere proiect
Acest proiect își propune să clasifice recenziile de film de pe IMDb în două categorii: **Pozitiv** și **Negativ**. Se utilizează atât un model de bază (Logistic Regression pe TF-IDF), cât și un model avansat (DistilBERT fine-tuned) pentru analiza de sentiment.

## Obiective
- Automatizarea analizei de sentiment pentru recenzii de film
- Obținerea unei acurateți ≥ 85% și F1-score ≥ 0.85
- Cod clar, modular, ușor de extins și de înțeles

## Tehnologii folosite
- Python ≥ 3.8
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- Jupyter Notebook
- Matplotlib, Seaborn (vizualizare)

## Structură proiect
```
.
├── data/           # Date brute și modele salvate
├── notebooks/      # Notebook-uri pentru EDA, baseline, fine-tuning
├── reports/        # Rapoarte, grafice, rezultate
├── src/            # Cod sursă modular (preprocesare, training, evaluare)
├── requirements.txt
└── README.md
```

## Instalare și setup
1. Clonează acest repository:
   ```bash
   git clone https://github.com/LiviuTcaci/imdb-review-classifier.git 
   cd imdb-review-classifier
   ```
2. Creează un mediu virtual și activează-l:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Instalează toate dependințele:
   ```bash
   pip install -r requirements.txt
   ```
4. Pornește Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Cum rulezi codul
- Pentru explorarea datelor și experimentare, folosește notebook-urile din `notebooks/`.
- Pentru rularea pipeline-ului complet, folosește scripturile din `src/`:
  - `src/preprocess.py` — preprocesare text
  - `src/train.py` — antrenare model
  - `src/evaluate.py` — evaluare model

## Exemple de utilizare
- Rulează EDA:
  - Deschide `notebooks/01_eda.ipynb` și execută celulele pentru a explora datele.
- Rulează baseline:
  - Deschide `notebooks/02_baseline.ipynb` sau folosește `src/train.py` cu opțiunea baseline.
- Rulează fine-tuning:
  - Deschide `notebooks/03_fine_tuning.ipynb` sau folosește `src/train.py` cu opțiunea transformer.

## Resurse utile
- [Documentație Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Documentație Transformers](https://huggingface.co/docs/transformers/)
- [Documentație PyTorch](https://pytorch.org/docs/stable/index.html)