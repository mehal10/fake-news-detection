# Fake News Detection and News Sentiment Analysis System

Research-oriented Django web application that classifies news-style text as **Real / Fake** and **Positive / Negative / Neutral** using classical NLP and machine learning (scikit-learn, NLTK, Gensim). No React, Node, Firebase, or paid APIs.

## Features

- Landing page and educational **About** section
- **Dashboard** with live SQLite statistics and Chart.js charts
- **Dataset management** — CSV upload, flexible column mapping, validation
- **NLP preprocessing** demo (clean, tokenize, stem, lemmatize)
- **Fake news detection** — Multinomial Naïve Bayes, Logistic Regression, Word2Vec + LR
- **Sentiment analysis** — 3-class Logistic Regression / Naïve Bayes
- **Combined analysis** — both pipelines on one article
- **Analysis history** — search, filters, sort, pagination, detail, delete, export
- **Reports & analytics** — trends, model evaluation tables, CSV / PDF export

Predictions always come from trained Joblib models. Confidence scores are model estimates, not verified factual accuracy.

## Project structure

```
manage.py
requirements.txt
news_analysis/          # settings, root URLs
core/                   # landing, about
dashboard/
detector/
sentiment/
combined_analysis/
history/
analytics/
ml_engine/
  models.py             # Dataset metadata
  services/             # preprocessing, trainers, predictors, registry
templates/
static/css|js/
media/datasets/         # uploaded CSVs
media/models/           # trained model files + registry_index.json
```

## Requirements

- Python 3.10+
- See `requirements.txt` (Django, pandas, numpy, scikit-learn, nltk, gensim, joblib, reportlab)

## Setup (Windows)

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"

python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/

Optional admin:

```bat
python manage.py createsuperuser
```

## Setup (Linux / macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
python manage.py migrate
python manage.py runserver
```

## Dataset setup

1. Open **ML Engine → Dataset Management → Upload**.
2. Upload a **CSV** (UTF-8) with a text column and a label column.
3. Map columns if auto-detect is wrong.
4. Status should become **validated** / **ready**.

### Fake news labels

Accepts values such as `real`/`fake`, `0`/`1`, `true`/`false` (mapped consistently). Unknown labels are **not** assumed to be real.

Public example corpus: [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — document the source when you import it.

### Sentiment labels

Use `positive` / `negative` / `neutral` (or common synonyms handled by the validator).

Sample files may exist under `media/datasets/` for smoke tests only; train on larger labelled data for meaningful accuracy.

## Model training

1. Open **ML Engine → Model Training**.
2. Choose task tab (**Fake News** or **Sentiment**).
3. Select dataset, algorithm, test split, feature settings.
4. Start training. Artifacts are stored under `media/models/<model_id>/` and registered in `registry_index.json`.
5. Activate the preferred model version.

Training fits vectorizers **only on the training split** (no test leakage). Evaluation metrics are computed on the held-out test split and stored with the model metadata.

## Running predictions

| Page | URL |
|------|-----|
| Fake News Detection | `/detector/` |
| Sentiment Analysis | `/sentiment/` |
| Combined Analysis | `/combined/` |

Provide headline and/or body text (or a `.txt` upload). Results show labels, probabilities, model name, and timing. Successful runs are saved to **Analysis History**.

## Testing

```bat
python manage.py check
python manage.py test
```

View and form tests mock predictors where needed so the suite does not require large trained models. Service-level preprocessing tests exercise empty text, URLs, HTML, and negation handling.

## Production notes

| Setting | Guidance |
|---------|----------|
| `DJANGO_DEBUG` | Set to `0` |
| `DJANGO_SECRET_KEY` | Strong random secret |
| `DJANGO_ALLOWED_HOSTS` | Your hostname(s) |
| `DJANGO_SSL` | `1` behind HTTPS for secure cookies |

```bat
set DJANGO_DEBUG=0
set DJANGO_SECRET_KEY=your-long-random-string
set DJANGO_ALLOWED_HOSTS=example.com
python manage.py collectstatic
```

Do not serve with `runserver` in production. Use a WSGI server (e.g. gunicorn) behind a reverse proxy. Media uploads accept **CSV datasets only** (extension, size ≤ 25 MB, basic binary rejection). Uploaded files are never executed.

## Architecture (summary)

```
Browser (HTML/CSS/JS/Chart.js)
    → Django views / templates
        → ml_engine services (preprocess → features → predict)
            → Joblib models on disk
        → SQLite (history, dataset metadata)
        → Dashboard / Analytics aggregations
```

Separate **fake news** and **sentiment** pipelines; combined analysis runs both and stores **one** history row.

## Known limitations

- Quality depends entirely on labelled training data size and domain match.
- Classical models (NB / LR / mean Word2Vec) will not match large transformer systems.
- Sentiment is linguistic polarity, not credibility.
- Fake/Real is pattern matching against training labels, not independent fact-checking.
- Small demo datasets produce unstable metrics; treat them as illustrative.
- Predictor cache is per-process (suitable for development / single worker).

## License / academic use

Intended for education and research prototypes. Replace team placeholders on the About page with your institution details.

## Bundled trained models (this package)

| Task | Model | Test accuracy | Training data |
|------|--------|---------------|---------------|
| Fake news | Logistic Regression + TF-IDF (active) | **~94%** | Fake or Real News (~6,200 articles) |
| Fake news | Naïve Bayes + TF-IDF | **~91%** | same |
| Sentiment | Logistic Regression + TF-IDF (active) | **~88%** | Movie reviews + neutral (balanced ~3,200) |
| Sentiment | Naïve Bayes + TF-IDF | **~92%** | (unbalanced 3-class set) |

**Domain note:** Fake-news models were trained on English **political news** style text. Short product reviews or highly technical science blurbs may score less reliably than news articles. Sentiment models lean on review-style language for positive/negative.

Datasets are under `media/datasets/`. You can retrain anytime from **Model Training**.
