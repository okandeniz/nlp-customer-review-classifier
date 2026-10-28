# Customer Review Classification | NLP & Machine Learning

An end-to-end Natural Language Processing (NLP) project that predicts whether a customer **recommends a clothing product** based on the written review.

The project covers exploratory data analysis, text preprocessing, classical machine learning, TensorFlow deep learning, Optuna hyperparameter tuning, model evaluation, and a working **FastAPI + Streamlit** application.

> **Task:** Product recommendation classification, not general sentiment analysis. A customer can mention a product problem and still recommend it.

## Project highlights

- **Dataset:** [Women's E-Commerce Clothing Reviews (Kaggle)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- **Input:** Customer review text (`Review Text`)
- **Target:** `Recommended IND` — `0` = Not Recommended; `1` = Recommended
- **Final model:** TF-IDF + class-weighted Logistic Regression
- **Decision threshold:** `0.40` for the Recommended class
- **Test accuracy:** **90.02%**
- **Test Macro F1:** **0.8357**
- **Test Not Recommended recall:** **75.61%**
- **Deployment:** FastAPI prediction service and Streamlit user interface

## Application architecture

```text
Customer review
      |
      v
Streamlit app (app.py)
      | HTTP POST /predict
      v
FastAPI backend (src/api/main.py)
      |
      v
Saved scikit-learn Pipeline
(TF-IDF + Balanced Logistic Regression)
      |
      v
Predicted class + estimated class probabilities
```

The API loads the saved pipeline and its metadata when the server starts. The Streamlit app sends requests to the API instead of loading the model directly.

## Dataset and preparation

The original Kaggle CSV contains **23,486 reviews**. The project uses only review text as the model input; rating and other post-review fields are **not** used as predictors.

Preparation steps:

1. Remove rows without review text.
2. Keep `review_text` and the binary `target` label.
3. Check class imbalance and review-length distributions.
4. Explore unigram and bigram usage for both classes.
5. Remove extra duplicate review texts after normalization.
6. Split the cleaned data into **80% training**, **10% validation**, and **10% test**, with stratification.

After cleaning, **22,634 reviews** remain: 18,107 train, 2,263 validation, and 2,264 test. About 82% of the reviews belong to the Recommended class.

**Leakage prevention:** TF-IDF is fitted only on the training set. TensorFlow vocabulary layers are also adapted only on the appropriate training split. The test set is kept separate during model selection and tuning.

## Model experiments

The following results are **validation** Macro F1 scores after selecting a classification threshold for each model:

| Model | Selected threshold | Validation Macro F1 |
| --- | ---: | ---: |
| **TF-IDF + Balanced Logistic Regression** | **0.400** | **0.8303** |
| Bidirectional LSTM | 0.750 | 0.8289 |
| Dense Neural Network | 0.600 | 0.8248 |
| LSTM | 0.600 | 0.8233 |
| Conv1D | 0.525 | 0.8228 |
| Optuna-tuned Dense | 0.725 | 0.8200 |
| Optuna-tuned BiLSTM | 0.625 | 0.8181 |
| GRU | 0.725 | 0.8138 |

Experiments used TensorFlow/Keras with trainable embeddings, EarlyStopping, and Optuna searches for selected neural networks. Several deep learning models showed early overfitting. Smaller models reduced parameter counts, but did not improve validation Macro F1 over the selected Logistic Regression baseline.

**Why Logistic Regression?** It gave the highest observed validation Macro F1, is comparatively simple, and supports efficient inference. Small score differences between the leading models should not be treated as proof of statistical superiority.

## Final test results

The chosen model and its `0.40` threshold were fixed before the final test evaluation.

| Metric | Test result |
| --- | ---: |
| Accuracy | 0.9002 |
| Macro F1 | 0.8357 |
| Not Recommended recall | 0.7561 |
| Test samples | 2,264 |

Confusion matrix (rows are actual labels; columns are predicted labels):

| Actual / Predicted | Not Recommended (0) | Recommended (1) |
| --- | ---: | ---: |
| Not Recommended (0) | 310 | 100 |
| Recommended (1) | 126 | 1,728 |

The model produces **estimated probabilities**, not guaranteed confidence levels. Results may change for reviews from other platforms or product categories.

## Tech stack

- **Language:** Python 3.10
- **Data and ML:** pandas, NumPy, scikit-learn, matplotlib
- **Deep learning:** TensorFlow 2.10.1 / Keras
- **Tuning:** Optuna
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit, Requests
- **Testing:** Pytest, HTTPX
- **Environment:** VS Code and Windows; NVIDIA GPU used for TensorFlow experiments

> TensorFlow 2.10 is used in this project's tested native-Windows GPU environment. A different TensorFlow version or operating system may need different GPU setup steps. The deployed Logistic Regression pipeline does **not** require a GPU.

## Repository structure

```text
nlp-customer-review-classifier/
├── .streamlit/
│   └── config.toml
├── data/                       # Local dataset files (not committed)
├── models/
│   ├── customer_review_classifier.joblib
│   └── model_metadata.json
├── notebooks/                  # EDA and modeling notebooks
├── src/
│   ├── __init__.py
│   └── api/
│       ├── __init__.py
│       └── main.py
├── tests/
│   └── test_api.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Notebook file names and optional folders may vary in a local copy.

## Run locally (Windows / VS Code)

### 1. Get the project and create an environment

```powershell
git clone https://github.com/<YOUR_USERNAME>/nlp-customer-review-classifier.git
cd nlp-customer-review-classifier

conda create -n tf-nlp-gpu python=3.10 -y
conda activate tf-nlp-gpu
python -m pip install -r requirements.txt
```

If you already have the working `tf-nlp-gpu` environment, activate it instead of recreating it.

### 2. Get the model files

The API needs both files below:

```text
models/customer_review_classifier.joblib
models/model_metadata.json
```

These files are intended to be tracked in this repository if the model artifact is within GitHub's file limits. If the files are not present, export them from the training notebook before starting the API. Only load model files from sources you trust: `joblib` deserialization can execute Python code.

### 3. Start FastAPI

From the project root, run:

```powershell
python -m uvicorn src.api.main:app --reload
```

- Health endpoint: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

Example request for `POST /predict`:

```json
{
  "review_text": "I really love this dress."
}
```

Example response from the saved final model:

```json
{
  "prediction": 1,
  "label": "Recommended",
  "probability_recommended": 0.9488,
  "probability_not_recommended": 0.0512,
  "threshold": 0.4
}
```

### 4. Start Streamlit

Open **another terminal** in the project root:

```powershell
conda activate tf-nlp-gpu
python -m streamlit run app.py
```

Open http://localhost:8501, enter a review, and select **Analyze Review**.

The default API address is `http://127.0.0.1:8000`. To change it, set `REVIEW_API_URL` before starting Streamlit.

### 5. Run automated tests

```powershell
python -m pytest tests/test_api.py -v
```

The project includes six tests covering API health, prediction output, and invalid inputs. All six passed in the development environment.

## Streamlit design palette

| Use | Color |
| --- | --- |
| Primary blue | `#689EB8` |
| Page background | `#F3F4F6` |
| Neutral gray | `#A79B94` |
| Coral accent | `#FF5A60` |

The theme is configured in `.streamlit/config.toml`.

## Limitations and future improvements

- Classes are imbalanced, so accuracy alone does not describe model quality.
- The task predicts a customer's **recommendation**, which can differ from the emotional tone of the text.
- Some reviews contain both positive and negative statements.
- Estimated probabilities have not been formally calibrated.
- Domain shift to other stores, categories, or languages may reduce performance.
- Future work could include calibration, model monitoring, and further evaluation on external data.

## References

- Dataset: [Women's E-Commerce Clothing Reviews — Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- Learning reference: [mrdbourke/tensorflow-deep-learning](https://github.com/mrdbourke/tensorflow-deep-learning)

---

*Built as an end-to-end data science portfolio project.*
