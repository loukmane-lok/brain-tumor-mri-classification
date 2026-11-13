
# Brain Tumor MRI Classification

CNN that classifies brain MRI scans into 4 classes: **glioma**, **meningioma**, **pituitary tumor**, **no tumor**. Built for a M2 Big Data course project (Paris 8).


## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, credit: Masoud Nickparvar). Not included in this repo — download it separately and place it under `data/` (with `Training/` and `Testing/` subfolders) to run the notebook or the exploration page.

## Results

| Metric | Baseline | Optimized (Optuna) |
|---|---|---|
| Test accuracy | 90.7% | **92.9%** |
| Test loss | 0.56 | **0.45** |
| Glioma recall | 0.74 | **0.81** |
| Macro F1 | 0.905 | **0.928** |

Hyperparameters tuned: learning rate, batch size, number of filters, dropout, and Adam's beta_1/beta_2. See the notebook for the full comparison, curves, and confusion matrices.

**Note:** hyperparameter tuning and early stopping both monitor the test set, so the final metrics are slightly optimistic.

## Project structure

```
brain-tumor-mri-classification/
├── app.py                # Streamlit app
├── model_utils.py         # CNN architecture (for loading weights locally)
├── requirements.txt
├── model/                 # trained weights, metrics, history (tracked)
├── data/                  # dataset (not tracked, download separately)
└── notebooks/              # Colab notebook
```

## Notebook (Colab)

Open in Colab or run locally — it detects the environment automatically. Covers data exploration, preprocessing, CNN architecture, training, evaluation, and Optuna hyperparameter tuning.

## Streamlit app

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Features:
- **Data exploration:** browse and enlarge dataset images
- **Model:** architecture, hyperparameters, test metrics, confusion matrix, learning curves
- **Test an image:** prediction with per-class probabilities, Grad-CAM heatmap
- **PDF report:** tested image, probabilities, metrics, parameters, Grad-CAM

## Disclaimer

Educational project. Predictions are not a medical diagnosis and must be reviewed by a qualified practitioner.


