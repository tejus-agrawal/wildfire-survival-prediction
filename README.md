# Wildfire Survival Prediction - Palisades SAM Model

## What it Does

This project leverages satellite imagery and machine learning to predict wildfire survival for homes. It uses the **Segment Anything Model (SAM)** for semantic segmentation of satellite imagery to extract features like structure area, vegetation density, and defensible space. These features are then used to train and compare **XGBoost** and **Deep Learning (MLP/CNN)** models. A comprehensive interactive dashboard allows for risk analysis and comparison of model predictions against ground truth data.

## Quick Start

1. **Install dependencies** (see [SETUP.md](SETUP.md) for detailed instructions):
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data** from the [Google Drive folder](https://drive.google.com/drive/folders/1-3cwA9ihXukHSiez9MXFWmQ5-lxJ1lBL?usp=sharing) and place it in the `data/` directory.

3. **Run the dashboard**:
   ```bash
   cd app
   streamlit run dashboard.py
   ```

For complete setup instructions, data processing workflows, and training procedures, see [SETUP.md](SETUP.md).

## Video Links

<!-- Add your demo video link here -->
<!-- Add your technical walkthrough video link here -->

## Evaluation

### Model Performance

The models were evaluated on a held-out test set with the following results:

**XGBoost (Baseline)**
- Accuracy: **63.28%**
- ROC-AUC: **0.6852**

**MLP with Focal Loss**
- Accuracy: **59.40%**
- ROC-AUC: **0.6288**

**Best Model**: XGBoost achieved superior performance on the test set, with an AUC score of 0.6852 compared to the MLP's 0.6288. The XGBoost model was selected as the final model for deployment.

### Model Comparison

The XGBoost model outperformed the deep learning MLP approach, demonstrating that gradient boosting on engineered tabular features (structure area, vegetation density, defensible space, and their interactions) was more effective for this wildfire survival prediction task than the neural network architecture tested.
