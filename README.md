# Wildfire Survival Prediction - Palisades SAM Model

## Project Overview
This project uses satellite imagery and machine learning to predict wildfire survival for homes. It leverages the Segment Anything Model (SAM) for feature extraction and compares performance between XGBoost and Deep Learning approaches.

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd wildfire-survival-prediction
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with `segment-anything`, install it directly from git:*
    ```bash
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

### 3. Running the Notebooks
Start the Jupyter Notebook server:
```bash
jupyter notebook
```

Execute the notebooks in the following order:
1.  **`notebooks/data_cleaning.ipynb`**: Cleans raw DINS data and downloads satellite imagery.
2.  **`notebooks/sam_extraction.ipynb`**: Uses SAM to extract masks and features (tree/structure area) from images.
3.  **`notebooks/mlp_training.ipynb`**: Trains and compares XGBoost vs. MLP models on physics-based features.
4.  **`notebooks/cnn_training.ipynb`**: (Optional) Trains a ResNet-18 model directly on images.
5.  **`notebooks/app_preparation.ipynb`**: Prepares artifacts for the deployment app.

## Project Structure
- `data/`: Contains raw CSVs, processed datasets, and images.
- `models/`: Stores trained model weights (e.g., `best_model.pth`, `best_model.json`).
- `notebooks/`: Jupyter notebooks for the pipeline.
- `src/`: Shared utility scripts (if any).
- `app/`: Application code (dashboard.py).

