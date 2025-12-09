# Setup Instructions

## Prerequisites
- Python 3.9+
- Git

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd wildfire-survival-prediction
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with `segment-anything`, install it directly from git:*
    ```bash
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

## Data Access

**Large files (Images, Mask Tensors, and Raw Datasets) are hosted externally.**

> **https://drive.google.com/drive/folders/1-3cwA9ihXukHSiez9MXFWmQ5-lxJ1lBL?usp=sharing**

Please download the contents and place them in the `data/` directory following this structure:
```
data/
├── raw/
│   └── dins_raw.csv          # Original California DINS dataset
├── images/                   # Satellite imagery (~20k images)
├── mask_tensors/             # Pre-computed SAM mask tensors
└── processed/
    └── clean_homes.csv       # Cleaned metadata
```

## Running the Pipeline

### A. Jupyter Notebooks (Data Processing & Training)
Start the Jupyter Notebook server:
```bash
jupyter notebook
```
Execute the notebooks in the following order:
1. **`notebooks/data_cleaning.ipynb`**: Cleans raw DINS data and filters for residential structures. (Used to create the clean_homes.csv)
2. **`notebooks/esri_home_footprints.ipynb`**: Downloads historical pre-fire satellite imagery using the Esri Wayback API. (Used to generate the images in data)
3. **`notebooks/sam_extraction.ipynb`**: Uses SAM to extract masks (Structure, Tree, Grass) and computes physics-based features. (Used to generate the final_dataset.csv and mask_tensors files)
4. **`notebooks/mlp_training.ipynb`**: (Optional) Trains and compares XGBoost vs. MLP (Focal Loss) models on tabular features.
5. **`notebooks/cnn_training.ipynb`**: Trains a ResNet-18 model directly on semantic image stacks.
6. **`notebooks/app_preparation.ipynb`**: Prepares the final dataset for the dashboard.

### B. Interactive Dashboard
Run the Streamlit app to visualize results:
```bash
cd app
streamlit run dashboard.py
```

## Project Structure
- `data/`: Datasets and imagery.
- `models/`: Trained model weights (`best_resnet.pth`, `best_model.json`).
- `notebooks/`: Workflow notebooks.
- `app/`: Streamlit dashboard code (`dashboard.py`).
- `src/`: Core Python modules:
    - `acquisition.py`: Esri Wayback imagery downloading logic.
    - `data.py`: PyTorch Dataset classes.
    - `features.py`: Feature engineering and SAM mask processing.
    - `models.py`: PyTorch model definitions (ResNet, MLP, FocalLoss).
    - `utils.py`: Hardware acceleration and helper functions.

