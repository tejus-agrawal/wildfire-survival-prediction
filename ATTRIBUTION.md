# Project Attribution & Methodology

This project was developed to analyze wildfire survival factors for residential homes using advanced computer vision and machine learning techniques.

## Data Sources
- **California DINS Database**: The "Damage Inspection Network Strategy" dataset provided the ground truth for home survival (Burned vs. Survived) and structural metadata.
- **Esri Wayback Imagery**: Historical high-resolution satellite imagery (World Imagery) was programmatically acquired for ~22,000 home locations using the Esri Wayback API. We specifically targeted pre-fire imagery (circa late 2024) to analyze vegetation conditions before the event.

## Methodologies & Libraries

### 1. Computer Vision & Feature Extraction
- **Segment Anything Model (SAM)**: Meta's SAM (ViT-H/ViT-B) was used to zero-shot segment satellite images into semantic masks (Structures, Trees, Grass).
- **OpenCV & Numpy**: Used for post-processing SAM masks to calculate:
    - **Defensible Space**: Distance from structure to nearest vegetation.
    - **Vegetation Density**: Ratio of fuel (trees/grass) to lot size.
    - **Structure Compactness**: Geometric complexity of the home footprint.
- **Mercantile**: Used for tile math to stitch satellite imagery from XYZ tile servers.

### 2. Machine Learning Models
- **XGBoost**: Gradient boosted decision trees were used as the primary baseline for tabular data prediction, achieving strong performance on physics-based features.
- **PyTorch (Deep Learning)**:
    - **MLP**: A custom Multi-Layer Perceptron with **Focal Loss** was implemented to handle class imbalance.
    - **ResNet-18**: A Convolutional Neural Network (CNN) was trained on "semantic stacks" (multi-channel tensors representing structure, vegetation, and risk maps) to capture spatial patterns.

### 3. Application & Visualization
- **Streamlit**: Used to build the interactive web dashboard.
- **PyDeck**: Powered the geospatial visualizations, enabling interactive mapping of risk factors and prediction comparisons.

## Development Tools
- **Cursor (IDE)**: Development environment.
- **Gemini 3 Pro**: AI coding assistant used for refactoring, code optimization, and documentation generation.
