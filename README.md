# AI Crop Analytics – PMFBY

An AI-based crop image analytics application developed as an internship team project for crop identification, health/damage detection, growth-stage estimation, and agricultural advisory support.

## My Contribution

I developed the crop damage/disease detection model and contributed the Streamlit frontend design used for interacting with the analytics system.

### Hybrid Damage Detection Model

- Hybrid ResNet50 + DenseNet121 architecture
- ResNet50 feature vector: 2048 dimensions
- DenseNet121 feature vector: 1024 dimensions
- Feature fusion: 3072 dimensions
- Batch normalization + SE (Squeeze-and-Excitation) channel attention
- MLP classification head: 3072 → 1024 → 512 → 38
- Trained with Cross-Entropy loss with label smoothing and AdamW
- Best validation accuracy in the documented experiment: **99.35%**
- Test accuracy in the documented experiment: **98.63%**

## Full Application

The repository keeps the complete crop analytics application together with my model contribution:

```
User Image
    ↓
Streamlit Frontend
    ↓
Hybrid ResNet50 + DenseNet121
    ↓
Crop / Disease Classification
    ↓
Health & Damage Information
    ↓
Growth Stage + Agricultural Advisory
```

The existing application workflow also contains crop metadata, severity/insurance presentation, growth-stage heuristics, confidence handling, and top-5 predictions.

## Project Structure

```
AI-Crop-Analytics-PMBY/
│
├── app/
│   └── app.py
│
├── models/
│   ├── hybrid_model.py
│   ├── hybrid_resnet_densenet_checkpoint.pth
│   ├── best_hybrid_resnet_densenet.pth
│   └── README.md
│
├── notebooks/
│   └── resnet-densenet.ipynb
│
├── legacy_models/
│   ├── crop_model.h5
│   ├── crop_model_v2.keras
│   └── README.md
│
├── docs/
│   └── MY_CONTRIBUTION.md
│
├── scripts/
│   └── test_model.py
│
├── screenshots/
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## Application Screenshots

### Upload Interface

![Upload Interface](screenshots/upload_interface.png)

### Analysis Result

![Analysis Result](screenshots/analysis_result.png)

## Technologies

Python, PyTorch, Torchvision, Streamlit, NumPy, Pillow, Scikit-learn, Jupyter Notebook.

## Dataset

The training notebook uses a PlantVillage-style 38-class crop disease dataset with RGB images resized to 224 × 224. The dataset itself is not included in this repository.

## Running the Application

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app/app.py
```

The application will open in the browser at the local Streamlit address shown in the terminal.

## Model Files

The trained `.pth` files are large model artifacts. They are configured for Git LFS using `.gitattributes`.

For local inference, the application expects:

```
models/hybrid_resnet_densenet_checkpoint.pth
```

## Training Notebook

`notebooks/resnet-densenet.ipynb` contains the model development workflow, dataset preparation, augmentation, training setup, validation, testing, classification results, and checkpoint creation.