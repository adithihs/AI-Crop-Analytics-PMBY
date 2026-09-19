# My Contribution

This document identifies the work contributed by me within the broader internship team project.

## 1. Damage / Disease Detection Model

I developed the hybrid image-classification component based on **ResNet50 + DenseNet121**.

### Architecture

- ResNet50 backbone → 2048-dimensional feature representation
- DenseNet121 backbone → 1024-dimensional feature representation
- Concatenation → 3072-dimensional fused feature vector
- BatchNorm1d
- SE-based channel attention
- MLP classifier → 1024 → 512 → 38 classes

The training notebook records **99.35% best validation accuracy** and **98.63% test accuracy** for the documented experiment.

## 2. Frontend

I designed the Streamlit frontend used to present the crop-analysis workflow, including image upload, system status, analysis controls, result cards, confidence display, health/damage information, growth-stage display, and advisory presentation.

## 3. Integration

The hybrid model is included as a deployable model backend in `models/`, while the full Streamlit application is in `app/app.py`.

## 4. Portfolio Presentation

The repository includes the full team application, but clearly separates the individual contribution so the project is not presented as entirely individual work.
