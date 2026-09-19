# Model Artifacts

The main deployable model is the trained **Hybrid ResNet50 + DenseNet121** checkpoint.

### Architecture

```text
Input: 224 × 224 RGB image
        │
        ├── ResNet50 → 2048 features
        │
        └── DenseNet121 → 1024 features
                    │
                    ▼
             Concatenate 3072
                    │
             BatchNorm + SE
                    │
             MLP 3072 → 1024 → 512 → 38
                    │
                    ▼
             Crop/Disease Class
```

The checkpoint was created by the accompanying notebook and contains the model state dictionary and class-name metadata.

The `.pth` files are configured for Git LFS because of their size.
