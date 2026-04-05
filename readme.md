# RV Health Indices Prediction from TEE Images

This repository provides a pipeline for predicting clinically relevant indices that assess the health status of the **right ventricle (RV)** from transesophageal echocardiography (TEE) images.

## Target Indices

| Index | Full Name |
|-------|-----------|
| **TAPSE** | Tricuspid Annular Plane Systolic Excursion |
| **RVLFS** | RV Linear Strain |
| **RVFAC** | RV Fractional Area Change |
| **RV Diameter** | Right Ventricular Diameter |

## Approach

Indices are derived by tracking three anatomical landmarks across TEE frames:

1. Free wall tricuspid annulus point
2. Septal wall tricuspid annulus point
3. Apex of the RV

## Pipelines

> ⚠️ **Note:** Only the `twod` pipeline is currently functional. The `2D+T` pipeline is not working — likely due to insufficient dataset variability for spatiotemporal analysis, or a potential bug. Further investigation is needed.

### `twod` ✅
Individual 2D TEE frames are fed into a model that predicts and tracks the three target landmarks per frame, from which clinical indices are subsequently calculated.

For a detailed explanation of how this pipeline works, see [`./twod/README.md`](./twod/README.md).

### `2D+T` ⚠️ (not working)
Sequences of 64 consecutive frames are fed into a 3D spatiotemporal model, which outputs all three landmark coordinates for each frame simultaneously.

## Getting Started

```bash
git clone https://github.com/MatteoMissana/tapse_estimation
cd tapse_estimation
pip install -r requirements.txt
pip install -e .
```

Then refer to [`./twod/README.md`](./twod/README.md) for pipeline-specific instructions.

## Demo (click on it for the video)

[![Demo Video](https://img.youtube.com/vi/IUViyJUNPxE/0.jpg)](https://youtu.be/IUViyJUNPxE)
