# RV Health Indices Prediction from TEE Images

This repository is designed for the prediction of clinically relevant indices for assessing the health status of the right ventricle (RV) from transesophageal echocardiography (TEE) images. The indices targeted are:

- **TAPSE** (Tricuspid Annular Plane Systolic Excursion)  
- **RVLFS** (RV Longitudinal Fractional Shortening)  
- **RVFAC** (RV Fractional Area Change)  
- **RV Diameter**

To derive these, the approach consists of predicting three anatomical landmarks from TEE images:

1. Free wall tricuspid annulus point  
2. Septal wall tricuspid annulus point  
3. Apex of the RV

## Implemented Pipelines

- **2D**: Individual 2D TEE frames are fed into a model that predicts the three target points per frame.  
- **2D+T**: Sequences of 64 consecutive frames are fed into a 3D spatiotemporal model, which outputs all three points for each frame simultaneously.  
- **Other**: Not detailed here.

---

## Main Scripts

| Script | Description |
| `2d/training.py` | Performs training for the 2D pipeline. Refer to the script for detailed information on arguments, model definitions, and training logic. |
