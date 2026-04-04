# RV Health Indices Prediction from TEE Images

This repository is designed for the prediction of clinically relevant indices for assessing the health status of the right ventricle (RV) from transesophageal echocardiography (TEE) images. The indices targeted are:

- **TAPSE** (Tricuspid Annular Plane Systolic Excursion)  
- **RVLFS** (RV Longitudinal Fractional Shortening)  
- **RVFAC** (RV Fractional Area Change)  
- **RV Diameter**

To derive these, the approach consists of tracking three anatomical landmarks from TEE images:

1. Free wall tricuspid annulus point  
2. Septal wall tricuspid annulus point  
3. Apex of the RV

## Implemented Pipelines

- **twod**: Individual 2D TEE frames are fed into a model that predicts and tracks the three target points per frame, and calculates the clinical indices consequently.  
- **2D+T**: Sequences of 64 consecutive frames are fed into a 3D spatiotemporal model, which outputs all three points for each frame simultaneously.  
- **Other**: Not detailed here.

---

See the readme in each folder for detailed info

for everything to  launch this command:
pip install -e . 
