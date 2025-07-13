# 🧠 Liver Cancer Detection and Classification from CT Scans

This project presents a lightweight, modular deep learning pipeline for **automatic liver tumor segmentation and classification** using CT scans. It features:

- A 2D U-Net model for binary tumor segmentation
- A simple 2D CNN classifier for tumor type identification
- A fully interactive **Streamlit UI** for real-time testing and visualization
- Memory-efficient processing using **lazy loading**
- CLI-based training and evaluation pipeline

---

## 📁 Project Structure

```
Liver_Cancer_Model/
├── model_weights/
│ ├── classifier2d_epoch30.pt
│ ├── unet2d_epoch30.pt
├── src/
│ ├── models.py # UNet2D and Simple2DClassifier
│ ├── train.py # training script
│ ├── datasets.py
│ ├── data.py
│ └── eval.py
├── preprocess.py
├── main_pipeline.py
├── test_pipeline.py
├── full_pipeline.py
├── inference_utils.py
├── app.py # Streamlit interface
├── requirements.txt
└── README.md
```
---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/AhmedSobeih/Liver_Cancer_Model.git
cd Liver_Cancer_Model
```

### 2. Set Up the Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit interface (fastest option)

```bash
streamlit run app.py
```

- Upload a CT scan
- View tumor segmentation slice-by-slice
- See predicted tumor type with description

### OR

### 3. Download the Dataset

Download the Primary Liver Cancer CECT Dataset: https://www.scidb.cn/en/detail?dataSetId=d685a0b9f8974a2a9d7c880be1dc36e9

### 4. Run the Preprocessing
```bash
# Flattening CT files…"
python preprocess.py /path/to/CT/source   path/to/CT/target
#  Flattening mask files…"
python preprocess.py /path/to/mask/source   path/to/mask/target
```

### 4. Train the models
```bash
python full_pipeline.py \
  --ct_folder  /path/to/CT/target  \
  --mask_folder /path/to/mask/target \
  --csv_path    /path/to/patient_data.csv \
  --epochs 30 \
  --export_folder /path/to/desired/output/directory \
  --batch_size_seg 128 \
  --batch_size_cls 128
```

### 5. Run the Streamlit on your own trained models

```bash
streamlit run app.py
```

- Upload a CT scan
- View tumor segmentation slice-by-slice
- See predicted tumor type with description

## 📊 Results

- Segmentation pixel accuracy: 0.9927
- Classification accuracy: 89.7%
- Training time reduced 87.5% via lazy loading
- Batch size improved from 2 → 128

## 📬 Contact

For questions, feel free to contact:
- Shehabeldin Solyman – st196462@stud.uni-stuttgart.de
- Ahmed Eid – st192266@stud.uni-stuttgart.de
