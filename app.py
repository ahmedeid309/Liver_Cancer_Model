# app.py – Streamlit UI for liver‑tumour detection (segmentation) and optional subtype classification

import os
import gzip
import shutil
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch
from scipy.ndimage import label
from skimage.transform import resize

from models import UNet2D, Simple2DClassifier
from inference_utils import load_volume, slice_batch_to_tensor

# -----------------------------------------------------------------------------
# Descriptions
# -----------------------------------------------------------------------------
LONG_LABELS = ["Combined Hepatocellular-Cholangiocarcinoma","Cystic Neoplasm","Hepatocellular Carcinoma","Intrahepatic Cholangiocarcinoma"]
DESCRIPTIONS = ["Combined hepatocellular-cholangiocarcinoma (CHCC) is a rare and complex liver cancer that exhibits histological features of both HCC and ICC. This hybrid tumor is thought to arise from hepatic progenitor cells and shares risk factors with both tumor types, including cirrhosis, hepatitis infections, and chronic liver inflammation. Patients may experience symptoms such as abdominal discomfort, fatigue, weight loss, and jaundice. CHCC is particularly challenging to diagnose and treat due to its mixed cellular composition and often exhibits more aggressive behavior than either HCC or ICC alone. Treatment strategies are typically multimodal, involving surgery, chemotherapy, and targeted therapies, but prognosis remains poor in advanced stages.",
                "Cystic neoplasms (CNs) of the liver are a group of rare, potentially malignant cystic tumors that include biliary cystadenomas and cystadenocarcinomas. These lesions often present as incidental findings during imaging but may cause symptoms such as abdominal fullness, pain, or nausea when large. While many cystic lesions are benign, some have the potential to become malignant, especially if they are multilocular or exhibit mural nodules. Risk factors are less well-defined but may include congenital biliary malformations or chronic irritation of the bile ducts. Complete surgical removal is usually recommended to prevent malignant transformation. Early diagnosis and resection generally offer an excellent prognosis.",
                "Hepatocellular carcinoma (HCC) is the most common type of primary liver cancer, typically arising in the setting of chronic liver disease. It is strongly associated with cirrhosis, chronic hepatitis B or C infections, alcohol abuse, and non-alcoholic fatty liver disease (NAFLD). Symptoms often include abdominal pain, weight loss, fatigue, jaundice, and a palpable mass in the upper right abdomen. HCC is considered highly aggressive due to its tendency to invade blood vessels and metastasize. Early detection is critical, as prognosis worsens significantly in advanced stages. Treatment options include surgical resection, liver transplantation, local ablation, and systemic therapies such as targeted drugs or immunotherapy.",
                "Intrahepatic cholangiocarcinoma (ICC) is a malignant tumor that originates in the bile ducts within the liver. It is the second most common primary liver cancer and often presents with nonspecific symptoms such as right upper quadrant pain, jaundice, weight loss, and fatigue. Risk factors include primary sclerosing cholangitis, liver fluke infections (e.g., Opisthorchis, Clonorchis), chronic biliary inflammation, and certain congenital bile duct anomalies. ICC is typically diagnosed at a late stage due to its subtle onset and aggressive nature, making it difficult to treat. Surgical resection offers the best chance for survival, but many cases require systemic chemotherapy or palliative care."
                ]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEGMENTATION_MODEL_PATH = "model_weights/unet2d_epoch30.pt"
CLASSIFICATION_MODEL_PATH = "model_weights/classifier2d_epoch30.pt"
LABELS = ["CHCC", "CN", "HCC", "ICC"]          # order used during training

# -----------------------------------------------------------------------------
# Model loading (cached so it runs only once per session)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    # segmentation U‑Net
    seg_model = UNet2D().to(DEVICE)
    ckpt = torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE)
    seg_model.load_state_dict(ckpt["model_state"])
    seg_model.eval()

    # subtype classifier (optional)
    clf_model = Simple2DClassifier(n_classes=len(LABELS)).to(DEVICE)
    ckpt = torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE)
    clf_model.load_state_dict(ckpt["model_state"])
    clf_model.eval()

    return seg_model, clf_model

seg_model, clf_model = load_models()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """Return binary mask of the largest connected component (or original if none)."""
    labelled, n_feat = label(mask)
    if n_feat == 0:
        return mask.astype(np.uint8)
    largest_lbl = np.argmax(np.bincount(labelled.flat)[1:]) + 1
    return (labelled == largest_lbl).astype(np.uint8)


def extract_patch(image: np.ndarray, mask: np.ndarray, output_size=(128, 32)) -> np.ndarray:
    """Crop the bounding box of *mask* from *image* and resize to *output_size*."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros(output_size, dtype=np.float32)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    patch = image[y_min : y_max + 1, x_min : x_max + 1]
    return resize(patch, output_size, anti_aliasing=True)


def run_classification(model: torch.nn.Module, slice_img: np.ndarray) -> np.ndarray:
    """Return softmax probabilities for a single axial slice."""
    t = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        return torch.softmax(logits, 1).cpu().squeeze().numpy()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("CT‑scan Liver Cancer Detection")
uploaded_file = st.file_uploader("Upload a NIfTI file (.nii or .nii.gz)")

if uploaded_file:
    # --- Save the upload to a temporary file ----------------------------------
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # --- Decompress if necessary ---------------------------------------------
    if suffix == ".nii.gz":
        dec_path = tmp_path[:-3]
        with gzip.open(tmp_path, "rb") as f_in, open(dec_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.unlink(tmp_path)
        nifti_path = dec_path
    else:
        nifti_path = tmp_path

    # --- Load volume (same preprocessing as used for training) ---------------
    vol = load_volume(nifti_path)      # shape D×H×W, raw float32 HU
    os.unlink(nifti_path)              # clean temp file

    st.success(f"CT scan loaded. Volume shape = {vol.shape} (D×H×W)")

    # -------------------------------------------------------------------------
    # Whole‑volume segmentation (single forward pass)                          
    # -------------------------------------------------------------------------
    tensor = slice_batch_to_tensor(vol, DEVICE)   # tensor D×1×H×W (float32)
    with torch.no_grad():
        logits = seg_model(tensor)
        probs = torch.sigmoid(logits)
        masks = (probs > 0.5).float().cpu().numpy()[:, 0]   # D×H×W binary

    tumor_present_volume = masks.sum() > 0
    st.markdown(
        f"**Cancer detected (anywhere in volume):** "
        f"{'✅ YES' if tumor_present_volume else '❌ NO'}"
        f"- **Tumour pixels : **  {int(masks.sum())}"
    )

    # -------------------------------------------------------------------------
    # Slice selector                                                           
    # -------------------------------------------------------------------------
    D = vol.shape[0]
    slice_idx = st.slider("Select axial slice", 0, D - 1, D // 2)

    slice_img = vol[slice_idx]              # H×W float32
    mask_pred = masks[slice_idx]            # H×W binary
    
    mask_largest = get_largest_component(mask_pred)
    patch = extract_patch(slice_img, mask_largest)
    tumor_present_slice = mask_pred.sum() > 0

    # -------------------------------------------------------------------------
    # Visualisation                                                            
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(slice_img, cmap="gray"); axs[0].set_title("Original")
    axs[1].imshow(mask_pred, cmap="gray"); axs[1].set_title("Mask")
    axs[2].imshow(patch, cmap="gray");     axs[2].set_title("Tumour patch")
    for ax in axs:
        ax.axis("off")
    st.pyplot(fig)

    # -------------------------------------------------------------------------
    # Subtype classification (optional)                                        
    # -------------------------------------------------------------------------
    if tumor_present_volume:
        probs_cls = run_classification(clf_model, slice_img)  # could also use patch
        pred_cls = int(np.argmax(probs_cls))
        st.subheader(f"Classification (subtype, slice‑level) - `{LABELS[pred_cls]}`")
        st.subheader(f"Predicted class: `{LONG_LABELS[pred_cls]}` with Confidence `{probs_cls[pred_cls] * 100:.2f}%`")
        # st.markdown(f"**`{LABELS[pred_cls]}`**")
        st.write(DESCRIPTIONS[pred_cls])
    else:
        st.info("No tumour detected in volume – subtype classification skipped.")

   
