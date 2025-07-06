# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_and_split_data_from_paths(csv_path, ct_root, mask_root):
    df = pd.read_csv(csv_path)
    df["cancer_type_str"] = df["cancer_type"]
    label_encoder = LabelEncoder()
    df["cancer_type"] = label_encoder.fit_transform(df["cancer_type_str"])

    df["ct_path"] = df["ct_path"].apply(lambda x: os.path.join(ct_root, os.path.basename(x).replace(".nii.gz", ".nii")))
    print("number of NAN values in mask")
    print(df["mask_path"].isna().sum())


    df = df[df["mask_path"].notna()].copy()
    df["mask_path"] = df["mask_path"].apply(lambda x: os.path.join(mask_root, os.path.basename(x).replace(".nii.gz", ".nii")))

    df_seg = df[df["mask_path"].notna()].reset_index(drop=True)
    train_seg, temp_seg = train_test_split(df_seg, test_size=0.30, random_state=42)
    val_seg, test_seg = train_test_split(temp_seg, test_size=0.50, random_state=42)

    train_cls, temp_cls = train_test_split(df, test_size=0.30, random_state=42, stratify=df["cancer_type"])
    val_cls, test_cls = train_test_split(temp_cls, test_size=0.50, random_state=42, stratify=temp_cls["cancer_type"])

    return df, train_seg, val_seg, test_seg, train_cls, val_cls, test_cls, label_encoder
