# test_pipeline.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
from src.data import load_and_split_data_from_paths
from src.datasets import CT2DSliceSegDataset, CT2DSliceClassDataset
from src.models import UNet2D, Simple2DClassifier
from src.eval import evaluate_segmentation, evaluate_classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_folder', type=str, required=True, help='Path to CT scans folder')
    parser.add_argument('--mask_folder', type=str, required=True, help='Path to mask folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to patient_data.csv file')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to folder with saved models')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", DEVICE)

    # Load data
    df, _, _, test_seg, _, _, test_cls, label_encoder = load_and_split_data_from_paths(
        args.csv_path, args.ct_folder, args.mask_folder
    )

    test_seg_ds = CT2DSliceSegDataset(test_seg, args.ct_folder, args.mask_folder)
    test_cls_ds = CT2DSliceClassDataset(test_cls[test_cls["mask_path"].notna()], label_encoder, args.ct_folder, args.mask_folder)

    # Load models
    unet2d = UNet2D().to(DEVICE)
    cls2d = Simple2DClassifier(n_classes=len(label_encoder.classes_)).to(DEVICE)

    unet_path = os.path.join(args.model_folder, "unet2d.pth")
    cls_path = os.path.join(args.model_folder, "classifier2d.pth")

    unet2d.load_state_dict(torch.load(unet_path, map_location=DEVICE))
    cls2d.load_state_dict(torch.load(cls_path, map_location=DEVICE))

    # Evaluate
    print("\nTesting segmentation model...")
    evaluate_segmentation(unet2d, test_seg_ds, DEVICE)

    print("\nTesting classification model...")
    evaluate_classification(cls2d, test_cls_ds, label_encoder, DEVICE)
