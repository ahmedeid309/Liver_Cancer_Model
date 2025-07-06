# main_pipeline.py
import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.data import load_and_split_data_from_paths
from src.datasets import CT2DSliceSegDataset, CT2DSliceClassDataset
from src.models import UNet2D, Simple2DClassifier
from src.train import train_segmentation, train_classification
from src.eval import evaluate_segmentation, evaluate_classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_folder', type=str, required=True, help='Path to CT scans folder')
    parser.add_argument('--mask_folder', type=str, required=True, help='Path to mask folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to patient_data.csv file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--export_folder', type=str, required=True, help='Path to export trained models')
    parser.add_argument('--batch_size_seg', type=int, default=16, help='Batch size for segmentation training')
    parser.add_argument('--batch_size_cls', type=int, default=32, help='Batch size for classification training')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Device:", torch.cuda.get_device_name(DEVICE))
    print("Running on:", DEVICE)

    # 1. Load data
    df, train_seg, val_seg, _, train_cls, val_cls, _, label_encoder = load_and_split_data_from_paths(
        args.csv_path, args.ct_folder, args.mask_folder
    )

    # 2. Prepare datasets and loaders
    train_seg_ds = CT2DSliceSegDataset(train_seg, args.ct_folder, args.mask_folder)
    val_seg_ds   = CT2DSliceSegDataset(val_seg,   args.ct_folder, args.mask_folder)

    train_cls_ds = CT2DSliceClassDataset(train_cls[train_cls["mask_path"].notna()], label_encoder,
                                         args.ct_folder, args.mask_folder)
    val_cls_ds   = CT2DSliceClassDataset(val_cls[val_cls["mask_path"].notna()],   label_encoder,
                                         args.ct_folder, args.mask_folder)

    train_seg_loader = DataLoader(train_seg_ds, batch_size=args.batch_size_seg,
                                  shuffle=True, num_workers=8, pin_memory=True)
    train_cls_loader = DataLoader(train_cls_ds, batch_size=args.batch_size_cls,
                                  shuffle=True, num_workers=8, pin_memory=True)

    # 3. Train or resume models
    os.makedirs(args.export_folder, exist_ok=True)

    # Try loading last checkpoint for segmentation
    unet2d = UNet2D().to(DEVICE)
    start_epoch_seg = 0
    for i in range(args.epochs, 0, -1):
        ckpt_path = os.path.join(args.export_folder, f"unet2d_epoch{i:02d}.pt")
        if os.path.exists(ckpt_path):
            print(f"🔁 Resuming segmentation training from epoch {i}")
            checkpoint = torch.load(ckpt_path)
            unet2d.load_state_dict(checkpoint['model_state'])
            start_epoch_seg = checkpoint['epoch']
            break

    print("\nTraining 2D U-Net for segmentation...")
    train_segmentation(unet2d, train_seg_loader, DEVICE,
                       export_dir=args.export_folder,
                       start_epoch=start_epoch_seg, epochs=args.epochs)

    torch.save(unet2d.state_dict(), os.path.join(args.export_folder, "unet2d_final.pth"))

    # Try loading last checkpoint for classification
    cls2d = Simple2DClassifier(n_classes=len(label_encoder.classes_)).to(DEVICE)
    start_epoch_cls = 0
    for i in range(args.epochs, 0, -1):
        ckpt_path = os.path.join(args.export_folder, f"classifier2d_epoch{i:02d}.pt")
        if os.path.exists(ckpt_path):
            print(f"🔁 Resuming classification training from epoch {i}")
            checkpoint = torch.load(ckpt_path)
            cls2d.load_state_dict(checkpoint['model_state'])
            start_epoch_cls = checkpoint['epoch']
            break

    print("\nTraining classifier for tumor type...")
    train_classification(cls2d, train_cls_loader, DEVICE,
                         export_dir=args.export_folder,
                         start_epoch=start_epoch_cls, epochs=args.epochs)

    torch.save(cls2d.state_dict(), os.path.join(args.export_folder, "classifier2d_final.pth"))

    # 5. Validate with same batch sizes used for training
    print("\nEvaluating segmentation on validation set...")
    evaluate_segmentation(unet2d, val_seg_ds, DEVICE, batch_size=args.batch_size_seg)

    print("\nEvaluating classification on validation set...")
    evaluate_classification(cls2d, val_cls_ds, label_encoder, DEVICE, batch_size=args.batch_size_cls)
