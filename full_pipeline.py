# main_pipeline.py  – train, validate *and* test in one go
import argparse, os, torch
from torch.utils.data import DataLoader

from src.data     import load_and_split_data_from_paths
from src.datasets import CT2DSliceSegDataset, CT2DSliceClassDataset
from src.models   import UNet2D, Simple2DClassifier
from src.train    import train_segmentation, train_classification
from src.eval     import evaluate_segmentation, evaluate_classification

# --------------------------------------------------------------------------- #
#                           Command-line arguments                            #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--ct_folder',      required=True)
parser.add_argument('--mask_folder',    required=True)
parser.add_argument('--csv_path',       required=True)
parser.add_argument('--export_folder',  required=True,
                    help='Where checkpoints (.pt) and final .pth files go')
parser.add_argument('--epochs',         type=int,  default=10)
parser.add_argument('--batch_size_seg', type=int,  default=16)
parser.add_argument('--batch_size_cls', type=int,  default=32)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {DEVICE}\n")

# --------------------------------------------------------------------------- #
# 1) Data split (train / val / test)                                          #
# --------------------------------------------------------------------------- #
(
    df,
    train_seg, val_seg, test_seg,          # ← segmentation splits
    train_cls, val_cls, test_cls,          # ← classification splits
    label_encoder
) = load_and_split_data_from_paths(
        args.csv_path, args.ct_folder, args.mask_folder
)
print("Label mapping (index → cancer type):")
for idx, name in enumerate(label_encoder.classes_):
    print(f"  {idx}: {name}")


# --------------------------------------------------------------------------- #
# 2) Datasets & loaders                                                       #
# --------------------------------------------------------------------------- #
# -- segmentation --
train_seg_ds = CT2DSliceSegDataset(train_seg, args.ct_folder, args.mask_folder)
val_seg_ds   = CT2DSliceSegDataset(val_seg,   args.ct_folder, args.mask_folder)
test_seg_ds  = CT2DSliceSegDataset(test_seg,  args.ct_folder, args.mask_folder)

# -- classification (skip rows without masks so that slices align) --
train_cls_ds = CT2DSliceClassDataset(
    train_cls[train_cls["mask_path"].notna()],
    label_encoder, args.ct_folder, args.mask_folder
)
val_cls_ds   = CT2DSliceClassDataset(
    val_cls[val_cls["mask_path"].notna()],
    label_encoder, args.ct_folder, args.mask_folder
)
test_cls_ds  = CT2DSliceClassDataset(
    test_cls[test_cls["mask_path"].notna()],
    label_encoder, args.ct_folder, args.mask_folder
)

# loaders
train_seg_loader = DataLoader(train_seg_ds, batch_size=args.batch_size_seg,
                              shuffle=True,  num_workers=8, pin_memory=True)
train_cls_loader = DataLoader(train_cls_ds, batch_size=args.batch_size_cls,
                              shuffle=True,  num_workers=8, pin_memory=True)

# --------------------------------------------------------------------------- #
# 3) Train / resume SEGMENTATION model                                        #
# --------------------------------------------------------------------------- #
os.makedirs(args.export_folder, exist_ok=True)
unet2d = UNet2D().to(DEVICE)

start_epoch_seg = 0
for i in range(args.epochs, 0, -1):
    ck = os.path.join(args.export_folder, f"unet2d_epoch{i:02d}.pt")
    if os.path.exists(ck):
        print(f"🔁  Resuming segmentation from epoch {i}")
        checkpoint = torch.load(ck, map_location=DEVICE)
        unet2d.load_state_dict(checkpoint['model_state'])
        start_epoch_seg = checkpoint['epoch']
        break

print("\n🟢 Training 2-D U-Net (segmentation)")
train_segmentation(
    unet2d, train_seg_loader, DEVICE,
    export_dir=args.export_folder,
    start_epoch=start_epoch_seg, epochs=args.epochs
)
torch.save(unet2d.state_dict(),
           os.path.join(args.export_folder, "unet2d_final.pth"))

# --------------------------------------------------------------------------- #
# 4) Train / resume CLASSIFICATION model                                      #
# --------------------------------------------------------------------------- #
cls2d = Simple2DClassifier(n_classes=len(label_encoder.classes_)).to(DEVICE)

start_epoch_cls = 0
for i in range(args.epochs, 0, -1):
    ck = os.path.join(args.export_folder, f"classifier2d_epoch{i:02d}.pt")
    if os.path.exists(ck):
        print(f"🔁  Resuming classification from epoch {i}")
        checkpoint = torch.load(ck, map_location=DEVICE)
        cls2d.load_state_dict(checkpoint['model_state'])
        start_epoch_cls = checkpoint['epoch']
        break

print("\n🟢 Training classifier (tumor type)")
train_classification(
    cls2d, train_cls_loader, DEVICE,
    export_dir=args.export_folder,
    start_epoch=start_epoch_cls, epochs=args.epochs
)
torch.save(cls2d.state_dict(),
           os.path.join(args.export_folder, "classifier2d_final.pth"))

# --------------------------------------------------------------------------- #
# 5) VALIDATION                                                               #
# --------------------------------------------------------------------------- #
print("\n🔍 Validation results")
evaluate_segmentation(unet2d, val_seg_ds, DEVICE,
                      batch_size=args.batch_size_seg)
evaluate_classification(cls2d, val_cls_ds, label_encoder, DEVICE,
                       batch_size=args.batch_size_cls)

# --------------------------------------------------------------------------- #
# 6) TESTING (previously in test_pipeline.py)                                 #
# --------------------------------------------------------------------------- #
print("\n🏁 Final test-set results")
evaluate_segmentation(unet2d, test_seg_ds, DEVICE,
                      batch_size=args.batch_size_seg)
evaluate_classification(cls2d, test_cls_ds, label_encoder, DEVICE,
                       batch_size=args.batch_size_cls)
