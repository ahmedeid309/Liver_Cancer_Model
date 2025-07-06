import os
import shutil
import argparse

def flatten_and_copy_nii_files(source_root, target_root):
    """
    Copy .nii and .nii.gz files from nested directories to a flat target directory, skipping duplicates.
    """
    if os.path.exists(target_root):
        print(f"Directory {target_root} already exists. Skipping flattening to avoid duplicates.")
        return

    os.makedirs(target_root, exist_ok=True)
    for dirpath, _, filenames in os.walk(source_root):
        for file in filenames:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                src_path = os.path.join(dirpath, file)
                dst_path = os.path.join(target_root, file)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                  #  print(f"Copied {file}")
                #else:
                   # print(f"Skipped duplicate: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten and copy .nii files from nested directories to a flat target directory.")
    parser.add_argument("source_root", help="Path to the source directory containing nested .nii files.")
    parser.add_argument("target_root", help="Path to the target directory where files will be copied.")
    args = parser.parse_args()

    flatten_and_copy_nii_files(args.source_root, args.target_root)
