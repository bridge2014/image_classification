import os
from pathlib import Path
from PIL import Image
import tensorflow as tf  # only if you also want to check TF can load it

def validate_dataset(root_dir):
    bad_files = []
    total = 0
    good = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            total += 1
            path = Path(root) / file
            try:
                # PIL way (what Keras uses internally)
                with Image.open(path) as img:
                    img.verify()          # basic integrity check
                    img = Image.open(path)  # reopen because verify closes it
                    img.load()            # force full decode

                # Optional: also try TF load
                # tf.keras.utils.load_img(path)

                good += 1
                print(f"OK: {path}")

            except Exception as e:
                bad_files.append((str(path), str(e)))
                print(f"BAD: {path} ? {e}")

    print(f"\nSummary:")
    print(f"Total files scanned: {total}")
    print(f"Good images: {good}")
    print(f"Bad / suspicious files: {len(bad_files)}")

    if bad_files:
        print("\nProblematic files:")
        for p, err in bad_files:
            print(f"  {p}  ?  {err}")

    return bad_files

# === Change these paths ===
TRAIN_DIR = "/vast/home/fwang/image_ai/data/test/"      # or validation/test dir
validate_dataset(TRAIN_DIR)