import os
from PIL import Image, UnidentifiedImageError

def check_dataset(root_dir, extensions={".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
    results = {}

    total_all = 0
    good_all = 0
    bad_all = 0

    print(f"\n🔍 Checking dataset: {root_dir}\n")

    # Walk through Train / Validation / Test folders
    for split in os.listdir(root_dir):
        split_path = os.path.join(root_dir, split)

        if not os.path.isdir(split_path):
            continue

        split_total = 0
        split_good = 0
        split_bad = 0
        bad_files = []

        for dirpath, _, filenames in os.walk(split_path):
            for file in filenames:

                if not any(file.lower().endswith(ext) for ext in extensions):
                    continue

                path = os.path.join(dirpath, file)
                split_total += 1

                # empty file check
                if os.path.getsize(path) == 0:
                    split_bad += 1
                    bad_files.append((path, "Empty file"))
                    continue

                try:
                    img = Image.open(path)
                    img.verify()

                    split_good += 1

                except (UnidentifiedImageError, OSError, SyntaxError):
                    split_bad += 1
                    bad_files.append((path, "Corrupted / unreadable"))

        # store split results
        results[split] = {
            "total": split_total,
            "good": split_good,
            "bad": split_bad,
            "bad_files": bad_files
        }

        total_all += split_total
        good_all += split_good
        bad_all += split_bad

    # FINAL REPORT
    print("\n📊 OVERALL DATASET REPORT")
    print(f"Total images : {total_all}")
    print(f"Good images  : {good_all}")
    print(f"Bad images   : {bad_all}")

    print("\n📂 SPLIT-WISE REPORT")

    for split, stats in results.items():
        print(f"\n➡ {split.upper()}")
        print(f"Total : {stats['total']}")
        print(f"Good  : {stats['good']}")
        print(f"Bad   : {stats['bad']}")

        # show sample bad images
        if stats["bad_files"]:
            print("  ❌ Sample bad images:")
            for path, reason in stats["bad_files"][:10]:
                print(f"   - {path} -> {reason}")

    return results


if __name__ == "__main__":
    dataset_path = "/home/durai/Documents/MiniProj/Dataset"
    check_dataset(dataset_path)
