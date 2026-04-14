import os
from PIL import Image, UnidentifiedImageError

def delete_bad_images(root_dir, extensions={".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
    bad_files = []

    total = 0
    deleted = 0

    print(f"\n🧹 Scanning for bad images in: {root_dir}\n")

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:

            if not any(file.lower().endswith(ext) for ext in extensions):
                continue

            path = os.path.join(dirpath, file)
            total += 1

            # check empty file
            if os.path.getsize(path) == 0:
                bad_files.append((path, "Empty file"))
                continue

            try:
                img = Image.open(path)
                img.verify()   # detect corruption

            except (UnidentifiedImageError, OSError, SyntaxError):
                bad_files.append((path, "Corrupted / unreadable"))

    print(f"\n📊 Summary before deletion")
    print(f"Total images checked: {total}")
    print(f"Bad images found    : {len(bad_files)}")

    if not bad_files:
        print("\n✅ No bad images found. Nothing to delete.")
        return

    # confirm before deletion
    confirm = input("\n⚠️ Delete all bad images? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("❌ Deletion cancelled.")
        return

    print("\n🗑️ Deleting bad images...\n")

    for path, reason in bad_files:
        try:
            os.remove(path)
            deleted += 1
            print(f"Deleted: {path} -> {reason}")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

    print(f"\n✅ Done!")
    print(f"Deleted {deleted} bad images out of {len(bad_files)} found.")


if __name__ == "__main__":
    dataset_path = "/home/durai/Documents/MiniProj/Dataset"
    delete_bad_images(dataset_path)
