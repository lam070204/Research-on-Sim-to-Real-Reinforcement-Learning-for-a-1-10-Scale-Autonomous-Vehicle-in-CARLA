from pathlib import Path
from collections import Counter
from PIL import Image
import re

folders = {
    "OLD TRAIN": Path("autoencoder-semantic/dataset_rgb_autopilot/train/rgb"),
    "OLD TEST": Path("autoencoder-semantic/dataset_rgb_autopilot/test/rgb"),
    "NEW TRAIN": Path("RGB_DATA_COLLECTION/dataset_new_16000/train/rgb"),
    "NEW TEST": Path("RGB_DATA_COLLECTION/dataset_new_16000/test/rgb"),
}

total = 0
corrupted = []
wrong_size = []
wrong_mode = []
spawn_counts = Counter()

spawn_pattern = re.compile(r"_spawn_(\d+)_")

for label, folder in folders.items():
    files = sorted(folder.glob("*.png"))
    print(f"{label:10}: {len(files)}")

    for path in files:
        total += 1

        try:
            with Image.open(path) as image:
                image.load()

                if image.size != (160, 80):
                    wrong_size.append((str(path), image.size))

                if image.mode != "RGB":
                    wrong_mode.append((str(path), image.mode))

        except Exception as error:
            corrupted.append((str(path), str(error)))
            continue

        if "dataset_new_16000" in str(path):
            match = spawn_pattern.search(path.name)
            if match:
                spawn_counts[int(match.group(1))] += 1

print()
print("===== KET QUA KIEM TRA =====")
print("TOTAL CHECKED :", total)
print("CORRUPTED     :", len(corrupted))
print("WRONG SIZE    :", len(wrong_size))
print("WRONG MODE    :", len(wrong_mode))

print()
print("NEW DATA THEO SPAWN:")
for spawn in [1, 4, 6, 7, 8, 9, 10, 11]:
    print(f"Spawn {spawn:02d}: {spawn_counts[spawn]}")

if corrupted:
    print("\nVI DU ANH HONG:")
    for item in corrupted[:10]:
        print(item)

if wrong_size:
    print("\nVI DU SAI KICH THUOC:")
    for item in wrong_size[:10]:
        print(item)

if wrong_mode:
    print("\nVI DU SAI MODE MAU:")
    for item in wrong_mode[:10]:
        print(item)
