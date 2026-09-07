import os
import random
import shutil

source = "data/challenge1"
destination = "dataset"

classes = ["smoke", "no_smoke"]

for class_name in classes:
    images = os.listdir(os.path.join(source, class_name))

    images = [
        image for image in images
        if image.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, split_images in splits.items():
        folder = os.path.join(destination, split, class_name)
        os.makedirs(folder, exist_ok=True)

        for image in split_images:
            shutil.copy(
                os.path.join(source, class_name, image),
                os.path.join(folder, image)
            )

print("Dataset split complete!")

