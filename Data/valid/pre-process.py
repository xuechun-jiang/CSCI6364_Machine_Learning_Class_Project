import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
KEEP_LAST = 7

root_dir = os.getcwd()

for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)

    if not os.path.isdir(subdir_path):
        continue

    images = sorted([
        f for f in os.listdir(subdir_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    keep_images = images[-KEEP_LAST:]  

    for img in images:
        if img not in keep_images:
            os.remove(os.path.join(subdir_path, img))

    print(f"{subdir}: Keep Last {len(keep_images)} ")
