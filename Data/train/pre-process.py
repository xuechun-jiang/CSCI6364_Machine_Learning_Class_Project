import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
KEEP_FIRST = 45

root_dir = os.getcwd()

for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)

    if not os.path.isdir(subdir_path):
        continue

    images = [
        f for f in os.listdir(subdir_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    images.sort()  

    for img in images[KEEP_FIRST:]:
        os.remove(os.path.join(subdir_path, img))

    print(f"{subdir}: Keep {KEEP_FIRST} delete {max(0, len(images) - KEEP_FIRST)} ")
