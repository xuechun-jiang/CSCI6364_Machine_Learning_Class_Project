import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
KEEP_START = 45
KEEP_COUNT = 7

root_dir = os.getcwd()

for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    images = sorted([
        f for f in os.listdir(subdir_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    keep_images = images[KEEP_START : KEEP_START + KEEP_COUNT]

    for img in images:
        if img not in keep_images:
            os.remove(os.path.join(subdir_path, img))

    print(f"{subdir}: Keep {len(keep_images)}  {KEEP_START}–{KEEP_START+KEEP_COUNT-1}）")
