import os
import shutil

# Paths
train_src = "dataset/train"
test_src = "dataset/test"
output_dir = "dataset_binary"

# Define normal and abnormal categories
normal_classes = ["normal_columnar", "normal_intermediate", "normal_superficial"]
abnormal_classes = ["carcinoma_in_situ", "light_dysplastic", "moderate_dysplastic", "severe_dysplastic"]

# Helper function
def copy_images(src_dir, dest_dir, categories):
    os.makedirs(dest_dir, exist_ok=True)
    for category in categories:
        category_path = os.path.join(src_dir, category)
        if not os.path.exists(category_path):
            print(f"⚠️ Skipping missing folder: {category_path}")
            continue
        for img_name in os.listdir(category_path):
            src_img = os.path.join(category_path, img_name)
            dest_img = os.path.join(dest_dir, img_name)
            shutil.copy(src_img, dest_img)

# Create binary dataset folders
for split in ["train", "test"]:
    base = os.path.join(output_dir, split)
    os.makedirs(os.path.join(base, "Normal"), exist_ok=True)
    os.makedirs(os.path.join(base, "Abnormal"), exist_ok=True)

    # Copy images
    copy_images(os.path.join(train_src if split == "train" else test_src), os.path.join(base, "Normal"), normal_classes)
    copy_images(os.path.join(train_src if split == "train" else test_src), os.path.join(base, "Abnormal"), abnormal_classes)

print("✅ Binary dataset created successfully in 'dataset_binary/'")
