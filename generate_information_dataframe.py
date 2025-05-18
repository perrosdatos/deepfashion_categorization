import os
import pandas as pd

def collect_image_paths(root_folder, valid_extensions={".jpg", ".jpeg", ".png", ".webp"}):
    data = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(full_path, root_folder)
                category = os.path.basename(os.path.dirname(full_path))
                data.append({"category": category, "path": rel_path})
    return pd.DataFrame(data)

if __name__ == "__main__":
    cropped_path = "../custom_dataset/train/Cropped/"
    labeled_path = "../custom_dataset/train/Labeled/"

    cropped_df = collect_image_paths(cropped_path)
    cropped_df.to_csv(os.path.join(cropped_path, "information_dataframe.csv"), index=False)

    labeled_df = collect_image_paths(labeled_path)
    labeled_df.to_csv(os.path.join(labeled_path, "information_dataframe.csv"), index=False)