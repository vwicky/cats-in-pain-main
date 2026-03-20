import os
from sklearn.model_selection import train_test_split

def build_file_list(root_dir, val_ratio=0.2, random_state=42):
    file_list = []
    labels = []

    # Only keep real directories (classes)
    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(root_dir, cls)

        for file in os.listdir(cls_path):
            if (
                file.endswith((".wav", ".mp3"))
                and "_aug" not in file
            ):
                file_list.append(
                    (os.path.join(cls_path, file), class_to_idx[cls])
                )
                labels.append(class_to_idx[cls])

    train_files, val_files = train_test_split(
        file_list,
        test_size=val_ratio,
        stratify=labels,
        random_state=random_state
    )

    return train_files, val_files, class_to_idx