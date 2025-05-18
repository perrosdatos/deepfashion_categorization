import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

def evaluate_and_save_metrics(model, val_ds, id2label, output_folder):
    """
    Evaluates accuracy, top-2, top-3 accuracy, precision, recall, F1-score,
    and instance count per class. Prints the results and saves them to an Excel file.

    Parameters:
    - model: Trained Keras model
    - val_ds: Validation tf.data.Dataset (batched)
    - id2label: Dictionary mapping label IDs to class names
    - output_folder: Folder path to save the Excel file
    """
    y_true = []
    y_pred = []
    y_pred_probs = []

    # Collect predictions and true labels
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = tf.argmax(probs, axis=1).numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_pred_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)

    # Top-k accuracy
    acc = accuracy_score(y_true, y_pred)
    top_2_acc = tf.keras.metrics.top_k_categorical_accuracy(
        tf.one_hot(y_true, depth=len(id2label)), y_pred_probs, k=2
    ).numpy().mean()
    top_3_acc = tf.keras.metrics.top_k_categorical_accuracy(
        tf.one_hot(y_true, depth=len(id2label)), y_pred_probs, k=3
    ).numpy().mean()

    # Class-wise metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Count true instances per class
    true_counts = Counter(y_true)

    # Print and collect per-class results
    print(f"\n✅ Accuracy:     {acc:.4f}")
    print(f"🎯 Top-2 Acc:    {top_2_acc:.4f}")
    print(f"🎯 Top-3 Acc:    {top_3_acc:.4f}")
    print("\n📊 Per-class metrics:\n")

    rows = []
    for idx, class_name in id2label.items():
        count = true_counts.get(idx, 0)
        row = {
            "class_id": idx,
            "class_name": class_name,
            "count": count,
            "precision": precision[idx],
            "recall": recall[idx],
            "f1_score": f1[idx]
        }
        print(f"{class_name:15}  Count: {count:<4}  Precision: {precision[idx]:.4f}  Recall: {recall[idx]:.4f}  F1: {f1[idx]:.4f}")
        rows.append(row)

    # Save to Excel
    df_metrics = pd.DataFrame(rows)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "classification_report.xlsx")
    df_metrics.to_excel(output_path, index=False)
    print(f"\n📁 Report saved to: {output_path}")

        

import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image

def plot_random_images_per_category(df, image_column='path', category_column='global_category', max_per_row=4):
    """
    Plot one random image per category from the dataframe.

    Parameters:
    - df: pandas.DataFrame with image paths and categories.
    - image_column: str, column name with image file paths.
    - category_column: str, column name with image categories.
    - max_per_row: int, number of images per row in the plot grid.
    """
    categories = df[category_column].unique()
    sampled_rows = df.groupby(category_column).apply(lambda x: x.sample(1)).reset_index(drop=True)

    num_categories = len(categories)
    num_cols = min(max_per_row, num_categories)
    num_rows = (num_categories + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = axes.flatten() if num_categories > 1 else [axes]

    for ax, (_, row) in zip(axes, sampled_rows.iterrows()):
        try:
            image = Image.open(row[image_column])
            ax.imshow(image)
            ax.set_title(f"{row[category_column]}")
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, "Error loading image", ha='center')
            ax.set_title(row[category_column])
            ax.axis('off')
            print(f"Failed to load image at {row[image_column]}: {e}")

    # Hide unused subplots
    for ax in axes[len(sampled_rows):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Save to Excel
    df_metrics = pd.DataFrame(rows)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "classification_report.xlsx")
    df_metrics.to_excel(output_path, index=False)
    print(f"\n📁 Report saved to: {output_path}")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_top3_predictions(model, val_ds, id2label, rows=3, cols=8):
    """
    Muestra imágenes con las Top-3 predicciones de un modelo, coloreando en verde
    si la etiqueta verdadera está entre ellas, y rojo en caso contrario.

    Parameters:
    - model: modelo Keras o TensorFlow.
    - val_ds: dataset iterable (como tf.data.Dataset).
    - id2label: diccionario {id: nombre_clase}.
    - rows: número de filas a mostrar.
    - cols: número de columnas a mostrar.
    """
    val_images, val_labels = next(iter(val_ds))
    pred_probs = model.predict(val_images)
    
    top3_pred_indices = tf.math.top_k(pred_probs, k=3).indices.numpy()

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle("Top-3 Predicciones del modelo", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= len(val_images):
            break

        image = val_images[i].numpy()
        true_label = int(val_labels[i].numpy())
        top3_labels = top3_pred_indices[i]

        top3_names = [id2label[idx] for idx in top3_labels]
        pred_text = "\n".join([f"{j+1}: {name}" for j, name in enumerate(top3_names)])

        color = "green" if true_label in top3_labels else "red"

        ax.imshow(image)
        ax.set_title(f"{pred_text}\nT: {id2label[true_label]}", color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import Counter

def compare_models_metrics(models_list, val_ds_list, id2label, output_folder):
    """
    Evaluates multiple models on val_ds and saves:
    - Sheet 1: global accuracy, top-2, top-3 accuracy per model
    - Sheet 2: per-class metrics (precision, recall, f1, count)

    Parameters:
    - models_list: List of tuples (model_name, model)
    - val_ds: tf.data.Dataset (batched) with (images, labels)
    - id2label: dict {class_id: class_name}
    - output_folder: path to save Excel file
    """
    summary_rows = []
    detailed_rows = []

    for model_name, model, val_idx in models_list:
        val_ds = val_ds_list[val_idx]
        y_true = []
        y_pred = []
        y_pred_probs = []

        for images, labels in val_ds:
            probs = model.predict(images, verbose=0)
            preds = tf.argmax(probs, axis=1).numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_pred_probs.extend(probs)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_probs = np.array(y_pred_probs)

        # Overall metrics
        acc = accuracy_score(y_true, y_pred)
        top_2 = tf.keras.metrics.top_k_categorical_accuracy(
            tf.one_hot(y_true, depth=len(id2label)), y_pred_probs, k=2
        ).numpy().mean()
        top_3 = tf.keras.metrics.top_k_categorical_accuracy(
            tf.one_hot(y_true, depth=len(id2label)), y_pred_probs, k=3
        ).numpy().mean()

        summary_rows.append({
            "model": model_name,
            "accuracy": acc,
            "top_2_accuracy": top_2,
            "top_3_accuracy": top_3
        })

        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        true_counts = Counter(y_true)

        for idx, class_name in id2label.items():
            detailed_rows.append({
                "model": model_name,
                "class_id": idx,
                "class_name": class_name,
                "count": true_counts.get(idx, 0),
                "precision": precision[idx],
                "recall": recall[idx],
                "f1_score": f1[idx]
            })

    # Save to Excel
    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.DataFrame(detailed_rows)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "models_comparison_report.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df_summary.to_excel(writer, index=False, sheet_name="summary")
        df_detailed.to_excel(writer, index=False, sheet_name="per_class")

    print(f"\n📁 Report saved to: {output_path}")
    return df_summary, df_detailed
