import numpy as np
import os

from Respira import RavdessDataset, EmotionClassifier

from sklearn.metrics import confusion_matrix as cm
import seaborn as sn
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(model, features, labels, out_path: str = "results/output.png"):
    y_pred = []

    for feature in features:
        logits = model(feature).tolist()
        max_logit = logits.index(max(logits))
        y_pred.append(max_logit)

    classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise"]

    # Build confusion matrix   
    cf_matrix = cm(labels, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 8, index = [i for i in classes],
                            columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(out_path)

if __name__ == "__main__":
    # Load dataset from disk (or create a new one)
    if not os.path.exists("results/dataset.bin"):
        dataset = RavdessDataset()
        dataset.save_to_disk("results/dataset.bin")
    else:
        dataset = RavdessDataset("results/dataset.bin")

    # Train model using cross-validation strategy
    results_dir = f"results"
    accuracy_path = os.path.join(results_dir, f"accuracy.txt")
    accuracy_file = open(accuracy_path, "w")

    for i in range(5):
        print(f"Training on Fold {i}...")

        model = EmotionClassifier()
        train, test, test_dict = dataset.cv_fold(i, batch_size=100, shuffle=True)

        model.update_weights(train, results_dir, batch_size=100)
        accuracy = model.evaluate(test)

        accuracy_file.write(f"Fold {i}: {accuracy}%\n")

        confusion_path = os.path.join(results_dir, f"cv{i}-confusion_matrix.png")
        confusion_matrix(model, test_dict["features"], test_dict["labels"], confusion_path)

        model_path = os.path.join(results_dir, f"cv{i}-model.bin")
        model.save_model(model_path)

        print(f"Test accuracy: {accuracy} %")
        print()
