from sklearn.metrics import confusion_matrix as cm
import seaborn as sn
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Respira import RavdessDataset, EmotionClassifier

def confusion_matrix(model, features, labels):
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
    plt.savefig('output.png')


if __name__ == "__main__":
    dataset = RavdessDataset("dataset.bin")
    model = EmotionClassifier("results/respira-emoc.bin")

    confusion_matrix(model, dataset.features, dataset.labels)