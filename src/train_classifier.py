#from datasets import load_dataset
#import json
#from progress.bar import Bar
import os
import torch
from torch.utils.data import DataLoader, Dataset
from Respira import RavdessDataset, EmotionClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np
from sklearn.preprocessing import StandardScaler

def classifier_accuracy(model, features, labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(labels)):
        sample = features[i]

        # Reshape to 2D matrix if necessary
        if sample.shape[0] == 1:
            sample = sample.reshape(-1, 1)
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)

        # Predict and evaluate
        prediction = model.predict(sample)

        if prediction == labels[i] and prediction == True:
            TP += 1
        elif prediction == labels[i] and prediction == False:
            TN += 1
        elif prediction != labels[i] and prediction == True:
            FP += 1
        else:
            FN += 1

        accuracy = (TP + TN) / (TP + FP + TN + FN)

    return accuracy

if __name__ == "__main__":
    # Load dataset from disk
    if not os.path.exists("sdataset.bin"):
        dataset = RavdessDataset()
        dataset.save_to_disk("sdataset.bin")
    else:
        dataset = RavdessDataset("dataset.bin")

    # Train model
    model = EmotionClassifier()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    model.update_weights(dataloader, "results", batch_size=32, n_epochs=100)
