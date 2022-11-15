#from datasets import load_dataset
#import json
#from progress.bar import Bar
import os
import torch
from Respira import RavdessDataset, EmotionClassifier
import numpy as np

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
