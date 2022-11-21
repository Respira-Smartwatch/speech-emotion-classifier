import numpy as np
import os

from Respira import RavdessDataset, EmotionClassifier

if __name__ == "__main__":
    # Load dataset from disk (or create a new one)
    if not os.path.exists("dataset.bin"):
        dataset = RavdessDataset()
        dataset.save_to_disk("dataset.bin")
    else:
        dataset = RavdessDataset("dataset.bin")

    # Train model
    model = EmotionClassifier()
    dataloader = dataset.dataloader(batch_size=100)
    
    model.update_weights(dataloader, "results", batch_size=100, n_epochs=10)
