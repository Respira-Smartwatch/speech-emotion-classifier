from datasets import load_dataset
import json
from progress.bar import Bar
import torch
from torch.utils.data import DataLoader, Dataset
from Respira import FeatureExtractor, EmotionClassifier, RavdessDataset

if __name__ == "__main__":
    # Load dataset from disk
    dataset = RavdessDataset()
    dataset.load_from_disk("dataset.json")
    train_dataloader = DataLoader(dataset.to_pytorch(), batch_size=32, shuffle=True)

    # Train model
    model = EmotionClassifier()
    model.update_weights(train_dataloader, "results")
