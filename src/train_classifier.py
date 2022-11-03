from datasets import load_dataset
from progress.bar import Bar
import torch
from torch.utils.data import DataLoader
from Respira import FeatureExtractor, EmotionClassifier

class RavdessDataset(Dataset):
    def __init__(self, features, labels, transform=None, target_transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        return feature, label

if __name__ == "__main__":
    # Download RAVDESS dataset
    ravdess = load_dataset("narad/ravdess", split="train")
    ravdess = ravdess.train_test_split(test_size=0.2, seed=0xbeef)["test"]
    ravdess = ravdess.remove_columns(["text", "speaker_id", "speaker_gender"])

    # Instantiate FeatureExtractor to generate logits for classifier input
    feature_extractor = FeatureExtractor()

    features = []
    labels = []

    # Get progress bar
    bar = Bar("Building dataset", max=len(ravdess["labels"]), suffix="%(percent).1f%% - %(eta_td)ds")

    for audio, label in zip(ravdess["audio"], ravdess["labels"]):
        # Extract feature/label from dataset
        path = audio["path"]
        emission = feature_extractor(path)

        # Get logits from FeatureExtractor
        logits, _ = feature_extractor.decode(emission)
        
        # Pad or truncate feature to 512-length
        logits_len = len(logits)
        if logits_len <= 128:
            n_add = 128 - logits_len
            logits += [torch.tensor(0)] * n_add
        else:
            n_remove = logits_len - 128
            logits = logits[:n_remove]

        features.append(logits)
        labels.append(label)

        bar.next()
    bar.finish()

    bar = Bar("Verifying dataset", max=len(features), suffix="%(percent).1f%% - %(eta_td)ds")
    for feature in features:
        assert(len(feature) == 128)
        bar.next()
    bar.finish()

    # Convert to PyTorch dataloader and export
    training_data = RavdessDataset(features, labels)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
