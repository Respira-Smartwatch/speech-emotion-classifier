from datasets import load_dataset
import json
from progress.bar import Bar
import torch
from torch.utils.data import Dataset
from Respira import FeatureExtractor

class RavdessDataClass(Dataset):
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

class RavdessDataset:
    def __init__(self):
        self.data = {
            "features": [],
            "labels": []
        }

    def load_from_remote(self):
        # Download RAVDESS dataset
        ravdess = load_dataset("narad/ravdess", split="train")
        ravdess = ravdess.train_test_split(test_size=0.2, seed=0xbeef)["train"]
        ravdess = ravdess.remove_columns(["text", "speaker_id", "speaker_gender"])

        # Instantiate FeatureExtractor to generate logits for classifier input
        feature_extractor = FeatureExtractor()

        features = []
        labels = []

        # Get progress bar
        bar = Bar("Building dataset", max=len(ravdess["labels"]), suffix="%(percent).1f%% - %(eta_td)s")

        for audio, label in zip(ravdess["audio"], ravdess["labels"]):
            # Extract feature/label from dataset
            path = audio["path"]
            emission = feature_extractor(path)

            # Get logits from FeatureExtractor
            logits, _ = feature_extractor.decode(emission)

            logits = [x.item() for x in logits]
            
            # Pad or truncate feature to 512-length
            logits_len = len(logits)
            if logits_len <= 128:
                n_add = 128 - logits_len
                logits += [0] * n_add
            else:
                n_remove = logits_len - 128
                logits = logits[:n_remove]

            features.append(logits)
            labels.append(label)

            bar.next()
        bar.finish()

        # Store internally
        self.data["features"] = features
        self.data["labels"] = labels

    def load_from_disk(self, input_path):
        data = json.load(open(input_path, "r"))
        self.data["features"] = data["features"]
        self.data["labels"] = data["labels"]
            
    def serialize(self, output_path:str):
        with open(output_path, "w") as outfile:
            json.dump(self.data, outfile)

    def to_pytorch(self):
        features = self.data["features"]
        labels = self.data["labels"]

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        print(labels)

        return RavdessDataClass(features, labels)

if __name__ == "__main__":
    dataset = RavdessDataset()
    dataset.load_from_remote()
    dataset.serialize("dataset.json")
