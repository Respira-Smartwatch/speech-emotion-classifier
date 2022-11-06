from datasets import load_dataset
from progress.bar import Bar
from Respira import FeatureExtractor
import torch, torchaudio
from torch.utils.data import Dataset

class RavdessDataset(Dataset):
    def __init__(self, path=None):
        self.features = []
        self.labels = []

        if (path == None):
            self.__populate_from_remote()
        else:
            self.__populate_from_disk(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label

    def __populate_from_remote(self):
        # Download RAVDESS dataset and extract trainm split
        # TODO: Split into Train(Actors 0-19) and Test(Actors20-24)
        ravdess = load_dataset("narad/ravdess", split="train")
        ravdess = ravdess.train_test_split(test_size=0.2, seed=0xbeef)["train"]
        ravdess = ravdess.remove_columns(["text", "speaker_id", "speaker_gender"])
        ravdess = ravdess.with_format("torch")

        # Instantiate FeatureExtractor and model to convert audio data to logits
        feature_extractor = FeatureExtractor()

        # Instantiate progress bar, as process may take some time
        bar = Bar("Building dataset", max=len(ravdess["labels"]), suffix="%(percent).1f%% - %(eta_td)s")

        # Build dataset
        i = 0
        for audio, label in zip(ravdess["audio"], ravdess["labels"]):
            # Extract audio data and features
            audio_path = audio["path"]
            waveform, samplerate = torchaudio.load(audio_path)
            emission = feature_extractor(waveform, samplerate)

            # Collapse all timesteps into a single feature
            feature = torch.mean(emission[0], 0)

            self.features.append(feature)
            self.labels.append(label)

            bar.next()
        bar.finish()

    def __populate_from_disk(self, path):
        data = torch.load(path)

        self.features = data["features"]
        self.labels = data["labels"]

        print(f"Imported {len(self.features)} features")

    def save_to_disk(self, path):
        with open(path, "w") as outfile:
            data = {
                "features": self.features,
                "labels": self.labels
            }

            torch.save(data, path)
