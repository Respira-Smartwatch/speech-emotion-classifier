import os
import progressbar
import requests
import tempfile
import torch, torchaudio

from torch.utils.data import Dataset, DataLoader
from zipfile import ZipFile

from Respira import FeatureExtractor

class RavdessDataloader(Dataset):
    def __init__(self, aggregate_data: dict):
        self.features = aggregate_data["features"]
        self.labels = aggregate_data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RavdessDataset():
    def __init__(self, path: str = None):
        home_dir = os.path.expanduser("~")
        self.cache_dir = os.path.join(home_dir, ".cache/respira/ravdess_extracted")

        self.actors = []

        # Load existing dataset if specified
        if path != None:
            print(f"Using existing dataset at {path}")
            data = torch.load(path)

            for i in range(24):
                actor_key = f"actor{i}"
                self.actors.append(data[actor_key])

        # Process raw data if it is cached
        elif os.path.exists(self.cache_dir):
            print(f"Building dataset using raw data cached at {self.cache_dir}")
            self.__process_raw_data()

        # Download raw data and process it
        else:
            print(f"Downloading raw data to {self.cache_dir}")
            self.__download_raw_data()
            self.__process_raw_data()

    def __download_raw_data(self):
        # Download dataset from official repository
        dataset_url = "https://www.zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        response = requests.get(dataset_url)

        temp = tempfile.NamedTemporaryFile()
        temp.write(response.content)

        with ZipFile(temp.name, "r") as zip:
            zip.extractall(self.cache_dir)

        temp.close()

    def __process_raw_data(self):
        # Gather all 24 actor directories
        actor_dirs = [dir for dir in os.listdir(self.cache_dir) if "Actor_" in dir]
        assert(len(actor_dirs) == 24)

        # Display a progress bar, as process may take some time
        #bar = Bar("Building dataset: ", max=24*60, suffix="%(percent).1f%% - %(eta_td)s")
        bar = progressbar.ProgressBar(max_value=1440).start()
        bar_idx = 0

        # Instantiate FeatureExtractor and model to convert audio data to logits
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = FeatureExtractor()

        # Extract and tag audio files from each actor, in order
        for i in range(1, len(actor_dirs) + 1):
            # Enter the actor directory and check that it contains 60 audio files
            actor_path = os.path.join(self.cache_dir, f"Actor_{i:02}")
            audios = os.listdir(actor_path)
            assert(len(audios) == 60)

            # For each audio file, extract Wav2Vec2 features and the label
            features = []
            labels = []

            for audio in audios:
                # Extract features for all timesteps then collapse into single feature vector
                audio_path = os.path.join(actor_path, audio)
                waveform, samplerate = torchaudio.load(audio_path)
                waveform.to(device)
                emission = feature_extractor(waveform, samplerate)

                # The emission is a (batch_size x timesteps x 1024) list
                # The following line collapses all of the timesteps into a
                # (batch_size x 1024) list, where batch_size=1
                feature = torch.mean(emission, dim=1)[0]

                # Determine label from filename
                label = int(audio.split("-")[2]) - 1

                features.append(feature)
                labels.append(label)

                bar.update(bar_idx)
                bar_idx += 1

            self.actors.append({"features": features, "labels": labels})
        bar.finish()

    def save_to_disk(self, output_path: str):
        aggregate_data = {}

        for i, actor in enumerate(self.actors):
            aggregate_data[f"actor{i}"] = actor
        
        torch.save(aggregate_data, output_path)

    def dataloader(self, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        # Compile aggregate data from all actors
        aggregate_data = {
            "features": [],
            "labels": []
        }

        for actor in self.actors:
            aggregate_data["features"] += actor["features"]
            aggregate_data["labels"] += actor["labels"]

        # Build Dataloader
        return DataLoader(RavdessDataloader(aggregate_data), batch_size=batch_size, shuffle=shuffle)

    def cv_fold(self, fold: int, batch_size: int = 1, shuffle: bool = False):
        if fold == 0:
            actors = [1, 4, 13, 14, 15]
        elif fold == 1:
            actors = [2, 5, 6, 12, 17]
        elif fold == 2:
            actors = [9, 10, 11, 18, 19]
        elif fold == 3:
            actors = [7, 16, 20, 22, 23]
        else:
            actors = [0, 3, 9, 21]

        test_aggregate_data = {
            "features": [],
            "labels": []
        }

        train_aggregate_data = {
            "features": [],
            "labels": []
        }

        for i, actor in enumerate(self.actors):
            if i in actors:
                test_aggregate_data["features"] += actor["features"]
                test_aggregate_data["labels"] += actor["labels"]
            else:
                train_aggregate_data["features"] += actor["features"]
                train_aggregate_data["labels"] += actor["labels"]

        train_dataloader = DataLoader(RavdessDataloader(train_aggregate_data), batch_size=batch_size, shuffle=shuffle) 
        test_dataloader = DataLoader(RavdessDataloader(test_aggregate_data), batch_size=1, shuffle=shuffle)

        return train_dataloader, test_dataloader, test_aggregate_data
        