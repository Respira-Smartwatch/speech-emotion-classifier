import samplerate
from soundfile import SoundFile
import torch
from transformers import AutoModelForCTC, AutoFeatureExtractor, AutoTokenizer, AutoProcessor

class FeatureExtractor:
    def __init__(self):
        # The feature extractor converts 16 kHz audio data into a format acceptable to the model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h", torchscript=True)
        
        # The model produces a vector of word embeddings from an input sequence
        self.model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h", torchscript=True)
        
        # The processor decodes model outputs, translating word embeddings into their lexical equivalents
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h", torchscript=True)

        # Initialize the model
        self.model.eval()

    def __samples_from_audio(self, audio_path):
        # Import raw audio data
        sound = SoundFile(audio_path)
        audio_data = sound.read()

        # Resample to 16 kHz
        ratio = 16_000 / sound.samplerate
        audio_data = samplerate.resample(audio_data, ratio, "sinc_fastest")

        # Extract features
        input_values = self.feature_extractor(audio_data, return_tensors="pt", sampling_rate=16_000).input_values

        return input_values

    def __call__(self, audio_path):
        input_values = self.__samples_from_audio(audio_path)

        logits = self.model(input_values)[0]
        pred_ids = torch.argmax(logits, axis=-1)[0]

        return pred_ids

    def decode(self, pred_ids):
        words = self.processor.decode(pred_ids)
        return words

    def test_model(self, dataloader):
        loss_fn = torch.nn.MSELoss

        size = len(dataloader.features)
        n_batches = len(dataloader)

        test_loss, n_correct = 0, 0

        with torch.no_grad():
            for x, y in dataloader:

                print(f"Input: {x}")
                print(f"Label: {y}")
                pred = self.__call__(x)
                print(f"Prediction: {pred}")
                test_loss += loss_fn(pred, y).item()

                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= n_batches
        n_correct /= size

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
