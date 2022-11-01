import samplerate
from soundfile import SoundFile
import torch
from transformers import AutoModelForCTC, AutoFeatureExtractor, AutoTokenizer

class FeatureExtractor:
    def __init__(self):
        # The feature extractor converts 16 kHz audio data into a format acceptable to the model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h", torchscript=True)
        
        # The model produces a vector of word embeddings from an input sequence
        self.model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h", torchscript=True)
        
        # The tokenizer decodes model outputs, translating word embeddings into their lexical equivalents
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h", torchscript=True)

        # Initialize the model
        self.model.eval()

    def __samples_from_audio(self, audio_path):
        # Import raw audio data
        sound = SoundFile(audio_path)
        audio_data = sound.read(always_2d=False)

        # Resample to 16 kHz
        ratio = 16_000 / sound.samplerate
        audio_data = samplerate.resample(audio_data, ratio, "sinc_fastest")

        # Extract features
        input_values = self.feature_extractor(audio_data, return_tensors="pt", sampling_rate=16_000).input_values

        return input_values

    def __call__(self, audio_path):
        input_values = self.__samples_from_audio(audio_path)

        logits = self.model(input_values)[0]
        pred_ids = torch.argmax(logits, axis=-1)

        return pred_ids[0]

    def decode(self, pred_ids):
        outputs = self.tokenizer.decode(pred_ids, output_word_offsets=True)
        time_offset = self.model.config.inputs_to_logits_ratio / self.feature_extractor.sampling_rate
        
        word_offsets = [
            {
                "word": d["word"],
                "start_time": round(d["start_offset"] * time_offset, 2),
                "end_time": round(d["end_offset"] * time_offset, 2),
            }
                for d in outputs.word_offsets
        ]

        words = []
        for word_offset in word_offsets:
            words.append(word_offset["word"])

        return words
