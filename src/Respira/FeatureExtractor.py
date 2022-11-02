from .GreedyCTCDecoder import GreedyCTCDecoder
import torch
import torchaudio

class FeatureExtractor:
    def __init__(self):
        torch.random.manual_seed(0xbeef)
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = bundle.get_model().to(self.device)
        self.decoder = GreedyCTCDecoder(bundle.get_labels())

        self.sample_rate = bundle.sample_rate

    def __call__(self, audio_path):
        # Open audio data and resample if necessary
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        # Get logits
        with torch.inference_mode():
            emission, _ = self.model(waveform)
        
        return emission[0]

    def decode(self, emission):
        transcript = self.decoder(emission)
        return transcript
