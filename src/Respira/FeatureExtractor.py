from .GreedyCTCDecoder import GreedyCTCDecoder
import torch
import torchaudio

class FeatureExtractor:
    def __init__(self):
        torch.random.manual_seed(0xbeef)
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = bundle.get_model().to(self.device)
        self.decoder = GreedyCTCDecoder(torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_labels())

        self.sample_rate = bundle.sample_rate

    def __call__(self, torchaudio_data, sample_rate):
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(torchaudio_data, sample_rate, self.sample_rate)

        # Get logits
        with torch.inference_mode():
            emission, _ = self.model(waveform)
        
        return emission

    def decode(self, emission):
        indices, transcript = self.decoder(emission)
        return indices, transcript
