import torch
import torchaudio

# HACK: Bypass SSL verification to download PyTorch models on firewalled machines
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class FeatureExtractor:
    def __init__(self):
        torch.random.manual_seed(0xbeef)
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = bundle.get_model().to(self.device)
        self.sample_rate = bundle.sample_rate

    def __call__(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        # Get logits
        with torch.inference_mode():
            waveform = waveform.to(self.device)
            emission, _ = self.encoder(waveform)
         
        return emission[0].clone().detach()

    def from_samples(self, samples):
        with torch.inference_mode():
            waveform = samples.to(self.device)
            emission, _ = self.encoder(waveform)

        return emission[0].clone().detach()

