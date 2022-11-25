from Respira import EmotionClassifier, FeatureExtractor
import sys
import torch, torchaudio

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    model = EmotionClassifier("results/respira-emoc.bin")
    
    audio_path = sys.argv[1]
    waveform, samplerate = torchaudio.load(audio_path)
    emission = feature_extractor(waveform, samplerate)

    # Collapse all timesteps into a single feature
    feature = torch.mean(emission, dim=1)

    logits = model(feature)[0].tolist()
    max_logit = logits.index(max(logits))
    category = ["positive", "negative"][max_logit]

    print(logits)
    print(category)
