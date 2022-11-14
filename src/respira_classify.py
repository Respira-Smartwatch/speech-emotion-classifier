from Respira import EmotionClassifier, FeatureExtractor
import sys
import torch, torchaudio

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()

    model = EmotionClassifier()
    state_dict = torch.load("results/respira-emoc.bin")
    model.load_state_dict(state_dict)
    model.eval()

    audio_path = sys.argv[1]
    waveform, samplerate = torchaudio.load(audio_path)
    emission = feature_extractor(waveform, samplerate)

    # Collapse all timesteps into a single feature
    feature = torch.mean(emission, dim=1)

    logits = model(feature)[0].tolist()
    max_logit = logits.index(max(logits))
    category = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise"][max_logit]

    print(logits)
    print(category)
