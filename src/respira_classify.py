from Respira import EmotionClassifier, FeatureExtractor
import sys
import torch

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    model = EmotionClassifier("results/respira-emoc.bin")
    
    audio_path = sys.argv[1]
    feature = torch.tensor(feature_extractor(audio_path))

    logits = model(feature)[0].tolist()

    max_logit = logits.index(max(logits))
    category = ["positive", "negative"][max_logit]

    print(logits)
    print(category)

