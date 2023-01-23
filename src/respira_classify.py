from Respira import EmotionClassifier, FeatureExtractor
import sys

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    model = EmotionClassifier("results/respira-emoc.bin")
    
    audio_path = sys.argv[1]
    feature = feature_extractor(audio_path)

    logits = model(feature)[0].tolist()

    max_logit = logits.index(max(logits))
    category = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise"][max_logit]

    print(logits)
    print(category)

