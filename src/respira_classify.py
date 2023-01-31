import numpy as np
import sys

from Respira import EmotionClassifier, FeatureExtractor

if __name__ == "__main__":
    model = EmotionClassifier("results/respira-emoc.bin")
    
    audio_path = sys.argv[1]
    emission = FeatureExtractor.from_path(audio_path)
    emission = np.hstack((emission["mfcc"], emission["chroma"], emission["mel"]))

    prediction, probabilities = model([emission])

    category = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise"][prediction[0]]
    probability = max(probabilities[0]) * 100

    print(f"{prediction} ({probability} %)")
