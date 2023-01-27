import soundfile
import librosa
import numpy as np
from numpy import ndarray


class FeatureExtractor:
    @staticmethod
    def from_samples(data, sample_rate):
        result = {}

        # Extract cepstral coefficients
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        result["mfcc"] = mfcc

        # Extract pitch
        stft = np.abs(librosa.stft(data))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma = np.mean(chroma.T, axis=0)
        result["chroma"] = chroma

        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        mel = np.mean(mel.T, axis=0)
        result["mel"] = mel

        return result

    @staticmethod
    def from_path(audio_path):
        # Load audio data and mix to mono
        with soundfile.SoundFile(audio_path) as file:
            data = librosa.load(path=file, sr=16000, mono=True)[0]

        return FeatureExtractor.from_samples(data, 16000)
