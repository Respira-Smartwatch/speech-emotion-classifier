import joblib
from sklearn.metrics import accuracy_score


class EmotionClassifier:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def __call__(self, features):
        return self.model.predict(features), self.model.predict_proba(features)

    def evaluate(self, features, labels):
        predictions = self.model.predict(features)
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        return accuracy
