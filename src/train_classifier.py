import joblib
import matplotlib.pyplot as plt
import os

from Respira import RavdessDataset, EmotionClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier


def confusion_matrix(features, labels, out_path: str = "results/confusion_matrix.png"):
    ConfusionMatrixDisplay.from_predictions(features, labels)
    plt.savefig(out_path, format="png")


if __name__ == "__main__":
    # Load dataset from disk (or create a new one)
    if not os.path.exists("results/dataset.bin"):
        dataset = RavdessDataset()
        dataset.save_to_disk("results/dataset.bin")
    else:
        dataset = RavdessDataset("results/dataset.bin")

    x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=0.20)
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    confusion_matrix(y_test, y_pred)

    joblib.dump(model, "results/respira-emoc.bin")
