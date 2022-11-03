import os
import torch

class EmotionClassifier(torch.nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()

        self.fc = torch.nn.Sequential (
            torch.nn.Linear(in_features=128, out_features=8),
            torch.nn.Softmax(dim=1)
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    def __accuracy_fn(logit, target, batch_size):
        """ Obtain accuracy for training round """
        n_correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * n_correct/batch_size
        return accuracy.item()

    def forward(self, x):
        return self.fc(x)

    def train(self, trainLoader, output_dir, learning_rate=1e-3,
                loss_fn=torch.nn.CrossEntropyLoss(), n_epochs=10):
        """
        Update the weights of the Emotion classifier

        features: [list]    List of (512x1) tensors of FeatureExtractor logits
        labels: [list]      List of integers corresponding to one of eight
                            emotion classes
        """

        # Setup torch environment
        torch.random.manual_seed(0xbeef)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        for epoch in range(n_epochs):
            running_loss = 0.0
            accuracy = 0.0

            model = self.train()

            for i, (features, labels) in enumerate(trainLoader):
                # Extract features/labels
                features = features.to(device)
                labels = labels.to(device)

                # Perform backpropagation
                logits = model(features)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()

                # Update model params
                optimizer.step()

                # Update metrics
                running_loss += loss.detach().item()
                accuracy += self.__accuracy_fn(logits, labels, batch_size=64)

            model.eval()

            print("Epoch: %d | Loss: %.4f | Train Accuracy: %.2f" % (epoch, running_loss / i, accuracy/i))

            output_filename = f"respira-emoc-{epoch}.bin"
            output_path = os.path.join(output_dir, output_filename)
            torch.save(model.state_dict, output_path)


