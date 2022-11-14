import os
import torch
import torchaudio

class EmotionClassifier(torch.nn.Module):
    def __init__(self):
        torch.random.manual_seed(0xbeef)
        super(EmotionClassifier, self).__init__()

        self.fc = torch.nn.Sequential (
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 1024),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 8)
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    def __accuracy_fn(self, logit, target, batch_size):
        """ Obtain accuracy for training round """
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()

    def forward(self, x):
        return self.fc(x)

    def update_weights(self, trainLoader, output_dir, batch_size=1, learning_rate=1e-3,
                loss_fn=torch.nn.CrossEntropyLoss(), n_epochs=10):
        # Make results dir
        if not os.path.exists("results"):
            os.makedirs("results")

        # Setup torch environment
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
                accuracy += self.__accuracy_fn(logits, labels, batch_size)

            model.eval()

            print("Epoch: %d | Loss: %.4f | Train Accuracy: %.2f" % (epoch, running_loss / i, accuracy/i))

        output_filename = f"respira-emoc.bin"
        output_path = os.path.join(output_dir, output_filename)
        torch.save(self.state_dict(), output_path)
