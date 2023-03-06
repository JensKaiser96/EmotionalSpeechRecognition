import torch
from tqdm import tqdm

from src.data.data import INPUT_SIZE, OUTPUT_SIZE


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out


def simple_lstm(learning_rate: float = 0.1):
    print("Creating simple LSTM model...")
    model = LSTMModel(
        input_dim=INPUT_SIZE,
        hidden_dim=INPUT_SIZE,
        layer_dim=1,
        output_dim=OUTPUT_SIZE
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer


def train(model, optimizer, train_data, dev_data, num_epochs, report_intermediate_accuracy=True):
    loss = 0
    for epoch in range(num_epochs):
        for i, (mel_spec_features, labels) in enumerate(progress_bar := tqdm(train_data, desc="Training Model:")):
            optimizer.zero_grad()
            outputs = model(mel_spec_features)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if report_intermediate_accuracy and i % (len(train_data) / 10) == 0:
                #accuracy = evaluate(model, dev_data)
                print(f"Loss: {loss.item()}") # \t Accuracy: {accuracy}")
    return loss


def smart_train(model, optimizer, train_data, dev_data, min_epochs, max_tries):
    best_accuracy = 0
    accuracy = 0
    tries = 0
    epoch = 0
    while tries <= max_tries or epoch <= min_epochs:
        epoch += 1
        loss = train(model, optimizer, train_data, dev_data, 1, report_intermediate_accuracy=False)
        accuracy = evaluate(model, dev_data)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            tries = 0
        else:
            tries += 1
        print(f"Epoch: {epoch} \t Loss: {loss * 100:05.2f} \t Accuracy: {accuracy:05.2f}")


@torch.no_grad()
def evaluate(model, data) -> float:
    # Calculate Accuracy
    correct = 0
    total = 0
    # Iterate through test dataset
    for mel_spec_features, labels in data:
        outputs = model(mel_spec_features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    return accuracy
