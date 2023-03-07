import json

import torch
from torch import nn
from tqdm import tqdm

from src.data.data import INPUT_SIZE, OUTPUT_SIZE, batch_size


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


class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size_s = (25, 5)

        self.conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size_s)
        self.pool = torch.nn.MaxPool1d(kernel_size=5)
        self.Wo = torch.nn.Linear(6, 2)

    def forward(self, x):
        #x = x.squeeze(dim=0)
        fs_ = [torch.relu(conv(x)) for conv in self.convs]
        as_ = [pool(f) for pool, f in zip(self.pools, fs_)]
        a = torch.cat(as_, dim=1)
        y = self.Wo(a)
        return y


class CNNModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(25, 1)),
            nn.ReLU(),
            nn.MaxPool2d((5, 3), 1),

            #nn.Conv2d(1, 1, kernel_size=(10, 1)),
            #nn.ReLU(),
            #nn.MaxPool2d((3, 3), 1),
        )
        self.lstm = nn.LSTM(24, 64, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(64, 2)

    def forward(self, x):
        #print(f"{x.shape=}")
        # x.shape = (fix<batch_size>, var<sequence_length>, fix<features, 26>)
        acnn = self.cnn(x)  # want shape (fix<batch_size>, var<sequence_length*?>, fix<features-?, 24>)
        #print(f"{acnn.shape=}")
        assert acnn.shape[0] == batch_size
        assert acnn.shape[2] == 24
        h_lstm, _ = self.lstm(acnn)
        h_lstm = h_lstm[:, -1, :]
        #print(f"{h_lstm.shape=}")
        h_linear = self.linear(h_lstm)
        #print(f"{h_linear.shape}")
        assert h_linear.shape == (1, 2)
        return torch.relu(h_linear)


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


def train(model, optimizer, train_data, num_epochs, report_intermediate_accuracy=True, path=""):
    loss = 0
    for epoch in range(num_epochs):
        for i, (mel_spec_features, labels) in enumerate(progress_bar := tqdm(train_data, desc="Training Model:")):
            optimizer.zero_grad()
            outputs = model(mel_spec_features)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if report_intermediate_accuracy and i % (len(train_data) / 10) == 0:
                print(f"Loss: {loss.item()}")  # \t Accuracy: {accuracy}")
    if path:
        torch.save(model.state_dict(), path)
    return loss


@torch.no_grad()
def annotate(model, data, path):
    buffer = {}
    for i, mel_spec_features in enumerate(data):
        output = model(mel_spec_features)
        valence = float(output[0, 0])
        activation = float(output[0, 1])
        buffer[i] = {"valence": valence, "activation": activation}
    with open(path, mode="w", encoding="utf-8") as f_out:
        json.dump(buffer, f_out)
