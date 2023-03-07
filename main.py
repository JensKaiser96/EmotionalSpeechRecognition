import torch

from src.models.models import train, annotate, CNNModel2, LinearModel
from src.data.data import get_train_loader, get_dev_loader


def main():
    # model, optimizer = simple_lstm(0.025)
    model = LinearModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)

    train(model, optimizer, get_train_loader(), num_epochs=50, path="data/models/small")
    annotate(model, get_dev_loader(), "data/dev_1.json")


if __name__ == '__main__':
    main()
