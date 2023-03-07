import torch

from src.models.models import train, annotate, CNNModel2
from src.data.data import get_train_loader, get_dev_loader


def main():
    # model, optimizer = simple_lstm(0.025)
    model = CNNModel2()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)

    train(model, optimizer, get_train_loader(small=True), num_epochs=5, path="data/models/small")
    annotate(model, get_dev_loader(), "data/dev_1.json")


if __name__ == '__main__':
    main()
