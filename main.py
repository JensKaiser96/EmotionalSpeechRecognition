from src.model.models import simple_lstm, train, smart_train
from src.data.data import get_train_loader, get_dev_loader


def main():
    model, optimizer = simple_lstm(0.025)

    smart_train(model, optimizer, get_train_loader(), get_dev_loader(), 5, 5)


if __name__ == '__main__':
    main()
