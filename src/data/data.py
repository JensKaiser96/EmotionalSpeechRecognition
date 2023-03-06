from typing import List
import matplotlib.pyplot as plt
import pandas
import torch

INPUT_SIZE = 26
OUTPUT_SIZE = 2

VALENCE = "valence"
ACTIVATION = "activation"
FEATURES = "features"

batch_size = 1

if __name__ != "__main__":
    dataset_path = "./data/"
else:
    dataset_path = "./../../data/"
train_dataset_path = dataset_path + "train.json"
dev_dataset_path = dataset_path + "dev.json"


class SERDataset(torch.utils.data.Dataset):
    def __init__(self, path, train):
        self.data_frame = pandas.read_json(path)
        self.train = train

    def __len__(self):
        return self.data_frame.shape[1]

    def __getitem__(self, item):
        data_point = self.data_frame[item]
        features = torch.tensor(data_point[FEATURES])
        if self.train:
            return features, torch.tensor([data_point[VALENCE], data_point[ACTIVATION]], dtype=torch.float)
        else:
            return features


def get_train_loader():
    print("Loading training data...")
    train_loader = torch.utils.data.DataLoader(
        dataset=SERDataset(train_dataset_path, train=True),
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader


def get_dev_loader():
    print("Loading development data...")
    dev_loader = torch.utils.data.DataLoader(
        dataset=SERDataset(dev_dataset_path, train=False),
        batch_size=batch_size,
        shuffle=False
    )
    return dev_loader


def plot(data: SERDataset = SERDataset(train_dataset_path, train=True), data_points: List[int] = None):
    if data_points is None:
        data_points = [0, 1, 2, 3]
    for data_point in data_points:
        fig, ax = plt.subplots()

        spec, (val, act) = data[data_point]
        c = ax.pcolor(spec)
        ax.set_title(f"{val=}, {act=}")

        plt.show()


if __name__ == '__main__':
    dataset_path = "./../../data/"
    plot()
