import pandas
import torch

INPUT_SIZE = 26
OUTPUT_SIZE = 2

VALENCE = "valence"
ACTIVATION = "activation"
FEATURES = "features"

batch_size = 256

if __name__ != "__main__":
    dataset_path = "./data/"
else:
    dataset_path = "./../../data/"
train_dataset_path = dataset_path + "train.json"
small_train_dataset_path = dataset_path + "first1k_train.json"
dev_dataset_path = dataset_path + "dev.json"


class SERDataset(torch.utils.data.Dataset):
    def __init__(self, path, return_labels=True, simple=False):
        self.data_frame = pandas.read_json(path)
        self.return_labels = return_labels
        self.simple = simple

    def __len__(self):
        return self.data_frame.shape[1]

    def __getitem__(self, item):
        data_point = self.data_frame[item]
        if self.simple:
            mean = torch.tensor(data_point[FEATURES]).mean(dim=0)
            var = torch.tensor(data_point[FEATURES]).var(dim=0)
            features = torch.cat((mean, var))
        else:
            features = data_point[FEATURES]
        if self.return_labels:
            labels = torch.tensor([data_point[VALENCE], data_point[ACTIVATION]], dtype=torch.float)
            return features, labels
        else:
            return features


def collate_fn(batch):
    # batch = [(features, label), (f2, l2), ...]
    features = [torch.tensor(feature) for feature, _ in batch]
    labels = [label for _, label in batch]
    t_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    return t_features, torch.stack(labels)


def get_train_loader(small=False):
    print("Loading training data...")
    if small:
        dataset_path = small_train_dataset_path
    else:
        dataset_path = train_dataset_path
    train_loader = torch.utils.data.DataLoader(
        dataset=SERDataset(dataset_path),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_loader


def get_dev_loader():
    print("Loading development data...")
    dev_loader = torch.utils.data.DataLoader(
        dataset=SERDataset(dev_dataset_path, return_labels=False),
        batch_size=1,
        shuffle=False
    )
    return dev_loader


def plot(data: SERDataset = SERDataset(train_dataset_path), data_points = None):
    import matplotlib.pyplot as plt
    if data_points is None:
        data_points = [0, 1, 2, 3]
    for data_point in data_points:
        fig, ax = plt.subplots()

        spec, (val, act) = data[data_point]
        c = ax.pcolor(spec)
        ax.set_title(f"{val=}, {act=}")

        plt.show()
