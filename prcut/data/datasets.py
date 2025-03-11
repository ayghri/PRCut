from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons
import torch
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10

# from sklearn.preprocessing import StandardScaler
import torch.utils.data as data


class TensorDataset(data.Dataset):

    _repr_indent = 4

    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets
        assert self.data.shape[0] == self.targets.shape[0]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(self._extra_repr())
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _extra_repr(self) -> str:
        return ""

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return self.data.shape[0]


class RepresentationDataset(TensorDataset):

    _repr_indent = 4

    def __init__(
        self,
        repr_dir: Path,
        dataset_name: str,
        split="train",
    ) -> None:
        self.root = repr_dir
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(
            torch.tensor(
                np.load(repr_dir.joinpath(f"{dataset_name}_feats_{split}.npy"))
            ).float(),
            torch.tensor(
                np.load(repr_dir.joinpath(f"{dataset_name}_y_{split}.npy"))
            ).float(),
        )
        self.dim = self.data.shape[1]

    def _extra_repr(self):
        return f"Root location: {self.root}"


class MoonDataset(TensorDataset):

    def __init__(self, n_samples, noise=0.1, shuffle=True, random_state=None) -> None:

        data, y = make_moons(
            n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state
        )
        super().__init__(
            torch.from_numpy(data).float(),
            torch.from_numpy(y),
        )


# def load_twomoon(noise=0.1) -> tuple:
#     data, y = make_moons(n_samples=7000, shuffle=True, noise=noise, random_state=None)
#     scaler = StandardScaler()
#     data = scaler.fit_transform(data)
#     x_train, x_test, y_train, y_test = train_test_split(
#         data, y, test_size=0.33, random_state=42
#     )
#     x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
#     y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
#     return x_train, y_train, x_test, y_test


# def load_full_mnist(data_dir):
#     train = MNIST(root=data_dir, train=True, download=True)
#     test = MNIST(root=data_dir, train=False, download=True)
#     x = torch.cat(
#         [
#             train.data.float().view(-1, 784) / 255.0,
#             test.data.float().view(-1, 784) / 255.0,
#         ],
#         0,
#     )
#     y = torch.cat([train.targets, test.targets], 0)
#     dataset = dict()
#     dataset["x"] = x
#     dataset["y"] = y
#     return dataset


# def load_cifar10_old() -> tuple:
#     train_set = datasets.CIFAR10(
#         root="../data", train=True, download=True, transform=tensor_transform
#     )
#     test_set = datasets.CIFAR10(
#         root="../data", train=False, download=True, transform=tensor_transform
#     )
#
#     # print(train_set)
#     x_train, y_train = zip(*train_set)
#     x_train, y_train = (
#         torch.cat(x_train, dim=0).view(-1, 32 * 32 * 3),
#         torch.Tensor(y_train),
#     )
#     x_test, y_test = zip(*test_set)
#     x_test, y_test = (
#         torch.cat(x_test, dim=0).view(-1, 32 * 32 * 3),
#         torch.Tensor(y_test),
#     )
#
#     return x_train, y_train, x_test, y_test
#
#


# def load_reuters() -> tuple:
#     with h5py.File("./datasets/reutersidf_total.h5", "r") as f:
#         x = np.asarray(f.get("data"), dtype="float32")
#         y = np.asarray(f.get("labels"), dtype="float32")
#
#         n_train = int(0.9 * len(x))
#         x_train, x_test = x[:n_train], x[n_train:]
#         y_train, y_test = y[:n_train], y[n_train:]
#
#     x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
#     y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
#
#     return x_train, y_train, x_test, y_test


# def load_data(dataset: str) -> tuple:
#     """
#     This function loads the dataset specified in the config file.
#
#
#     Args:
#         dataset (str or dictionary):    In case you want to load your own dataset,
#                                         you should specify the path to the data (and label if applicable)
#                                         files in the config file in a dictionary fashion under the key "dataset".
#
#     Raises:
#         ValueError: If the dataset is not found in the config file.
#
#     Returns:
#         tuple: A tuple containing the train and test data and labels.
#     """
#
#     if dataset == "mnist":
#         x_train, y_train, x_test, y_test = load_mnist()
#     elif dataset == "twomoons":
#         x_train, y_train, x_test, y_test = load_twomoon()
#     elif dataset == "reuters":
#         x_train, y_train, x_test, y_test = load_reuters()
#     else:
#         try:
#             data_path = dataset["dpath"]
#             if "lpath" in dataset:
#                 label_path = dataset["lpath"]
#             else:
#                 label_path = None
#         except:
#             raise ValueError("Could not find dataset path. Check your config file.")
#         x_train, y_train, x_test, y_test = load_from_path(data_path, label_path)
#
#     return x_train, x_test, y_train, y_test


# def load_from_path(dpath: str, lpath: str) -> tuple:
#     """
#     Load data from files specified by dpath and lpath.
#
#     Parameters:
#     - dpath (str): Path to the data file.
#     - lpath (str): Path to the label file.
#
#     Returns:
#     - tuple: x_train, y_train, x_test, y_test
#     """
#     X = np.loadtxt(dpath, delimiter=",", dtype=np.float32)
#     n_train = int(0.9 * len(X))
#
#     x_train, x_test = X[:n_train], X[n_train:]
#     x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
#
#     if lpath is not None:
#         y = np.loadtxt(lpath, delimiter=",", dtype=np.float32)
#         y_train, y_test = y[:n_train], y[n_train:]
#         y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
#
#     else:
#         y_train, y_test = None, None
#
#     return x_train, y_train, x_test, y_test
