from pathlib import Path
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.datasets import VisionDataset
from .solo_checkpoints import CHECKPOINTS_DB
from . import datasets


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    else:
        return image.convert("RGB")


def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return transforms.ToTensor()(x)


def get_default_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            _safe_to_tensor,
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def get_dataloaders(dataset, transform, batch_size, root_dir="data"):
    if transform is None:
        # just dummy resize -> both CLIP and DINO support 224 size of the image
        transform = get_default_transforms()
    train_dataset, val_dataset = get_datasets(dataset, transform, root_dir)
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    return trainloader, valloader


def get_mnist(data_dir, train=True, transform=None) -> VisionDataset:
    if transform is None:
        transform = transforms.ToTensor()
    return datasets.MNIST(
        root=data_dir, train=train, download=True, transform=transform
    )


def get_fashionmnist(data_dir, train=True, transform=None) -> VisionDataset:
    if transform is None:
        transform = transforms.ToTensor()
    return datasets.FashionMNIST(
        root=data_dir, train=train, download=True, transform=transform
    )


def get_cifar100(data_dir, train=True, transform=None) -> VisionDataset:
    if transform is None:
        transform = transforms.ToTensor()
    dataset = datasets.CIFAR100(
        root=data_dir, train=train, download=True, transform=transform
    )
    dataset.data = dataset.data.transpose(0, 3, 2, 1)
    return dataset


def get_cifar10(data_dir, train=True, transform=None) -> VisionDataset:
    if transform is None:
        transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )
    print(dataset.data.shape)
    dataset.data = dataset.data.transpose(0, 3, 2, 1)
    return dataset


def get_representation(root_dir, repr_name, dataset_name, split="train"):
    representation_dir = Path(root_dir).joinpath(repr_name)
    return datasets.RepresentationDataset(representation_dir, dataset_name, split)


def get_datasets(dataset_name, root_dir, transform=None):
    data_path = Path(root_dir).joinpath("datasets")

    if dataset_name == "cifar10":
        train_dataset = dsets.CIFAR10(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.CIFAR10(
            root=data_path, train=False, transform=transform, download=True
        )

    elif dataset_name == "cifar100":
        train_dataset = dsets.CIFAR100(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.CIFAR100(
            root=data_path, train=False, transform=transform, download=True
        )
    elif dataset_name == "caltech101":
        """
        Download manually from https://data.caltech.edu/records/mzrjq-6wc02
        and unzip to ./data/datasets/caltech-101/101_ObjectCategories
        """
        tmp_dataset = dsets.ImageFolder(
            root=str(data_path.joinpath("caltech-101/101_ObjectCategories")),
            transform=transform,
        )
        tmp_targets = np.array(tmp_dataset.targets)
        subset = []
        for t in np.unique(tmp_targets):
            np.random.seed(42)
            subset.extend(
                np.random.choice(np.where(tmp_targets == t)[0], size=30, replace=False)
            )
        subset_val = list(set([i for i in range(len(tmp_targets))]) - set(subset))
        train_dataset = Subset(tmp_dataset, subset)
        val_dataset = Subset(tmp_dataset, subset_val)

    elif dataset_name == "mnist":
        train_dataset = dsets.MNIST(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.MNIST(
            root=data_path, train=False, transform=transform, download=True
        )

    elif dataset_name == "fashionmnist":
        train_dataset = dsets.FashionMNIST(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.FashionMNIST(
            root=data_path, train=False, transform=transform, download=True
        )

    elif dataset_name == "stl10":
        train_dataset = dsets.STL10(
            root=data_path, split="train", transform=transform, download=True
        )
        val_dataset = dsets.STL10(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset_name == "eurosat":
        """
        Manually download wget https://madm.dfki.de/files/sentinel/EuroSAT.zip
        and then unzip to ./data/datasets/eurosat
        """
        tmp_dataset = dsets.EuroSAT(root=data_path, transform=transform, download=False)
        tmp_targets = np.array(tmp_dataset.targets)
        subset_train = []
        subset_val = []
        for t in np.unique(tmp_targets):
            np.random.seed(42)
            subset = np.random.choice(
                np.where(tmp_targets == t)[0], size=1500, replace=False
            )
            subset_train.extend(subset[:1000])
            subset_val.extend(subset[1000:])
        train_dataset = Subset(tmp_dataset, subset_train)
        val_dataset = Subset(tmp_dataset, subset_val)

    elif dataset_name == "sst":
        """
        Manually download and put to ./data/datasets/rendered-sst2
        wget https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz
        tar zxvf rendered-sst2.tgz
        """
        tmp_dataset1 = dsets.ImageFolder(
            root=str(data_path.joinpath("rendered-sst2/train")), transform=transform
        )
        tmp_dataset2 = dsets.ImageFolder(
            root=str(data_path.joinpath("rendered-sst2/valid")), transform=transform
        )
        train_dataset = ConcatDataset((tmp_dataset1, tmp_dataset2))
        val_dataset = dsets.ImageFolder(
            root=str(data_path.joinpath("rendered-sst2/test")), transform=transform
        )

    elif dataset_name == "imagenet":
        """
        Manually download from https://www.image-net.org/ and put to ./data/datasets/imagenet
        """
        train_dataset = dsets.ImageFolder(
            root=str(data_path.joinpath("imagenet/train")), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=str(data_path.joinpath("imagenet/val")), transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return train_dataset, val_dataset


def download_solo_checkpoint(
    method_name: str, dataset_name: str, checkpoints_dir, overwrite=False
):
    import gdown

    checkpoints_dir = Path(checkpoints_dir).joinpath("solo")
    checkpoints_dir.mkdir(exist_ok=True)

    entry = CHECKPOINTS_DB[dataset_name][method_name]
    checkpoint_url = entry["ckpt"]
    config_url = entry["config"]

    checkpoint_path = checkpoints_dir.joinpath(f"{dataset_name}_{method_name}.ckpt")
    config_path = checkpoints_dir.joinpath(f"{dataset_name}_{method_name}.json")
    if overwrite or (not checkpoint_path.exists()):
        print(
            "re-" * overwrite
            + f"downloading checkpoint {method_name} for {dataset_name}"
        )
        gdown.download(checkpoint_url, str(checkpoint_path))
        gdown.download(config_url, str(config_path))
    return checkpoint_path, config_path
