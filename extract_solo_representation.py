from pathlib import Path
import hydra
import torch
import numpy as np
from prcut.data.data_utils import download_solo_checkpoint
from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS
import hydra.core.global_hydra
from torch import nn
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Tuple
from solo.args.linear import parse_cfg
import warnings

warnings.filterwarnings("ignore")


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor, ...]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """
    model.eval()
    backbone_features, proj_features, labels = [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_features.append(outs["feats"].detach())
        proj_features.append(outs["z"])
        labels.append(lab)
    backbone_features = torch.cat(backbone_features)
    proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    return backbone_features, proj_features, labels


@hydra.main(version_base="1.2", config_path="./configs", config_name="solo")
def main(cfg: DictConfig):

    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    method_name = cfg.method
    dataset_name = cfg.dataset
    checkpoint_path, config_path = download_solo_checkpoint(
        method_name,
        dataset_name,
        cfg.checkpoints_dir,
        overwrite=cfg.overwrite,
    )
    representation_dir = Path(cfg.representations_dir).joinpath(cfg.method)
    representation_dir.mkdir(exist_ok=True)
    backbone = (
        METHODS[cfg["method"]]
        .load_from_checkpoint(checkpoint_path, strict=False, cfg=cfg)
        .cuda()
    )
    train_loader, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.dataset_dir,
        val_data_path=cfg.dataset_dir,
        data_format=cfg.data.format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=False,
    )

    train_data = extract_features(train_loader, backbone)
    test_data = extract_features(val_loader, backbone)
    for split, data in zip(["train", "val"], [train_data, test_data]):
        print(
            f"Saving to {representation_dir.joinpath(f'{dataset_name}_?_{split}.npy')}"
        )
        np.save(
            representation_dir.joinpath(f"{dataset_name}_feats_{split}.npy"),
            data[0].cpu().numpy(),
        )
        np.save(
            representation_dir.joinpath(f"{dataset_name}_feats_proj_{split}.npy"),
            data[1].cpu().numpy(),
        )
        np.save(
            representation_dir.joinpath(f"{dataset_name}_y_{split}.npy"),
            data[2].cpu().numpy(),
        )


if __name__ == "__main__":
    main()
