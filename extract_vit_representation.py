from pathlib import Path
import argparse
import random
from tqdm import tqdm
import numpy as np
import torch
import clip


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from prcut.data.data_utils import (
    get_dataloaders,
    _convert_image_to_rgb,
    _safe_to_tensor,
)

name2model = {
    "clipRN50": "RN50",
    "clipRN101": "RN101",
    "clipRN50x4": "RN50x4",
    "clipRN50x16": "RN50x16",
    "clipRN50x64": "RN50x64",
    "clipvitB32": "ViT-B/32",
    "clipvitB16": "ViT-B/16",
    "clipvitL14": "ViT-L/14",
    "dinov2": "facebookresearch/dinov2",
}


def load_model(model_name: str, device, root_dir):
    assert model_name in name2model.keys()

    root_dir = Path(root_dir)
    checkpoint_dir = root_dir.joinpath("checkpoints")
    preprocess = None
    if model_name == "dinov2":
        checkpoint_dir = checkpoint_dir.joinpath("dinov2")
        checkpoint_dir.mkdir(exist_ok=True)
        torch.hub.set_dir(checkpoint_dir)
        model = torch.hub.load(name2model[model_name], "dinov2_vitg14").to(device)
        model.eval()
        print(
            "Model parameters:",
            f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
        )

    else:
        checkpoint_dir = checkpoint_dir.joinpath("clip")
        checkpoint_dir.mkdir(exist_ok=True)
        model, preprocess = clip.load(
            name2model[model_name],
            device=device,
            download_root=str(checkpoint_dir),
        )
        model.eval()
        model = model.encode_image
        preprocess.transforms[2] = _convert_image_to_rgb
        preprocess.transforms[3] = _safe_to_tensor

    return model, preprocess


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to precompute embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="clipvitL14",
        help="Representation spaces to precompute",
        choices=[
            "clipRN50",
            "clipRN101",
            "clipRN50x4",
            "clipRN50x16",
            "clipRN50x64",
            "clipvitB32",
            "clipvitB16",
            "clipvitL14",
            "dinov2",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--root-dir", type=str, help="Root dir to store everything")
    parser.add_argument("--device-num", type=int, default=0, help="-1:cpu, >=0:cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args)


def get_features(dataloader, model, device):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            features = model(x.to(device))
            all_features.append(features.detach().cpu())
            all_labels.append(y.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)
    root_dir = Path(args.root_dir)
    model_name = args.model

    if args.device_num == -1:
        device = torch.device("cpu")
    else:
        assert args.device_num < torch.cuda.device_count()
        device = torch.device(f"cuda:{args.device_num}")

    model, preprocess = load_model(
        model_name,
        device=device,
        root_dir=args.root_dir,
    )
    trainloader, valloader = get_dataloaders(
        args.dataset,
        preprocess,
        args.batch_size,
        args.root_dir,
    )
    feats_train, y_train = get_features(trainloader, model, device)
    feats_val, y_val = get_features(valloader, model, device)

    model_dir = root_dir.joinpath(f"representations/{args.model}")
    model_dir.mkdir(exist_ok=True)

    np.save(model_dir.joinpath(f"{args.dataset}_feats_train.npy"), feats_train)
    np.save(model_dir.joinpath(f"{args.dataset}_y_train.npy"), y_train)

    np.save(model_dir.joinpath(f"{args.dataset}_feats_val.npy"), feats_val)
    np.save(model_dir.joinpath(f"{args.dataset}_y_val.npy"), y_val)


if __name__ == "__main__":
    run()
