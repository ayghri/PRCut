import argparse
import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from prcut.models.layers import DecoupledLinear

# from utils import seed_everything, get_cluster_acc, datasets_to_c
from prcut.configs.trainer import TurtleConfig
from prcut.losses.contrastive import SimCLRLoss
from prcut import constructors

from .base import BaseTrainer
from .augmentations import Transform


def _parse_args(args):
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument(
        "--dataset", type=str, help="Dataset to run TURTLE", required=True
    )
    parser.add_argument(
        "--phis",
        type=str,
        default=["clipvitL14", "dinov2"],
        nargs="+",
        help="Representation spaces to run TURTLE",
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
            "euclid",
        ],
    )
    # training
    parser.add_argument(
        "--gamma",
        type=float,
        default=10.0,
        help="Hyperparameter for entropy regularization in Eq. (12)",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=6000,
        help="Number of outer iterations to train task encoder",
    )
    parser.add_argument(
        "--inner_lr", type=float, default=0.001, help="Learning rate for inner loop"
    )
    parser.add_argument(
        "--outer_lr", type=float, default=0.001, help="Learning rate for task encoder"
    )
    # parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="warm start = initialize inner learner from previous iteration, cold start = initialize randomly, cold-start is used by default",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10,
        help="Number of inner steps at each outer iteration",
    )
    # others
    parser.add_argument(
        "--cross_val",
        action="store_true",
        help="Whether to perform cross-validation to compute generalization score after training",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--root_dir", type=str, default="data", help="Root dir to store everything"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args)


class Turtle(BaseTrainer):

    def __init__(
        self,
        config: TurtleConfig,
        task_encoder: DecoupledLinear,
        classifier: DecoupledLinear,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)

        # num_repr, repr_dims, num_clusters, *args, **kwargs) -> None:
        # assert len(repr_dims) == num_repr
        # self.task_encoder = DecoupledLinear(repr_dims, num_clusters)
        # self.classifiers = DecoupledLinear(repr_dims, num_clusters)
        self.task_encoder = task_encoder
        self.classifier = classifier
        self.task_optim = constructors.get_optimizer(
            config.outer_optim, task_encoder.parameters()
        )
        self.classifier_optim = constructors.get_optimizer(
            config.inner_optim, classifier.parameters()
        )

    # def train_step(self, x, retain_graph=False):

    def train_step(self, x, retain_graph=False) -> Dict[str, torch.Tensor]:
        y_task_per_space = self.task_encoder(x)
        y_task = y_task_per_space.mean(-1)
        classifier_loss = torch.tensor(0.0)
        for _ in range(self.config.num_inner_iter):
            self.classifier_optim.zero_grad()
            classifier_loss = F.cross_entropy(self.classifier(x), y_task.detach())
            classifier_loss.backward()
            self.classifier_optim.step()

        self.task_optim.zero_grad()
        pred_loss = F.cross_entropy(self.classifier(x).detach(), y_task)
        entropy_reg = torch.special.entr(y_task_per_space.mean(0)).sum()
        task_loss = pred_loss - self.config.gamma * entropy_reg
        task_loss.backward(retain_graph=retain_graph)
        self.task_optim.step()
        return {
            "classifier_loss": classifier_loss,
            "pred_loss": pred_loss,
            "entropy_reg": entropy_reg,
            "task_loss": task_loss,
        }

    # loss_name = "cross_entropy"
    #
    # def __init__(
    #     self,
    #     config: SimCLRConfig,
    #     encoder: nn.Module,
    #     augmenter: Transform,
    #     *args,
    #     **kwargs,
    # ):
    #     self.config = config
    #     self.encoder = encoder
    #     self.networks["encoder"] = encoder
    #     self.augmenter = augmenter
    #     self.device = torch.device(config.device_name)
    #     self.optimizer = constructors.get_optimizer(
    #         config.optimizer,
    #         self.encoder.parameters(),
    #     )
    #     self.scheduler = constructors.get_scheduler(config.scheduler, self.optimizer)
    #     self.criterion = SimCLRLoss(config.temperature)
    #
    # def encode(self, x, **kwargs):
    #     return self.encoder(x)
    #
    # def compute_batch_loss(self, x, **kwargs):
    #     # shape B,D
    #     z_1 = self.encode(self.augmenter(x))
    #     z_2 = self.encode(self.augmenter(x))
    #     return {self.loss_name: self.criterion(z_1, z_2)}
    #


def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)

    # Load pre-computed representations
    Zs_train = [
        np.load(
            f"{args.root_dir}/representations/{phi}/{args.dataset}_feats_train.npy"
        ).astype(np.float32)
        for phi in args.phis
    ]
    Zs_val = [
        np.load(
            f"{args.root_dir}/representations/{phi}/{args.dataset}_feats_val.npy"
        ).astype(np.float32)
        for phi in args.phis
    ]
    # y_gt_val = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")
    y_gt_val = np.load(
        f"{args.root_dir}/representations/{args.phis[0]}/{args.dataset}_y_val.npy"
    )
    print(f"Load dataset {args.dataset}")
    print(
        f"Representations of {args.phis}: "
        + " ".join(str(Z_train.shape) for Z_train in Zs_train)
    )

    n_tr, C = Zs_train[0].shape[0], datasets_to_c[args.dataset]
    feature_dims = [Z_train.shape[1] for Z_train in Zs_train]
    batch_size = min(args.batch_size, n_tr)
    print("Number of training samples:", n_tr)

    # Define task encoder
    task_encoder = [
        nn.utils.weight_norm(nn.Linear(d, C)).to(args.device) for d in feature_dims
    ]

    def task_encoding(Zs):
        assert len(Zs) == len(task_encoder)
        # Generate labeling by the average of $\sigmoid(\theta \phi(x))$, Eq. (9) in the paper
        label_per_space = [
            F.softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)
        ]  # shape of (K, N, C)
        labels = torch.mean(torch.stack(label_per_space), dim=0)  # shape of (N, C)
        return labels, label_per_space

    # we use Adam optimizer for faster convergence, other optimziers such as SGD could also work
    optimizer = torch.optim.Adam(
        sum([list(task_phi.parameters()) for task_phi in task_encoder], []),
        lr=args.outer_lr,
        betas=(0.9, 0.999),
    )

    # Define linear classifiers for the inner loop
    def init_inner():
        W_in = [nn.Linear(d, C).to(args.device) for d in feature_dims]
        inner_opt = torch.optim.Adam(
            sum([list(W.parameters()) for W in W_in], []),
            lr=args.inner_lr,
            betas=(0.9, 0.999),
        )

        return W_in, inner_opt

    W_in, inner_opt = init_inner()

    # start training
    iters_bar = tqdm(range(args.T))
    for i in iters_bar:
        optimizer.zero_grad()
        # load batch of data
        indices = np.random.choice(n_tr, size=batch_size, replace=False)
        Zs_tr = [
            torch.from_numpy(Z_train[indices]).to(args.device) for Z_train in Zs_train
        ]

        labels, label_per_space = task_encoding(Zs_tr)

        # init inner
        if not args.warm_start:
            # cold start, re-init every time
            W_in, inner_opt = init_inner()
        # else, warm start, keep previous

        # inner loop: update linear classifiers
        for idx_inner in range(args.M):
            inner_opt.zero_grad()
            # stop gradient by "labels.detach()" to perform first-order hypergradient approximation, i.e., Eq. (13) in the paper
            loss = sum(
                [
                    F.cross_entropy(w_in(z_tr), labels.detach())
                    for w_in, z_tr in zip(W_in, Zs_tr)
                ]
            )
            loss.backward()
            inner_opt.step()

        # update task encoder
        optimizer.zero_grad()
        pred_error = sum(
            [
                F.cross_entropy(w_in(z_tr).detach(), labels)
                for w_in, z_tr in zip(W_in, Zs_tr)
            ]
        )

        # entropy regularization
        entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space])

        # final loss, Eq. (12) in the paper
        (pred_error - args.gamma * entr_reg).backward()
        optimizer.step()

        # evaluation, compute clustering accuracy on test split
        if (i + 1) % 20 == 0 or (i + 1) == args.T:
            labels_val, _ = task_encoding(
                [torch.from_numpy(Z_val).to(args.device) for Z_val in Zs_val]
            )
            preds_val = labels_val.argmax(dim=1).detach().cpu().numpy()
            cluster_acc, _ = get_cluster_acc(preds_val, y_gt_val)

            iters_bar.set_description(
                f"Training loss {float(pred_error):.3f}, entropy {float(entr_reg):.3f}, found clusters {len(np.unique(preds_val))}/{C}, cluster acc {cluster_acc:.4f}"
            )

    print(f"Training finished! ")
    print(
        f"Training loss {float(pred_error):.3f}, entropy {float(entr_reg):.3f}, Number of found clusters {len(np.unique(preds_val))}/{C}, Cluster Acc {cluster_acc:.4f}"
    )

    # compute generalization score
    generalization_score = "not evaluated"
    if args.cross_val:
        from cross_val import LR_cross_validation

        # generate pseudo labels
        labels, _ = task_encoding(
            [torch.from_numpy(Z_train).to(args.device) for Z_train in Zs_train]
        )
        y_pred = labels.argmax(dim=-1).detach().cpu().numpy()
        del optimizer, W_in, inner_opt, pred_error, _, entr_reg, labels
        torch.cuda.empty_cache()

        # do cross-validation on pseudo-labels
        generalization_score = 0.0
        for Z_train in Zs_train:
            generalization_score += LR_cross_validation(
                Z_train,
                y_pred,
                num_epochs=(
                    1000
                    if args.dataset not in ["imagenet", "pcam", "kinetics700"]
                    else 400
                ),
            )

        generalization_score /= len(Zs_train)

    # save results
    num_spaces = len(args.phis)
    phis = "_".join(args.phis)
    exp_path = (
        f"{args.root_dir}/task_checkpoints/{num_spaces}space/{phis}/{args.dataset}"
    )
    inner_start = "warmstart" if args.warm_start else "coldstart"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    for task_phi in task_encoder:
        nn.utils.remove_weight_norm(task_phi)

    task_path = f"turtle_{phis}_innerlr{args.inner_lr}_outerlr{args.outer_lr}_T{args.T}_M{args.M}_{inner_start}_gamma{args.gamma}_bs{args.batch_size}_seed{args.seed}"
    torch.save(
        {f"phi{i+1}": task_phi.state_dict() for i, task_phi in enumerate(task_encoder)},
        f"{exp_path}/{task_path}.pt",
    )

    if not os.path.exists(f"{args.root_dir}/results/{num_spaces}space/{phis}"):
        os.makedirs(f"{args.root_dir}/results/{num_spaces}space/{phis}")

    with open(
        f"{args.root_dir}/results/{num_spaces}space/{phis}/turtle_{args.dataset}.txt",
        "a",
    ) as f:
        f.writelines(
            f"{phis:20}, Number of found clusters {len(np.unique(preds_val))}, Cluster Acc: {cluster_acc:.4f}, Generalizatoin Score {generalization_score}, {task_path} \n"
        )


if __name__ == "__main__":
    run()
