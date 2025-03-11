from typing import Tuple
import torch
from torch import nn


class TurtlePredictionLoss(nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, y_task, y_task_per_space, y_hat, y_hat_per_space):
        """
        y_task : (b,k)
        y_task_spaces: (b,s,k)
        y_hat: (b,s,k)
        y_hat_spaces: (b,s,k)

        """
        b, s, k = y_task_per_space.size(1)

        loss = F.cross_entropy(y_hat, y_task.detach(), reduction="sum") / s
        entr_reg = torch.special.entr(y_task_per_space.mean(0)).sum()

        pred_error = sum(
            [
                F.cross_entropy(w_in(z_tr).detach(), labels)
                for w_in, z_tr in zip(W_in, Zs_tr)
            ]
        )


class TurtleTaskLoss(nn.Module):

    def __init__(self, entropy_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_task, y_task_spaces, y_hat, y_hat_spaces):
        """
        y_task : (b,k)
        y_task_spaces: (b,s,k)
        y_hat: (b,s,k)
        y_hat_spaces: (b,s,k)

        """
        b, s, k = y_task_spaces.size(1)

        loss = F.cross_entropy(y_hat, y_task.detach(), reduction="sum") / s
        entr_reg = torch.special.entr(y_task_spaces.mean(0)).sum()

        pred_error = sum(
            [
                F.cross_entropy(w_in(z_tr).detach(), labels)
                for w_in, z_tr in zip(W_in, Zs_tr)
            ]
        )

        # entropy regularization

        # final loss, Eq. (12) in the paper
        (pred_error - args.gamma * entr_reg).backward()
        optimizer.step()

        sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space])
        sum(
            [
                F.cross_entropy(w_in(z_tr), labels.detach())
                for w_in, z_tr in zip(W_in, Zs_tr)
            ]
        )
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
            loss.backward()
            inner_opt.step()

        # update task encoder
        optimizer.zero_grad()
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
