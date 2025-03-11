from omegaconf import DictConfig, OmegaConf
from sklearn.neighbors import kneighbors_graph
from torch import optim
from tqdm import tqdm
import hydra
import numpy as np
import torch

from prcut.models.prcut import get_prcut_encoder
from prcut.metrics import evaluate_clustering
from prcut.losses.prcut import PRCutGradLoss
from prcut.losses.prcut import PRCutBatchLoss


@hydra.main(version_base="1.2", config_path="./configs", config_name="prcut")
def main(cfg: DictConfig):
    repr = cfg.repr
    Z_train = np.load(
        f"{cfg.root_dir}/representations/{repr}/{cfg.dataset}_feats_train.npy"
    ).astype(np.float32)
    y_train = np.load(
        f"{cfg.root_dir}/representations/{repr}/{cfg.dataset}_y_train.npy"
    )
    num_samples = Z_train.shape[0]
    print("Computing K-neighbors")
    W = kneighbors_graph(Z_train, n_neighbors=cfg.k)
    print("Computing K-neighbors: Done!")
    b_size = cfg.batch_size
    beta = cfg.beta
    lamb = 2.0 * b_size
    loss_beta = 0.9
    loss_estimate = None

    model = get_prcut_encoder(
        Z_train.shape[1], cfg.num_classes, cfg.num_layers, cfg.num_hidden_units
    ).cuda()
    print(model)
    op = optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # for the backprop target
    prcut_grad_fn = PRCutGradLoss()
    # for loss estimation without backprop graph
    prcut_loss_fn = PRCutBatchLoss(
        num_clusters=cfg.num_classes, gamma=cfg.gammma
    ).cuda()
    bar = tqdm(range(cfg.num_iter))
    eval_interval = cfg.eval_interval
    for _ in bar:
        total_loss = 0.0
        model.train()
        for _ in range(eval_interval):
            b_indices = np.random.choice(num_samples, 2 * b_size, replace=False)
            l_indices = b_indices[:b_size]
            r_indices = b_indices[b_size:]
            P = model(torch.tensor(Z_train[b_indices]).cuda())
            P_l = P[:b_size]
            P_r = P[b_size:]
            prcut_loss_fn.update_cluster_p(P)
            ov_P = prcut_loss_fn.clusters_p.data

            b_W = torch.tensor(
                W[l_indices][:, r_indices].toarray().astype(np.float32)
            ).cuda()

            sp_loss = prcut_grad_fn(b_W, P_l, P_r, ov_P, lamb) / b_W.sum()
            loss = 0.0
            loss += sp_loss
            loss += -beta * torch.special.entr(P.mean(0)).sum()
            with torch.no_grad():
                total_loss += prcut_loss_fn(b_W, P_l, P_r)

            op.zero_grad()
            loss.backward()
            op.step()

        if loss_estimate is None:
            loss_estimate = total_loss
        loss_estimate = loss_beta * loss_estimate + (1 - loss_beta) * total_loss
        with torch.no_grad():
            probas = model(torch.tensor(Z_train).cuda())
        results = evaluate_clustering(
            y_train,
            torch.argmax(probas, dim=1).detach().cpu().numpy(),
            num_classes=cfg.num_classes,
        )
        results.pop("confusion_matrix")
        results.update({"loss": total_loss})
        bar.set_description(
            f"ac:{results['accuracy']:.3f},nmi:{results['nmi']:.3f},ls:{loss_estimate:2.2f}"
        )
    results = evaluate_clustering(
        y_train,
        torch.argmax(probas, dim=1).detach().cpu().numpy(),
        num_classes=cfg.num_classes,
    )


if __name__ == "__main__":
    main()
