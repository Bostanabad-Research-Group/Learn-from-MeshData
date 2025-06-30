import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from FNO import (
    FNO_naca,
    FNO_elas,
    FNO_darcy,
    FNO_circles,
    FNO_maze
)
from UNet import (
    UNet_naca,
    UNet_elas,
    UNet_darcy,
    UNet_circles,
    UNet_maze,
    UNet_solid3d,
)
from data_utils import (
    get_data_naca,
    get_data_elas,
    get_data_darcy,
    get_data_circles,
    get_data_maze,
    get_data_solid2,
)

# -----------------------------------------------------------------------------
# Globals & Helpers
# -----------------------------------------------------------------------------

LR = 1e-3  # learning rate

tkwargs = {
    "dtype": torch.float32,
    "device": torch.device("cuda:0"),
}


def to_dev(x):
    """Convert a numpy torch.Tensor to the default dtype & device."""
    return x.type(tkwargs["dtype"]).to(tkwargs["device"])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Dataset Classes
# -----------------------------------------------------------------------------
class MyCustomDataset(Dataset):
    """Generic dataset holding three tensors + three auxiliary lists."""

    def __init__(self, t1, t2, t3, l1, l2, l3):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def __len__(self):
        return len(self.t1)

    def __getitem__(self, idx):
        return (
            self.t1[idx],
            self.t2[idx],
            self.t3[idx],
            torch.tensor(self.l1[idx], dtype=torch.float32),
            torch.tensor(self.l2[idx], dtype=torch.float32),
            torch.tensor(self.l3[idx], dtype=torch.float32),
        )


class MyCustomDatasetMaze(Dataset):
    """Simplified dataset: two tensors + two lists."""

    def __init__(self, t1, t2, l1, l2):
        self.t1 = t1
        self.t2 = t2
        self.l1 = l1
        self.l2 = l2

    def __len__(self):
        return len(self.t1)

    def __getitem__(self, idx):
        return (
            self.t1[idx],
            self.t2[idx],
            torch.tensor(self.l1[idx], dtype=torch.float32),
            torch.tensor(self.l2[idx], dtype=torch.float32),
        )


# -----------------------------------------------------------------------------
# Training / Evaluation Pipeline
# -----------------------------------------------------------------------------
def main(
    problem: str = "naca",
    epochs: int = 1000,
    model_type: str = "eunet",
    ntrain: int = 1000,
    res: int = 128,
    batch_size: int = 10,
    seed: int = 2025,
):
    set_seed(seed)

    # build experiment title
    d = datetime.now().strftime("%B%d_%H-%M")
    title = (
        f"{problem}_{model_type}_{d}"
        f"_ntrain{ntrain}_res{res}_epochs{epochs}_b{batch_size}"
        f"_seed{seed}_lr{LR}"
    )

    # -------------------------------------------------------------------------
    #   LOAD & PREPARE DATA
    # -------------------------------------------------------------------------
    if problem == "naca":

        data = get_data_naca(ntrain, res)
        (
            X_train,
            A_train,
            U_train,
            X_test,
            A_test,
            U_test,
            inp,
            ref,
        ) = data

        # send to device
        X_train, A_train, U_train = map(to_dev, (X_train, A_train, U_train))
        X_test, A_test, U_test = map(to_dev, (X_test, A_test, U_test))
        inp, ref = to_dev(inp), to_dev(ref)

        # choose model & loaders
        if model_type == "efno":
            train_ds = TensorDataset(X_train, A_train, U_train)
            test_ds = TensorDataset( X_test, A_test, U_test, inp, ref)
            model = FNO_naca

        else:  # UNet
            train_ds = TensorDataset(X_train, A_train, U_train)
            test_ds = TensorDataset(X_test, A_test, U_test, inp, ref)
            model = UNet_naca

    elif problem == "elas":
        (
            X_train,
            A_train,
            U_train,
            X_test,
            A_test,
            U_test,
            inp,
            ref,
        ) = get_data_elas(ntrain, res)

        # send to device
        tensors = [X_train, A_train, U_train, X_test, A_test, U_test, inp, ref]
        tensors = [to_dev(t) for t in tensors]
        X_train, A_train, U_train, X_test, A_test, U_test, inp, ref = tensors

        if model_type == "efno":
            train_ds = TensorDataset(
                X_train, A_train, U_train
            )
            test_ds = TensorDataset(
                X_test, A_test, U_test, inp, ref
            )
            model = FNO_elas
        else:
            train_ds = TensorDataset(X_train, A_train, U_train)
            test_ds = TensorDataset(X_test, A_test, U_test, inp, ref)
            model = UNet_elas

    elif problem == "darcy":
        (
            X_train,
            A_train,
            U_train,
            X_test,
            A_test,
            U_test,
            inp,
            ref,
        ) = get_data_darcy(ntrain, res)
        X_train, A_train, U_train, X_test, A_test, U_test, inp, ref = map(
            to_dev, (X_train, A_train, U_train, X_test, A_test, U_test, inp, ref)
        )

        if model_type == "efno":
            train_ds = TensorDataset(A_train, U_train)
            test_ds = TensorDataset(A_test, U_test, inp, ref)
            model = FNO_darcy
        else:
            train_ds = TensorDataset(X_train, A_train, U_train)
            test_ds = TensorDataset(X_test, A_test, U_test, inp, ref)
            model = UNet_darcy

    elif problem == "circles":
        A_train, U_train, A_test, U_test, inp, ref, _c = get_data_circles(
            ntrain, res
        )
        A_train, U_train, A_test, U_test = map(
            to_dev, (A_train, U_train, A_test, U_test)
        )

        train_ds = TensorDataset(A_train, U_train, A_train)
        test_ds = MyCustomDataset(A_test, U_test, A_test, inp, ref, _c)

        if model_type == "efno":
            model = FNO_circles
        else:
            model = UNet_circles

    elif problem == "maze":
        A_train, U_train, A_test, U_test, inp, ref = get_data_maze(
            ntrain, res
        )
        A_train, U_train, A_test, U_test = map(
            to_dev, (A_train, U_train, A_test, U_test)
        )

        train_ds = TensorDataset(A_train, U_train)
        test_ds = MyCustomDatasetMaze(A_test, U_test, inp, ref)

        if model_type == "efno":
            model = FNO_maze  # <-- or FNO_maze if available
        else:
            model = UNet_maze

    elif problem == "solid3d":
        (
            A_train,
            U_train,
            A_test,
            U_test,
            inp,
            ref,
            max_train,
            max_test,
            inp_tr,
            ref_tr,
        ) = get_data_solid2(ntrain, res)

        A_train, U_train, A_test, U_test, inp, ref, max_train, max_test, inp_tr, ref_tr = map(
            to_dev,
            (
                A_train,
                U_train,
                A_test,
                U_test,
                inp,
                ref,
                max_train,
                max_test,
                inp_tr,
                ref_tr,
            ),
        )

        train_ds = TensorDataset(A_train, U_train, max_train, inp_tr, ref_tr)
        test_ds = TensorDataset(A_test, U_test, inp, ref, max_test)
        model = UNet_solid3d

    else:
        raise ValueError(f"Unknown problem: {problem}")

    # create loaders
    if problem in ["elas" , "naca" , "darcy"]:
        test_batch = 100
    elif problem == "solid3d":
        test_batch = 5
    else: 
        test_batch = 1
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False)

    net = model(train_loader, test_loader)

    total_params = sum(p.numel() for p in net.compiled_model.parameters())
    print(f"Total parameters: {total_params:,}")

    start = time.time()
    loss_hist, l2_hist = net.fit(n_epochs=epochs, lr=LR, title=title)
    duration = time.time() - start

    # save history
    hist = pd.DataFrame({
        "loss": [l[0] for l in loss_hist],
    })
    hist.to_csv(f"./Results/{problem}/history_{title}.csv", index=False)

    # save error history (based ond the reconstructed and encoded values)
    rl2 = pd.DataFrame(np.array(l2_hist))
    rl2.to_csv(f"./Results/{problem}/rL2_{title}.csv", index=False)

    # save timing & model
    np.savetxt(f"./Results/{problem}/ET_params_{title}.txt", np.array([duration, total_params]))
    torch.save(net.compiled_model, f"./checkpoints/{title}_model.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train FNO/UNet models")
    parser.add_argument("--problem",    default="naca") #naca , elas , darcy , circles, maze , ("solid3d" only with eunet)
    parser.add_argument("--model",      default="eunet") #efno , eunet
    parser.add_argument("--ntrain",     type=int, default=1000)
    parser.add_argument("--res",          type=int, default=128)
    parser.add_argument("--epochs",     type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10) 
    parser.add_argument("--seed",       type=int,   default=2025)
    args = parser.parse_args()

    main(
        problem=args.problem,
        epochs=args.epochs,
        model_type=args.model,
        ntrain=args.ntrain,
        res=args.res,
        batch_size=args.batch_size,
        seed=args.seed,
    )
