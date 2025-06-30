import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm


from UNet import rL2_loss, inv_minmax
from FNO_utils import FNO2d
from pc_encoders import *

# Disable TF32 for reproducibility
torch.backends.cudnn.allow_tf32 = False

tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0"),
}

script_dir = os.path.dirname(os.path.abspath(__file__))


def mapback(x):
    return x


def edge_weighted_loss(target, pred):
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=target.device
    ).float()
    sobel_y = sobel_x.t()

    grad_x = F.conv2d(target, sobel_x.view(1, 1, 3, 3), padding=1)
    grad_y = F.conv2d(target, sobel_y.view(1, 1, 3, 3), padding=1)
    edge_map = torch.sqrt(grad_x**2 + grad_y**2)

    weights = 1 + 2.0 * (edge_map / (torch.max(edge_map) + 1e-8))
    error = pred - target
    weighted_error = weights * error

    return torch.linalg.norm(weighted_error) / torch.clamp(torch.linalg.norm(target), min=1e-6)


def combined_loss(img1, img2, alpha=0.8, beta=1.0):
    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)

    mse_loss = edge_weighted_loss(img1, img2)
    return mse_loss


def do_plot(x, u, fig, ax, position, xlabel, ylabel, title_top,
            vmin=None, vmax=None, cmap='jet'):
    ax = ax[position[0]]
    h = ax.scatter(
        x[..., 0], x[..., 1],
        c=u, marker='.', cmap=cmap, s=5,
        vmin=vmin, vmax=vmax
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.offsetText.set(size=10)
    cbar.ax.yaxis.offsetText.set_x(2)

    tick_locator = ticker.MaxNLocator(nbins=8)
    cbar.locator = tick_locator
    cbar.update_ticks()

    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel(ylabel, labelpad=-0.5)
    cbar.formatter.set_powerlimits((0, 0))
    ax.set_aspect('equal', 'box')
    ax.set_title(title_top, fontsize=15, pad=5)


class FNO_naca(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        modes = 12
        width = 32
        self.compiled_model = FNO2d(modes, modes, width, 3).to("cuda:0")

    def fit(self, n_epochs=1000, lr=5e-4, title="default"):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(),
            lr=lr,
            weight_decay=2e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=40,
            verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epochs_iter = tqdm(
            range(n_epochs),
            desc='Epoch',
            position=0,
            leave=True
        )

        for epoch in epochs_iter:
            self.compiled_model.train()
            epoch_loss = 0.0

            for X_batch, A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                train_output = self.compiled_model(X_batch, A_batch)

                loss = combined_loss(U_batch, train_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compiled_model.parameters(), 10
                )
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if (epoch + 1) % 50 == 0 or epoch == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2 = self.evaluate()
                    l2_err_hist.append([rL2_rec, rL2])

            desc = (
                f'Epoch {epoch} - loss {epoch_loss / len(self.train_loader):.3e} '
                f'- rl2_rec error {rL2_rec:.3e} , rl2_enc error {rL2:.3e}'
            )
            epochs_iter.set_description(desc)
            epochs_iter.update(1)

        return loss_hist, l2_err_hist

    def evaluate(self):
        total_rL2_rec = 0.0
        total_rL2 = 0.0

        with torch.no_grad():
            for X_test_batch, A_test_batch, U_test_batch, input_test, ref_test in self.test_loader:
                test_output = self.compiled_model(X_test_batch, A_test_batch)

                grid_u_test = test_output.squeeze(-1).unsqueeze(1)
                grid_o_test = A_test_batch[..., 0].unsqueeze(1)

                grid_u_test = mapback(grid_u_test)
                grid_o_test = mapback(grid_o_test)

                rec_test = reconstruct_reference_values(
                    grid_u_test / (grid_o_test + 1e-8),
                    input_test,
                    step='bilinear'
                )
                rL2 = torch.linalg.norm(rec_test - ref_test) / torch.linalg.norm(ref_test)
                total_rL2_rec += rL2.item()

                rL2 = torch.linalg.norm(test_output - U_test_batch) / torch.linalg.norm(U_test_batch)
                total_rL2 += rL2.item()

        avg_rL2_rec = total_rL2_rec / len(self.test_loader)
        avg_rL2 = total_rL2 / len(self.test_loader)


        return avg_rL2_rec, avg_rL2



class FNO_elas(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        modes = 12
        width = 32
        self.compiled_model = FNO2d(modes, modes, width, 4).to("cuda:0")

    def fit(self, n_epochs=1000, lr=5e-4, title="default"):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(), lr=lr, weight_decay=2e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=40, verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epochs_iter = tqdm(range(n_epochs), desc='Epoch', position=0, leave=True)

        for epoch in epochs_iter:
            self.compiled_model.train()
            epoch_loss = 0.0

            for X_batch, A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                train_output = self.compiled_model(X_batch, A_batch)

                loss = combined_loss(U_batch, train_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compiled_model.parameters(), 10
                )
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if epoch % 50 == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2 = self.evaluate()
                    l2_err_hist.append([rL2_rec, rL2])

            desc = (
                f'Epoch {epoch} - loss {epoch_loss/len(self.train_loader):.3e}'
                f' - rl2_rec {rL2_rec:.3e}, rl2_enc {rL2:.3e}'
            )
            epochs_iter.set_description(desc)
            epochs_iter.update(1)

        return loss_hist, l2_err_hist

    def evaluate(self):
        total_rL2_rec = 0.0
        total_rL2 = 0.0

        with torch.no_grad():
            for X_test, A_test, U_test, input_test, ref_test in self.test_loader:
                test_output = self.compiled_model(X_test, A_test)
                grid_u = test_output.squeeze(-1).unsqueeze(1)
                grid_o = A_test[..., 0].unsqueeze(1)

                grid_u = mapback(grid_u)
                grid_o = mapback(grid_o)

                rec_test = reconstruct_reference_values(
                    grid_u / (grid_o + 1e-8), input_test, step='bilinear'
                ).detach()
                rL2_rec = torch.linalg.norm(rec_test - ref_test) / torch.linalg.norm(ref_test)
                total_rL2_rec += rL2_rec.item()

                rL2_enc = torch.linalg.norm(test_output - U_test) / torch.linalg.norm(U_test)
                total_rL2 += rL2_enc.item()

        avg_rL2_rec = total_rL2_rec / len(self.test_loader)
        avg_rL2_enc = total_rL2 / len(self.test_loader)

        return avg_rL2_rec, avg_rL2_enc
    


class FNO_darcy(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        modes = 12
        width = 32
        self.compiled_model = FNO2d(modes, modes, width, 4).to(tkwargs['device'])

    def fit(self, n_epochs=1000, lr=5e-4, title="default"):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(),
            lr=lr,
            weight_decay=2e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=40,
            verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epochs = tqdm(
            range(n_epochs),
            desc='Epoch',
            position=0,
            leave=True
        )

        for epoch in epochs:
            self.compiled_model.train()
            epoch_loss = 0.0

            for A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                train_output = self.compiled_model(A_batch, A_batch)

                loss = combined_loss(U_batch, train_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.compiled_model.parameters(), 10)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if epoch % 50 == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2 = self.evaluate()
                    l2_err_hist.append([rL2_rec, rL2])

            desc = (
                f'Epoch {epoch} - loss {epoch_loss/len(self.train_loader):.3e}'
                f' - rl2_rec {rL2_rec:.3e}, rl2_enc {rL2:.3e}'
            )
            epochs.set_description(desc)

        return loss_hist, l2_err_hist

    def evaluate(self):
        num_u = den_u = num_rec_u = den_rec_u = 0.0

        with torch.no_grad():
            for A_test, U_test, input_test, ref_test in self.test_loader:
                ref_test = ref_test.to(tkwargs['device'])
                input_test = input_test.to(tkwargs['device'])

                test_output = self.compiled_model(A_test, A_test)
                grid_u = test_output.squeeze(-1).unsqueeze(1)
                grid_o = A_test[..., 0].unsqueeze(1)

                rec_test = reconstruct_reference_values(
                    grid_u / (grid_o + 1e-8),
                    input_test,
                    step='bilinear'
                ).detach()

                num_rec_u += torch.sum((rec_test - ref_test) ** 2)
                den_rec_u += torch.sum(ref_test ** 2)

                num_u += torch.sum((test_output - U_test) ** 2)
                den_u += torch.sum(U_test ** 2)

        avg_rL2_rec = torch.sqrt(num_rec_u / den_rec_u).item()
        avg_rL2 = torch.sqrt(num_u / den_u).item()

        return avg_rL2_rec, avg_rL2
    

class FNO_circles(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        modes = 12
        width = 32

        self.compiled_model = FNO2d(modes, modes, width, 3, 3).to(tkwargs['device'])

    def fit(self, n_epochs=1000, lr=5e-4, title="default", pca_flag=True):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(), lr=lr, weight_decay=2e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=40, verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epochs_iter = tqdm(range(n_epochs), desc='Epoch', position=0, leave=True)

        for epoch in epochs_iter:
            self.compiled_model.train()
            epoch_loss = 0.0
            w_v, w_p = 1.0, 1.0

            for A_batch, U_batch, Y_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    Y_batch = Y_batch.unsqueeze(1).unsqueeze(1)
                    input_features = A_batch
                    train_output = self.compiled_model(input_features, input_features)

                    loss_u = combined_loss(U_batch[..., 0:1], train_output[..., 0:1])
                    loss_v = combined_loss(U_batch[..., 1:2], train_output[..., 1:2])
                    loss_p = combined_loss(U_batch[..., 2:3], train_output[..., 2:3])

                    loss = loss_u + w_v * loss_v + w_p * loss_p


                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compiled_model.parameters(), max_norm=10.0
                )
                optimizer.step()

                epoch_loss += loss.item()


            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if (epoch + 1) % 50 == 0 or epoch == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    metrics = self.evaluate()
                    l2_err_hist.append(list(metrics))

            desc = (
                f'Epoch {epoch} - loss {epoch_loss/len(self.train_loader):.3e} - '
                f'rl2_rec_U {metrics[5]:.3e}, rl2_rec_p {metrics[4]:.3e}'
            )
            epochs_iter.set_description(desc)
            epochs_iter.update(1)

        return loss_hist, l2_err_hist

    def evaluate(self):
        num_u = den_u = num_v = den_v = num_p = den_p = 0.0
        num_rec_u = den_rec_u = num_rec_v = den_rec_v = num_rec_p = den_rec_p = num_rec_U = den_rec_U = 0.0

        with torch.no_grad():
            for A_test, U_test, Y_test, input_test, ref_test, c_test in self.test_loader:
                ref_test = ref_test.to(tkwargs['device'])
                input_test = input_test.to(tkwargs['device'])
                Y_test = Y_test.unsqueeze(1).unsqueeze(1)

                input_features = A_test
                test_output = self.compiled_model(input_features, input_features)

                grid_u = test_output.unsqueeze(1)
                grid_o = A_test[..., 0].unsqueeze(1)

                recs = []
                for i in range(test_output.shape[-1]):
                    rec = reconstruct_reference_values(
                        grid_u[..., i] / (grid_o + 1e-8),
                        input_test[0].to(torch.float32),
                        step='bilinear'
                    )
                    recs.append(rec)
                rec_test = torch.cat(recs, -1).detach()

                rec_test[..., 0] = inv_minmax(rec_test[..., 0], -0.05, 3.21)
                rec_test[..., 1] = inv_minmax(rec_test[..., 1], -1.90, 1.62)
                rec_test[..., 2] = inv_minmax(rec_test[..., 2], -4.93, 73.33)

                ref_perm = ref_test.permute(0, 2, 1, 3)
                U_test_mag = torch.sqrt(rec_test[..., 0]**2 + rec_test[..., 1]**2)
                U_ref_mag = torch.sqrt(ref_perm[..., 0]**2 + ref_perm[..., 1]**2)

                num_rec_U += torch.sum((U_test_mag - U_ref_mag)**2)
                den_rec_U += torch.sum(U_ref_mag**2)

                num_rec_u += torch.sum((rec_test[..., 0] - ref_perm[..., 0])**2)
                den_rec_u += torch.sum(ref_perm[..., 0]**2)
                num_rec_v += torch.sum((rec_test[..., 1] - ref_perm[..., 1])**2)
                den_rec_v += torch.sum(ref_perm[..., 1]**2)
                num_rec_p += torch.sum((rec_test[..., 2] - ref_perm[..., 2])**2)
                den_rec_p += torch.sum(ref_perm[..., 2]**2)

                num_u += torch.sum((test_output[..., 0] - U_test[..., 0])**2)
                den_u += torch.sum(U_test[..., 0]**2)
                num_v += torch.sum((test_output[..., 1] - U_test[..., 1])**2)
                den_v += torch.sum(U_test[..., 1]**2)
                num_p += torch.sum((test_output[..., 2] - U_test[..., 2])**2)
                den_p += torch.sum(U_test[..., 2]**2)

        metrics = (
            torch.sqrt(num_rec_u / den_rec_u).item(),
            torch.sqrt(num_u / den_u).item(),
            torch.sqrt(num_rec_v / den_rec_v).item(),
            torch.sqrt(num_v / den_v).item(),
            torch.sqrt(num_rec_p / den_rec_p).item(),
            torch.sqrt(num_p / den_p).item(),
            torch.sqrt(num_rec_U / den_rec_U).item(),
        )

        return metrics

class FNO_maze(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        modes = 12
        width = 32
        self.compiled_model = FNO2d(modes, modes, width, 4).to(tkwargs['device'])

    def fit(self, n_epochs=1000, lr=5e-4, title="default"):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(), lr=lr/10, weight_decay=2e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=40, verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epochs_iter = tqdm(
            range(n_epochs), desc='Epoch', position=0, leave=True
        )

        for epoch in epochs_iter:
            self.compiled_model.train()
            epoch_loss  = 0.0

            for A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                train_output = self.compiled_model(A_batch, A_batch)

                loss = combined_loss(U_batch, train_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compiled_model.parameters(), 10
                )
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if (epoch + 1) % 50 == 0 or epoch == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2 = self.evaluate()
                    l2_err_hist.append([rL2_rec, rL2])

            desc = (
                f'Epoch {epoch} - loss {epoch_loss/len(self.train_loader):.3e}'
                f' - rl2_rec error {rL2_rec:.3e}, rl2_enc error {rL2:.3e}'
            )
            epochs_iter.set_description(desc)
            epochs_iter.update(1)

        return loss_hist, l2_err_hist

    def evaluate(self):
        num_u = den_u = num_rec_u = den_rec_u = 0.0

        with torch.no_grad():
            for A_test, U_test, input_test, ref_test in self.test_loader:
                ref_test = ref_test.to(tkwargs['device'])
                input_test = input_test.to(tkwargs['device'])

                test_output = self.compiled_model(A_test, A_test)
                grid_u = test_output.squeeze(-1).unsqueeze(1)
                grid_o = A_test[..., 0].unsqueeze(1)

                rec_test = reconstruct_reference_values(
                    grid_u / (grid_o + 1e-8), input_test[0], step='bilinear'
                ).detach()
                rec_test = inv_minmax(rec_test, -4.0, 4.0)

                num_rec_u += torch.sum(
                    (rec_test.permute(0, 2, 1, 3) - ref_test) ** 2
                )
                den_rec_u += torch.sum(ref_test**2)

                num_u += torch.sum((test_output - U_test) ** 2)
                den_u += torch.sum(U_test**2)

        avg_rL2_rec = torch.sqrt(num_rec_u / den_rec_u).item()
        avg_rL2 = torch.sqrt(num_u / den_u).item()


        return avg_rL2_rec, avg_rL2
        

