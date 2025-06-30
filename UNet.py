import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from pc_encoders import *
from data_utils import inv_minmax
from UNet3d_utils import UNet3D
from UNet2d_utils import UNet, UNet_mo
import time

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Default tensor kwargs
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0"),
}

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))


def mapback(x):
    return x  # e.g., torch.exp(x) - 0.05


def edge_weighted_loss(target, pred):
    """Edge-weighted relative L2 loss using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=target.device,
                           dtype=target.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    grad_x = F.conv2d(target, sobel_x, padding=1)
    grad_y = F.conv2d(target, sobel_y, padding=1)
    edge_map = torch.sqrt(grad_x**2 + grad_y**2)

    weights = 1 + 2.0 * (edge_map / (edge_map.max() + 1e-8))
    error = pred - target
    weighted_error = weights * error

    return torch.linalg.norm(weighted_error) / \
           torch.clamp(torch.linalg.norm(target), min=1e-6)


def rL2_loss(target, output):
    """Relative L2 loss."""
    return torch.linalg.norm(target - output) / \
           torch.clamp(torch.linalg.norm(target), min=1e-6)


def combined_loss(img1, img2):
    """
    Combined loss (currently edge-weighted L2).
    Expects img tensors in shape [B, H, W, C].
    """
    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)
    return edge_weighted_loss(img1, img2)


def do_plot(x, u, fig, ax, position, xlabel, ylabel, title_top,
            vmin=None, vmax=None, cmap='jet'):
    ax = ax[position[0]]
    scatter = ax.scatter(
        x[..., 0], x[..., 1], c=u, marker='.', s=3,
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(scatter, cax=cax)

    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.offsetText.set(size=10, x=2)
    cbar.locator = ticker.MaxNLocator(nbins=8)
    cbar.update_ticks()
    cbar.formatter.set_powerlimits((0, 0))

    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel(ylabel, labelpad=-0.5)
    ax.set_aspect('equal', 'box')
    ax.set_title(title_top, fontsize=15, pad=5)


class UNet_naca(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.compiled_model = UNet(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            num_hidden_layers=4,

        ).to(tkwargs['device'])

    def fit(self, n_epochs=1000, lr=5e-4, title="default"):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(),
            lr=lr,
            weight_decay=2e-2
        )
        scaler = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=40, verbose=True
        )

        loss_hist, l2_err_hist = [], []
        epoch_iter = tqdm(range(n_epochs), desc='Epoch', position=0, leave=True)

        for epoch in epoch_iter:
            self.compiled_model.train()
            epoch_loss  = 0.0

            for X_batch, A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)

                inputs = A_batch.permute(0, 3, 1, 2).to(tkwargs['device'])
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.compiled_model(inputs)
                    outputs = outputs.permute(0, 2, 3, 1)
                    loss = combined_loss(U_batch, outputs)
            

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.compiled_model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if epoch % 50 == 0 or epoch == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2_enc = self.evaluate(title=title, epoch=epoch)
                    l2_err_hist.append([rL2_rec, rL2_enc])

            desc = (
                f"Epoch {epoch} - loss {epoch_loss/len(self.train_loader):.3e} - "
                f"rl2_rec {rL2_rec:.3e}, rl2_enc {rL2_enc:.3e}"
            )
            epoch_iter.set_description(desc)

        return loss_hist, l2_err_hist

    def evaluate(self, title, epoch):
        all_rec_preds, all_rec_refs = [], []
        all_enc_preds, all_enc_refs = [], []

        with torch.no_grad():
            for X_batch, A_batch, U_batch, input_test, ref_test in self.test_loader:
                inputs = A_batch.permute(0, 3, 1, 2).to(tkwargs['device'])
                outputs = self.compiled_model(inputs)
                outputs = outputs.permute(0, 2, 3, 1)

                grid_u = mapback(outputs.squeeze(-1).unsqueeze(1))
                grid_o = mapback(A_batch[..., 0].unsqueeze(1))

                rec = reconstruct_reference_values(
                    grid_u / (grid_o + 1e-6),
                    input_test,
                    step='bilinear'
                ).cpu()

                all_rec_preds.append(rec)
                all_rec_refs.append(ref_test.cpu())
                all_enc_preds.append(outputs.cpu())
                all_enc_refs.append(U_batch.cpu())

            all_rec_preds = torch.cat(all_rec_preds, dim=0)
            all_rec_refs = torch.cat(all_rec_refs, dim=0)
            all_enc_preds = torch.cat(all_enc_preds, dim=0)
            all_enc_refs = torch.cat(all_enc_refs, dim=0)

            rL2_rec = torch.linalg.norm(all_rec_preds - all_rec_refs)
            rL2_rec /= torch.linalg.norm(all_rec_refs)
            rL2_enc = torch.linalg.norm(all_enc_preds - all_enc_refs)
            rL2_enc /= torch.linalg.norm(all_enc_refs)

        return rL2_rec.item(), rL2_enc.item()

          
        
class UNet_elas(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize UNet model
        self.compiled_model = UNet(
            in_channels=2,    # 2 input channels
            out_channels=1,   # 1 output channel
            hidden_channels=16,
            num_hidden_layers=5,
        ).to("cuda:0")


    def fit(self, n_epochs=1000, lr=5e-4, title="default"):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(),
            lr=lr,
            weight_decay=2e-2
        )
        scaler = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=40,
            verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", position=0, leave=True)

        for epoch in epoch_iter:
            self.compiled_model.train()
            epoch_loss = 0.0

            for X_batch, A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                input_features = A_batch

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    input_tensor = input_features.permute(0, 3, 1, 2)
                    train_output = self.compiled_model(input_tensor)
                    train_output = train_output.permute(0, 2, 3, 1)
                    loss = combined_loss(U_batch, train_output)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.compiled_model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if epoch == 0 or (epoch + 1) % 50 == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2_enc = self.evaluate()
                    l2_err_hist.append([rL2_rec, rL2_enc])

            desc = (
                f"Epoch {epoch} - "
                f"loss {epoch_loss / len(self.train_loader):.3e} - "
                f"rl2_rec {rL2_rec:.3e}, rl2_enc {rL2_enc:.3e}"
            )
            epoch_iter.set_description(desc)

        return loss_hist, l2_err_hist

    def evaluate(self):
        all_rec_preds = []
        all_rec_refs = []
        all_enc_preds = []
        all_enc_refs = []

        with torch.no_grad():
            for X_test, A_test, U_test, input_test, ref_test in self.test_loader:
                input_tensor = A_test.permute(0, 3, 1, 2)
                test_output = self.compiled_model(input_tensor)
                test_output = test_output.permute(0, 2, 3, 1)

                grid_u = mapback(test_output.squeeze(-1).unsqueeze(1))
                grid_o = mapback(A_test[..., 0].unsqueeze(1))

                rec = reconstruct_reference_values(
                    grid_u / (grid_o + 1e-8),
                    input_test,
                    step="bilinear"
                ).detach()

                all_rec_preds.append(rec)
                all_rec_refs.append(ref_test)
                all_enc_preds.append(test_output)
                all_enc_refs.append(U_test)

            all_rec_preds = torch.cat(all_rec_preds, dim=0)
            all_rec_refs = torch.cat(all_rec_refs, dim=0)
            all_enc_preds = torch.cat(all_enc_preds, dim=0)
            all_enc_refs = torch.cat(all_enc_refs, dim=0)

            rL2_rec = torch.linalg.norm(all_rec_preds - all_rec_refs) \
                      / torch.linalg.norm(all_rec_refs)
            rL2_enc = torch.linalg.norm(all_enc_preds - all_enc_refs) \
                      / torch.linalg.norm(all_enc_refs)

        return rL2_rec.item(), rL2_enc.item()
            

class UNet_darcy(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize UNet model
        self.compiled_model = UNet(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            num_hidden_layers=5,
        ).to("cuda:0")

    def fit(self, n_epochs=1000, lr=5e-4, title="default"):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(),
            lr=lr,
            weight_decay=2e-2
        )
        scaler = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=40,
            verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epoch_iter = tqdm(range(n_epochs), desc='Epoch', position=0, leave=True)

        for epoch in epoch_iter:
            self.compiled_model.train()
            epoch_loss = 0.0

            for X_batch, A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                input_features = A_batch

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    x = input_features.permute(0, 3, 1, 2)
                    out = self.compiled_model(x)
                    out = out.permute(0, 2, 3, 1)
                    loss = combined_loss(U_batch, out)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.compiled_model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
            ])

            if epoch == 0 or (epoch + 1) % 50 == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2_enc = self.evaluate()
                    l2_err_hist.append([rL2_rec, rL2_enc])

            desc = (
                f"Epoch {epoch} - "
                f"loss {epoch_loss/len(self.train_loader):.3e} - "
                f"rl2_rec {rL2_rec:.3e}, rl2_enc {rL2_enc:.3e}"
            )
            epoch_iter.set_description(desc)

        return loss_hist, l2_err_hist

    def evaluate(self):
        total_rL2_rec = total_rL2_enc = 0.0

        with torch.no_grad():
            for X_test, A_test, U_test, input_test, ref_test in self.test_loader:
                x = A_test.permute(0, 3, 1, 2)
                out = self.compiled_model(x)
                out = out.permute(0, 2, 3, 1)

                grid_u = mapback(out.squeeze(-1).unsqueeze(1))
                grid_o = mapback(A_test[..., 0].unsqueeze(1))

                rec = reconstruct_reference_values(
                    grid_u / (grid_o + 1e-6),
                    input_test,
                    step='bilinear'
                ).detach()

                rL2_rec = torch.linalg.norm(rec - ref_test) / torch.linalg.norm(ref_test)
                total_rL2_rec += rL2_rec.item()

                rL2_enc = torch.linalg.norm(out - U_test) / torch.linalg.norm(U_test)
                total_rL2_enc += rL2_enc.item()

            avg_rL2_rec = total_rL2_rec / len(self.test_loader)
            avg_rL2_enc = total_rL2_enc / len(self.test_loader)


        return avg_rL2_rec, avg_rL2_enc




class UNet_circles(nn.Module):
    def __init__(self, train_loader, test_loader, s=128):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.res = s #change it based on the resolution of interest

        self.compiled_model = UNet_mo(
            in_channels=1,
            out_channels=3,
            hidden_channels=16,
            num_hidden_layers=5,
            res=self.res
        ).to(tkwargs['device'])

    def fit(self, n_epochs=1000, lr=5e-4, title="default", pca_flag=True):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(),
            lr=lr,
            weight_decay=2e-2
        )
        scaler = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=40
        )

        loss_hist = []
        l2_err_hist = []
        epoch_iter = tqdm(range(n_epochs), desc='Epoch', position=0, leave=True)

        for epoch in epoch_iter:
            self.compiled_model.train()
            epoch_loss = epoch_mse = epoch_ssim = 0.0

            for A_batch, U_batch, Y_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    input_tensor = A_batch.permute(0, 3, 1, 2)
                    train_output = self.compiled_model(input_tensor)
                    train_output = train_output.permute(0, 2, 3, 1)

                    loss_u = combined_loss(
                        U_batch[..., 0:1], train_output[..., 0:1]
                    )
                    loss_v = combined_loss(
                        U_batch[..., 1:2], train_output[..., 1:2]
                    )
                    loss_p = combined_loss(
                        U_batch[..., 2:3], train_output[..., 2:3]
                    )
                    loss = loss_u + loss_v + loss_p

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compiled_model.parameters(), max_norm=10.0
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_mse += loss.item()
                epoch_ssim += loss.item()

            scheduler.step(loss)
            loss_hist.append([
                epoch_loss / len(self.train_loader),
                epoch_mse  / len(self.train_loader),
                epoch_ssim / len(self.train_loader),
            ])

            if epoch == 0 or (epoch + 1) % 50 == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    results = self.evaluate()
                    l2_err_hist.append(results)

            desc = (
                f"Epoch {epoch} - "
                f"loss {epoch_loss/len(self.train_loader):.3e} - "
                f"rl2_rec_p {results[4]:.3e}, rl2_rec_U {results[6]:.3e}"
            )
            epoch_iter.set_description(desc)

        return loss_hist, l2_err_hist

    def evaluate(self):
        num_u = den_u = 0.0
        num_v = den_v = 0.0
        num_p = den_p = 0.0
        num_rec_u = den_rec_u = 0.0
        num_rec_v = den_rec_v = 0.0
        num_rec_p = den_rec_p = 0.0
        num_rec_U = den_rec_U = 0.0

        with torch.no_grad():
            for A_test, U_test, Y_test, input_test, ref_test, c_test in self.test_loader:
                device = tkwargs['device']
                ref_test = ref_test.to(device)
                input_test = input_test.to(device)
                c_test = c_test.to(device) / 4 - 1

                Y_test = Y_test.unsqueeze(1).unsqueeze(1)
                input_tensor = A_test.permute(0, 3, 1, 2)

                test_output = self.compiled_model(input_tensor)
                test_output = test_output.permute(0, 2, 3, 1)

                grid_u = test_output.unsqueeze(1)
                grid_o = A_test[..., 0].unsqueeze(1)

                recs = []
                for i in range(test_output.shape[-1]):
                    rec = reconstruct_reference_values(
                        grid_u[..., i] / (grid_o + 1e-8),
                        input_test[0].float(),
                        step='bilinear'
                    )
                    recs.append(rec)
                rec_test = torch.cat(recs, -1).detach()

                rec_test[..., 0] = inv_minmax(rec_test[..., 0], -0.05, 3.21)
                rec_test[..., 1] = inv_minmax(rec_test[..., 1], -1.90, 1.62)
                rec_test[..., 2] = inv_minmax(rec_test[..., 2], -4.93, 73.33)

                ref_perm = ref_test.permute(0, 2, 1, 3)

                # Reconstruction errors
                num_rec_u += ((rec_test[..., 0] - ref_perm[..., 0])**2).sum()
                den_rec_u += (ref_perm[..., 0]**2).sum()
                num_rec_v += ((rec_test[..., 1] - ref_perm[..., 1])**2).sum()
                den_rec_v += (ref_perm[..., 1]**2).sum()
                num_rec_p += ((rec_test[..., 2] - ref_perm[..., 2])**2).sum()
                den_rec_p += (ref_perm[..., 2]**2).sum()

                # Raw output errors
                num_u += ((test_output[..., 0] - U_test[..., 0])**2).sum()
                den_u += (U_test[..., 0]**2).sum()
                num_v += ((test_output[..., 1] - U_test[..., 1])**2).sum()
                den_v += (U_test[..., 1]**2).sum()
                num_p += ((test_output[..., 2] - U_test[..., 2])**2).sum()
                den_p += (U_test[..., 2]**2).sum()

                # Combined reconstruction magnitude error
                U_rec = torch.sqrt(rec_test[..., 0]**2 + rec_test[..., 1]**2)
                U_ref = torch.sqrt(ref_perm[..., 0]**2 + ref_perm[..., 1]**2)
                num_rec_U += ((U_rec - U_ref)**2).sum()
                den_rec_U += (U_ref**2).sum()

            avg_rec_u = torch.sqrt(num_rec_u / den_rec_u).item()
            avg_rec_v = torch.sqrt(num_rec_v / den_rec_v).item()
            avg_rec_p = torch.sqrt(num_rec_p / den_rec_p).item()
            avg_u = torch.sqrt(num_u / den_u).item()
            avg_v = torch.sqrt(num_v / den_v).item()
            avg_p = torch.sqrt(num_p / den_p).item()
            U_error = torch.sqrt(num_rec_U / den_rec_U).item()

        return (
            avg_rec_u, avg_u,
            avg_rec_v, avg_v,
            avg_rec_p, avg_p,
            U_error
        )

        

class UNet_maze(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.compiled_model = UNet(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            num_hidden_layers=5,
        ).to(tkwargs["device"])

    def fit(self, n_epochs=1000, lr=5e-4, title="default", pca_flag=True):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(),
            lr=lr,
            weight_decay=2e-2
        )
        scaler = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=40,
            verbose=True
        )

        loss_hist = []
        l2_err_hist = []
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", position=0, leave=True)

        for epoch in epoch_iter:
            self.compiled_model.train()
            epoch_loss = 0.0

            for A_batch, U_batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    x = A_batch.permute(0, 3, 1, 2)
                    out = self.compiled_model(x)
                    out = out.permute(0, 2, 3, 1)
                    loss = combined_loss(U_batch, out)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compiled_model.parameters(),
                    max_norm=10.0
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.detach().cpu().item()

            scheduler.step(loss)
            loss_hist.append([epoch_loss / len(self.train_loader)])

            if epoch == 0 or (epoch + 1) % 50 == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2_enc = self.evaluate()
                    l2_err_hist.append([rL2_rec, rL2_enc])

            desc = (
                f"Epoch {epoch} - "
                f"loss {epoch_loss/len(self.train_loader):.3e} - "
                f"rl2_rec error {rL2_rec:.3e}, rl2_enc error {rL2_enc:.3e}"
            )
            epoch_iter.set_description(desc)

        return loss_hist, l2_err_hist

    def evaluate(self):
        num_u = den_u = 0.0
        num_rec_u = den_rec_u = 0.0

        self.compiled_model.eval()
        with torch.no_grad():
            for A_test, U_test, input_test, ref_test in self.test_loader:
                device = tkwargs["device"]
                ref = ref_test.to(device)
                inp = input_test.to(device)

                x = A_test.permute(0, 3, 1, 2)
                out = self.compiled_model(x)
                out = out.permute(0, 2, 3, 1)

                grid_u = out.unsqueeze(1)
                grid_o = A_test[..., 0].unsqueeze(1)

                rec = reconstruct_reference_values(
                    grid_u[..., 0] / (grid_o + 1e-8),
                    inp[0],
                    step="bilinear"
                ).detach()

                rec = inv_minmax(rec, -4.0, 4.0)

                num_rec_u += ((rec.permute(0, 2, 1, 3) - ref) ** 2).sum()
                den_rec_u += (ref ** 2).sum()

                num_u += ((out - U_test) ** 2).sum()
                den_u += (U_test ** 2).sum()

            avg_rL2_rec = torch.sqrt(num_rec_u / den_rec_u).item()
            avg_rL2 = torch.sqrt(num_u / den_u).item()

        return avg_rL2_rec, avg_rL2



class UNet_solid3d(nn.Module):
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize model in float32
        self.compiled_model = UNet3D(
            in_channels=1,
            out_channels=1,
            hidden_channels=12,
            depth=4,
        ).to(tkwargs['device'])

    def get_gpu_temp(self):
        try:
            out = os.popen(
                'nvidia-smi --query-gpu=temperature.gpu '
                '--format=csv,noheader,nounits'
            ).read()
            return int(out.strip())
        except:
            return 0

    def cool_gpu(self, current, thresh, cooldown_temp, cooldown_time, iter_bar):
        while current > cooldown_temp:
            iter_bar.write(f"GPU temp {current}°C > cooldown {cooldown_temp}°C")
            torch.cuda.empty_cache()
            interval = 10
            for _ in range(cooldown_time // interval):
                time.sleep(interval)
                current = self.get_gpu_temp()
                iter_bar.write(f"Current GPU temp: {current}°C")
                if current <= cooldown_temp:
                    iter_bar.write("GPU cooled down, resuming.")
                    break
            else:
                iter_bar.write("Still hot, extending cooldown.")
        return current

    def fit(
        self,
        n_epochs=1000,
        lr=5e-4,
        title="default",
        temp_threshold=65,
        cooldown_temp=50,
        cooldown_time=60
    ):
        optimizer = torch.optim.AdamW(
            self.compiled_model.parameters(), lr=lr, weight_decay=2e-2
        )
        scaler = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=40, verbose=True
        )

        loss_hist, l2_err_hist = [], []
        epoch_iter = tqdm(range(n_epochs), desc='Epoch', position=0, leave=True)

        for epoch in epoch_iter:
            temp = self.get_gpu_temp()
            epoch_iter.write(f"Epoch {epoch} - GPU temp: {temp}°C")
            if temp > temp_threshold:
                temp = self.cool_gpu(
                    temp, temp_threshold, cooldown_temp,
                    cooldown_time, epoch_iter
                )

            self.compiled_model.train()
            epoch_loss = 0.0

            for A, U, max_train, inp, ref in self.train_loader:
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    max_train = max_train.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    U = U[..., 0:1]
                    x = A.permute(0, 4, 1, 2, 3)
                    out = self.compiled_model(x, max_train)
                    out = out.permute(0, 2, 3, 4, 1)
                    loss = torch.linalg.norm(U - out) / torch.clamp(torch.linalg.norm(U), min=1e-6)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compiled_model.parameters(), max_norm=5.0
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            scheduler.step(loss)
            loss_hist.append([epoch_loss / len(self.train_loader)])

            if epoch == 0 or (epoch + 1) % 5 == 0:
                self.compiled_model.eval()
                with torch.no_grad():
                    rL2_rec, rL2 = self.evaluate(
                        title=title, epoch=epoch
                    )
                    l2_err_hist.append([rL2_rec, rL2])

            desc = (
                f"Epoch {epoch} - loss {epoch_loss/len(self.train_loader):.3e} - "
                f"rl2_rec {rL2_rec:.3e}, rl2_enc {rL2:.3e}"
            )
            epoch_iter.set_description(desc)

        return loss_hist, l2_err_hist

    def evaluate(self, title=None, epoch=None):
        rec_nom, rec_den = [], []
        enc_nom, enc_den = [], []

        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            for A, U, inp, ref, max_test in self.test_loader:
                max_test = max_test.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                x = A.permute(0, 4, 1, 2, 3)
                out = self.compiled_model(x, max_test)
                out = out.permute(0, 2, 3, 4, 1)

                grid_u = out.float().permute(0, 4, 1, 2, 3)
                grid_o = x[:, 0:1].float()

                recs = []
                for i in range(grid_u.shape[1]):
                    recs.append(
                        reconstruct_reference_values3d(
                            grid_u[:, i:i+1] / (grid_o + 1e-8),
                            inp.float(),
                            step='trilinear'
                        ).detach()
                    )
                rec = torch.cat(recs, -1)
                rec[..., 0] = inv_minmax(rec[..., 0], 0., 310.)

                rec_nom.append((rec[..., 0] - ref[..., 0]))
                rec_den.append(ref[..., 0])
                enc_nom.append(out - U[..., 0:1])
                enc_den.append(U[..., 0:1])

        all_rec_nom = torch.cat(rec_nom)
        all_rec_den = torch.cat(rec_den)
        all_enc_nom = torch.cat(enc_nom)
        all_enc_den = torch.cat(enc_den)

        rL2_rec = torch.linalg.norm(all_rec_nom) / torch.linalg.norm(all_rec_den)
        rL2_enc = torch.linalg.norm(all_enc_nom) / torch.linalg.norm(all_enc_den)

        return rL2_rec.item(), rL2_enc.item()
