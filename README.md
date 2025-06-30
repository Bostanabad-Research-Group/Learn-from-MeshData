## 📖 Learning Mappings in Mesh-based Simulations
Code for the paper [Learning Mappings in Mesh-based Simulations](https://arxiv.org/abs/2506.12652) which introduces a parameter-free encoding scheme for handling point clouds (e.g., nodes in mesh-based data). Our encoder aggregates the linear footprint of points onto grid vertices and provide grid representations of the topology. Such structured representations are well-suited for downstream convolutional or FFT processing and enable efficient learning of mappings between encoded input-output pairs. We integrate this encoder with a customized UNet (E-UNet) and FNO (E-FNO) architechture and evaluate it on various 2D and 3D problems.



![flowchart](figures/flowchart.PNG)
_Figure 1: E-UNet 2D pipeline._

![examples](examples/sol.png)
_Figure 2: Examples._



---
## 🔧 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Bostanabad-Research-Group/Learn-from-MeshData.git
   cd Learn-from-MeshData
2. **Create a conda encironment & install dependencies**
   ```bash
   conda env create -f environment.yml
   ```
   
   ```bash
   conda activate env_learnmesh
   ```

---
## 📦 Prepare Data
Pre-built datasets for each problem are available for download from [here](https://ucirvine-my.sharepoint.com/:u:/g/personal/shirinh1_ad_uci_edu/EfGytUtAxJdEp48KwtdV3PYBMKn1MnA2VMrRo0pM0Cznmw?e=mCZx5v).

After downloading, place the `.pt` files in the `data/` directory as follows:
```
data/
├── naca_data.pt
├── elas_data.pt
├── darcy_data2.pt
├── circles_data.pt
├── maze_data.pt
└── solid3d_data.pt

```
---
## 🏃 Usage

All training and evaluation is managed via `main.py`:

```bash
python main.py \
  --problem naca \
  --model eunet \
  --ntrain 1000 \
  --res 128 \
  --epochs 100 \
  --batch_size 10 \
  --seed 2025
```

| Flag           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `--problem`    | One of `naca`, `elas`, `darcy`, `circles`, `maze`, `solid3d` |
| `--model`      | `efno` or `eunet`                                            |
| `--ntrain`     | Number of training samples                                   |
| `--res`        | Grid resolution (e.g. 128 for 128×128)                       |
| `--epochs`     | Training epochs                                              |
| `--batch_size` | Batch size                                                   |
| `--seed`       | Random seed for reproducibility                              |

After running, check:

- `checkpoints/` for saved model weights (`*_model.pt`)
- `Results/<problem>/history_<title>.csv` for training loss history
- `Results/<problem>/rL2_<title>.csv` for relative L2 errors
- `Results/<problem>/ET_params_<title>.txt` for elapsed time & parameter counts

---

## 📑 Citation
If you use this code or find our work interesting, please cite the following paper:
```bibtex
@article{hosseinmardi2025learning,
  title={Learning Mappings in Mesh-based Simulations},
  author={Hosseinmardi, Shirin and Bostanabad, Ramin},
  journal={arXiv preprint arXiv:2506.12652},
  year={2025}
}
```
    
