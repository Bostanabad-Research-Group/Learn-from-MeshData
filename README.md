# Learn-from-MeshData
Learning input-output relations in mesh-based simulations


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

    
