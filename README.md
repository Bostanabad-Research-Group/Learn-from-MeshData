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
    
