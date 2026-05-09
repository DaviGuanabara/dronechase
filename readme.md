
# Dronechase



Read-me outdated. 
Maintenance in progress.
I appreciate your patience. 


pip install .[interactive]

**Dronechase** is a multi-stage simulation framework for training autonomous aerial agents in defense scenarios using Deep Reinforcement Learning (DRL). The project focuses on **loyal wingmen drones** engaging loitering munitions in a 3D physics-based environment powered by PyBullet.

Developed as part of a Master’s thesis in Aeronautical Computing at ITA.

---

## 🚀 Project Overview

- 🧠 **Reinforcement Learning** with Stable Baselines3  
- 📡 **3D Simulation** using PyFlyt and PyBullet  
- 🛰️ **Loyal Wingmen vs. Loitering Munitions**  
- 🧪 **Curriculum-style training stages (stage01 → stage03)**  

Each stage increases environmental complexity and decision-making requirements.

---

## 🗂️ Repository Structure

```bash
apps/                    # Training and evaluation scripts by stage
├── stage01/             # Hyperparameter optimization with single threat
├── stage02/             # Generalization to multiple threats
├── stage03/             # Full scenario with active invaders
│   ├── support/         # Model loading and visualization scripts
│   └── docs/            # Stage-specific documentation
src/
├── threatengage/        # Core training environments and entities
├── threatsense/         # Experimental modules (e.g., LIDAR)
```

---

## ⚙️ Installation

Ensure you have C++ build tools installed (e.g., MSVC v14+, CMake, Windows SDK). Then:

```bash
git clone https://github.com/DaviGuanabara/dronechase.git
cd dronechase
pip install --all-extras
```

> 📦 Optional extras:  

> `pip install --extras "viz platform_input"` (includes Seaborn and TensorBoard)


---

## 📦 How to Use

Navigate to the `apps/` folder and run each stage's appropriate training or evaluation script.

Example:

```bash
python apps/stage01/train.py
```

Each stage corresponds to an experimental setup described in the academic dissertation.

---

## 📄 Academic References

### 🎓 Dissertation

**"Deep Reinforcement Learning Applied to Threat Engagement for Loyal Wingmen Drones"**  
Davi Guanabara Aragão – M.Sc. in Aeronautical Computing, ITA, 2024  
📎 [Access via ITA Library (login required)](http://www.bdita.bibl.ita.br/)

### 📝 Scientific Article

**"Deep Reinforcement Learning Applied for Threat Engagement by Loyal Wingmen Drones"**  
Accepted at IEEE SBR/WRE 2024  
📎 [View on IEEE Xplore](https://ieeexplore.ieee.org/document/10837749)

---

### 📁 Pretrained Models

All pretrained models used in the dissertation are available here:  
🔗 [OneDrive – Model Outputs](https://1drv.ms/f/c/1d0046dc1ae1123c/EjwS4RrcRgAggB0mxAYAAAABv3y_2LSkT8CMERe7Hf5ZXA?e=KYmCQj)

---

## ✈️ Citation

If you use this project, please cite:

```bibtex
@INPROCEEDINGS{10837749,
  author={De Aragão, Davi Guanabara and Maximo, Marcos R. O. A. and Fernando Basso Brancalion, José},
  booktitle={2024 Brazilian Symposium on Robotics (SBR) and 2024 Workshop on Robotics in Education (WRE)}, 
  title={Deep Reinforcement Learning Applied for Threat Engagement by Loyal Wingmen Drones}, 
  year={2024},
  volume={},
  number={},
  pages={56-61},
  keywords={Laser radar;Weapons;Conferences;Education;Decision making;Deep reinforcement learning;Vehicle dynamics;Robots;Optimization;Drones;Machine Learning;Neural Network;Unmanned Aerial Vehicle;Air Defense System},
  doi={10.1109/SBR/WRE63066.2024.10837749}}
```



Requirements:
conda create -n dronechase python=3.11
conda activate dronechase

conda install -c conda-forge pybullet

pip install pyflyt==0.11.1 --no-deps

poetry install


conda install -c pytorch pytorch torchvision
conda install -c fvcore -c iopath -c conda-forge fvcore iopath



conda install -c fvcore -c iopath -c conda-forge fvcore iopath ninja


## MACOS MX Installation


### Install PyTorch3D MACOS (requires PyTorch 2.4.1)

conda activate dronechase310

python -c "import torch; print(torch.__version__)"
confirme que é 2.4.1. Depois:

export MACOSX_DEPLOYMENT_TARGET=10.14
export CC=clang
export CXX=clang++
export CXXFLAGS="-Wno-invalid-specialization"
export CFLAGS="-Wno-invalid-specialization"
export MAX_JOBS=1

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
  --no-build-isolation \
  --no-cache-dir \
  -v

### Install Pybullet 3.2.4 (for PyFlyt)
conda activate dronechase310
conda install -c conda-forge pybullet

### Install Pyflyt 0.11.1 (for simulation)
pip install pyflyt==0.11.1 --no-deps

### Install project dependencies with Poetry
poetry lock
poetry install