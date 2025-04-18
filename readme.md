# Dronechase

**Dronechase** is a multi-stage simulation framework for training autonomous aerial agents in defense scenarios using Deep Reinforcement Learning (DRL). The project focuses on **loyal wingmen drones** engaging loitering munitions in a 3D physics-based environment powered by PyBullet.

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

Make sure you have C++ build tools installed (e.g., MSVC v14+, CMake, Windows SDK). Then run:

```bash
git clone https://github.com/DaviGuanabara/dronechase.git
cd dronechase
poetry install --all-extras
```

> 📦 Optional extras: `poetry install --extras "viz"` (includes Seaborn and TensorBoard)

---

## 📦 How to Use

Navigate to the `apps/` folder and run the appropriate training or evaluation script for each stage.

Example:

```bash
poetry run python apps/stage01/train.py
```

Each stage is independent and corresponds to a specific experimental setup described in the dissertation.

---

## 📄 Academic References

### 🎓 Dissertation

**"Deep Reinforcement Learning Applied to Threat Engagement for Loyal Wingmen Drones"**  
Davi Guanabara Aragão – M.Sc. in Aeronautical Computing, ITA, 2024  
📎 [Access via ITA Library (login required)](http://www.bdita.bibl.ita.br/)  
---

### 📝 Scientific Article

**"Deep Reinforcement Learning Applied for Threat Engagement by Loyal Wingmen Drones"**  
Accepted at IEEE SBR/WRE 2024  
📎 [Access on IEEE Xplore](https://ieeexplore.ieee.org/document/10837749)  
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

