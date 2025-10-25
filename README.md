# PhARMNet Artifact for “Bridging Data Gaps: Enhancing Wireless Localization with Physics-Informed Data Augmentation”

**Authors:**  
M. M. H. U. Mazumder (University of Utah)  
F. B. Mitchell (Intel Corporation)  
A. Bhaskara (University of Utah)  
S. K. Kasera (University of Utah)  
N. Patwari (University of Utah)

**Conference:**  
ACM International Conference on Emerging Networking Experiments and Technologies (ACM CoNEXT 2025)  
**DOI:** [10.1145/3768995](https://doi.org/10.1145/3768995)  
**Artifact Type:** Source Codes

---

## 📘 Overview
This repository provides the publicly available artifact accompanying the paper  
**“Bridging Data Gaps: Enhancing Wireless Localization with Physics-Informed Data Augmentation.”**

The artifact contains the complete implementation of **PhARMNet**, a physics-informed data augmentation and propagation modeling framework that integrates terrain-based features (TIREM) with neural models to improve wireless localization. It also includes the downstream UNet-based localization model enhanced with B-spline calibration and multi-transmitter clustering.

---

## 📂 Contents
PhARMNet_Artifact/
├─ LICENSE
├─ README.md
├─ Models/   
│   ├─ PhARMNet/
│   │   └─ models_pharmnet.py
│   └─ UNet_Localization_Bspline/
│       ├─ b_spline_silu.py
│       ├─ b_spline_utils.py
│       └─ models_bspline2.py
├─ PhARMNet Feature Generator/
│   ├─ tirem_features_generator.py
│   ├─ tirem_handling_two_transmitter_dataset.py
│   ├─ tirem_helper.py
│   └─ tirem2_handling_two_transmitter_dataset.py
└─ TIREM Libraries/
    ├─ common.py
    ├─ py_tirem_pred.py
    ├─ tirem_params.py
    └─ tirem_params_backup.py


## 🧠 How to Use the Artifact

This artifact provides the complete source code used in the paper’s experiments.  
It is intended for **inspection, reference, and reuse**—no execution or reproduction of results is required for the *Availability* badge.

---

### 📁 Folder Overview

#### `TIREM Libraries/`
Contains the **core Python modules** implementing the TIREM-based physics model used to extract terrain-dependent propagation features.
- `py_tirem_pred.py` – performs TIREM-based path-loss and attenuation calculations.  
- `tirem_params.py` – defines propagation constants, dielectric properties, and environmental parameters.  
- `common.py` – provides shared mathematical and geodesic utilities used across modules.  
- `tirem_params_backup.py` – alternate parameter configuration for ablation testing.

#### `PhARMNet Feature Generator/`
Contains **feature-generation scripts** that interface with the TIREM libraries to produce feature maps and dataset-ready inputs for the PhARMNet model.
- `tirem_features_generator.py` – extracts 14 physics-derived features per transmitter–receiver pair.  
- `tirem_handling_two_transmitter_dataset.py` – creates input maps for dual-transmitter (2-TX) datasets.  
- `tirem2_handling_two_transmitter_dataset.py` – extended handling script for mixed 1-TX/2-TX settings.  
- `tirem_helper.py` – utility functions for I/O, normalization, and visualization.

#### `Models/PhARMNet/`
Contains the **PhARMNet model** definition:
- `models_pharmnet.py` – physics-aware neural network trained on TIREM-corrected RSS data to model propagation residuals.

#### `Models/UNet_Localization_Bspline/`
Contains the **UNet-based localization model** with integrated B-spline calibration and clustering extensions:
- `b_spline_utils.py` – defines spline interpolation utilities and calibration layers.  
- `b_spline_silu.py` – implements differentiable spline-based activation for fine-grained calibration.  
- `models_bspline2.py` – full UNet model definition supporting multi-transmitter inference.

---

## 🌐 Dataset Access

The artifact does **not** include full datasets due to size and licensing, but all datasets used in the paper are publicly accessible and can be downloaded separately:

| Dataset | Description | Access Link |
|----------|--------------|-------------|
| **POWDER-FRS (DS1)** | 462.7 MHz campus-scale RSS measurements | [https://zenodo.org/records/10962857](https://zenodo.org/records/10962857) |
| **POWDER-CBRS (DS2)** | 3.5 GHz multi-transmitter CBRS measurements | [https://github.com/serhatadik/slc-3534MHz-meas](https://github.com/serhatadik/slc-3534MHz-meas) |
| **Antwerp LoRaWAN (DS3)** | City-scale LoRaWAN propagation traces | [https://zenodo.org/records/1193563](https://zenodo.org/records/1193563) |

Each dataset is used in accordance with its original license.  
Scripts in the `PhARMNet Feature Generator` directory can interface with these datasets once downloaded locally.

---

## 🧩 Citation

If you use or reference this artifact, please cite:
M. M. H. U. Mazumder, F. B. Mitchell, A. Bhaskara, S. K. Kasera and N. Patwari,
"Bridging Data Gaps: Enhancing Wireless Localization with Physics-Informed Data Augmentation," Proc. ACM CoNEXT 2025. DOI: 10.1145/3768995.


---

## 🪪 License

This artifact is released under the **MIT License** (see `LICENSE` file).

---

## 📬 Contact

For questions or feedback, please contact:  
**Md Mumtahin Habib Ullah Mazumder**  
📧 [mumtahinhabib0213@gmail.com](mailto:mumtahinhabib0213@gmail.com)
