# 🧬 Single-Cell Gene Regulatory Network Discovery

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)

An advanced deep learning pipeline for inferring directed Gene Regulatory Networks (GRNs) from single-cell RNA sequencing data using Graph Neural Networks (GNNs) and Trajectory Inference.

## 🚀 Key Features

- **Trajectory-Aware GRN Inference**: Integrates Diffusion Pseudotime (DPT) and PAGA to capture dynamic regulatory shifts.
- **Deep Learning Architecture**: Utilizes a GNN-based Variational Autoencoder (VAE) to model non-linear gene interactions.
- **Biological Priors**: Supports integration of TF-target prior knowledge (including human TRRUST v2).
- **Interactive Visualization**: Real-time exploration of inferred networks via an integrated Streamlit dashboard.
- **Scalable Preprocessing**: Built-in support for "Small Data" mode for rapid prototyping on large scRNA-seq datasets.

## 📁 Project Structure

```text
├── data/               # Raw and processed datasets (excluded from git)
├── notebooks/          # Exploratory analysis and training experiments
├── scripts/
│   └── dashboard.py    # Streamlit interactive dashboard
├── src/
│   ├── data/           # Preprocessing and feature engineering
│   ├── models/         # GNN model architecture
│   └── utils/          # Trajectory and visualization logic
├── run_pipeline.py     # Master execution script
└── requirements.txt    # Project dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/psMDHamdan/Single-Cell-Gene-Regulatory-Network-Discovery.git
   cd Single-Cell-Gene-Regulatory-Network-Discovery
   ```

2. **Set up the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Run the Full Pipeline
The pipeline handles everything from data acquisition to model training and static visualization:
```bash
python run_pipeline.py
```

### 2. Launch the Interactive Dashboard
Explore the regulatory hubs and edges interactively:
```bash
streamlit run scripts/dashboard.py
```

## 🧬 Biological Interpretation

The inferred network identifies "Master Regulators"—genes that act as central hubs in cellular decision-making. By analyzing the weight and direction of edges, researchers can identify novel targets for disease intervention or developmental studies.

## 📄 License

This project is licensed under the MIT License.
