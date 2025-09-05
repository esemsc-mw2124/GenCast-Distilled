# Distillation of GenCast for Efficient Diffusion-Based Weather Forecasting

This repository contains the implementation and supporting materials for my IRP studying the distillation of GenCast, a state-of-the-art conditional diffusion model for medium-range weather prediction. The work builds on GenCast, introduced by DeepMind, and investigates progressive distillation as a means to improve inference efficiency without significantly compromising model skill.

The project is documented in the accompanying [final report](./deliverables/mw2124-final-report.pdf).

<p align="center">
<img width="512" height="768" alt="gencast-model" src="https://github.com/user-attachments/assets/3c2aecb0-0773-42c6-ad6e-199daea5ffdd" />
</p>

### Project Overview

Recent advances in machine learning-based weather prediction (MLWP) have outperformed traditional numerical weather prediction (NWP) models in both skill and efficiency. GenCast is a probabilistic diffusion-based model capable of generating 15-day global forecasts in just 8 minutes.

This project investigates whether **progressive distillation**, a technique that reduces the number of diffusion steps required, can make GenCast even more efficient. Although full training and evaluation of a distilled model was not possible due to compute limitations, the training pipeline was successfully implemented and verified on reduced workloads. These results lay the groundwork for further experimentation and possible large-scale deployment of a distilled GenCast.

### Repository Structure

```
.
├── deliverables
│   ├── mw2124-final-report.pdf
│   ├── mw2124-project-plan.pdf
│   └── README.md
├── evaluation
│   ├── eval_helpers.py
│   ├── eval.ipynb
│   ├── inference_helpers.py
│   └── plotting_helpers.py
├── gencast_distillation
│   ├── __init__.py
│   ├── config.py
│   ├── get_data.py
│   ├── losses.py
│   ├── model.py
│   ├── run_train.py
│   ├── training.py
│   └── utils.py
├── logbook
│   ├── logbook.md
│   └── README.md
├── misc
│   ├── activate_vm.sh
│   └── delete_vm.sh
├── patches
│   ├── data_utils.py
│   └── dpm_solver_plus_plus_2s.py
├── pyproject.toml
├── python-version.txt
├── README.md
├── requirements.txt
└── title
    ├── README.md
    └── title.toml

```
---

### Reproducibility

This project is designed to be reproducible and open source. Important resources include:

- Final report: [deliverables/mw2124-final-report.pdf](./deliverables/mw2124-final-report.pdf)
- Open-source license: Apache 2.0
- Evaluation notebook: [`evaluation/eval.ipynb`](./evaluation/eval.ipynb)

The project builds on the GenCast and GraphCast codebases released by DeepMind:
- https://github.com/google-deepmind/graphcast

### Data Access

The model relies on ERA5 reanalysis data, which can be accessed through:

- ECMWF: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
- WeatherBench2 (recommended): https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5

Data loading and preprocessing are implemented in `gencast_distillation/get_data.py`.

### Inference and Evaluation

The repository includes code to run inference and evaluate the model:

1. Go to the evaluation directory:

    ```bash
    cd evaluation
    ```

2. Launch the notebook:

    ```bash
    jupyter notebook eval.ipynb
    ```

This notebook demonstrates how to load a model, run forecasts, and evaluate performance using standard meteorological metrics. Unfortunately, it only run on a TPU VM.


### Training the Distilled Model

Training logic is implemented in `gencast_distillation/run_train.py`. To begin training:

```bash
python -m gencast_distillation.run_train.py
```

---

### TPU VM Setup

This section provides instructions for setting up a TPU VM on Google Cloud Platform (GCP) to run and train the distilled GenCast model. The setup includes creating a project with billing enabled, provisioning a TPU VM, and installing dependencies.

#### 1. Create a TPU VM on Google Cloud

##### 1.1 One-time GCP Setup

Follow this guide to:
- Create a Google Cloud account
- Set up a new project with billing enabled
- Install and initialize the `gcloud` CLI on your local machine

**Guide:** [Setting up Google Cloud and gcloud CLI](https://cloud.google.com/sdk/docs/install)

##### 1.2 Activate the TPU VM

In your terminal, run the following setup script:

```bash
bash misc/activate_vm.sh
```

This provisions the TPU VM with a valid configuration.

##### 1.3 Connect to the TPU VM

Set the following environment variables and connect via SSH:

```bash
export TPU_NAME=gencast-mini-vm
export ZONE=us-south1-a
export PROJECT_ID=gencast-distillation

gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE

gcloud compute tpus tpu-vm describe $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

##### 1.4 Configure SSH Access

Copy the value of `accessConfig: externalIp` from the output above.

Then, run:

```bash
nano ~/.ssh/config
```

Paste the following, replacing `<ip>` and `<your-username>` accordingly:

```
Host tpu-vm
  HostName <ip>
  User <your-username>
  IdentityFile ~/.ssh/google_compute_engine
```

#### 2. Inside the TPU VM

Once connected to the VM, set up the environment:

```bash
git clone https://github.com/esemsc-mw2124/GenCast-Distilled.git
cd GenCast-Distilled/

sudo apt update
sudo apt install python3.10-venv

python3 -m venv venv
source venv/bin/activate

pip install -e .
pip install -r requirements.txt

cp -r ./patches/* venv/lib/python3.10/site-packages/graphcast
```

Your TPU VM environment is now fully set up.

#### 3. VM Cost and Teardown

Note: TPU VMs can be expensive — approximately **£40 per 24 hours**. Be sure to deactivate the VM when not in use:

```bash
bash misc/remove_vm.sh
```

---

### References

1. R. Lam et al. Learning skillful medium-range global weather forecasting. Science,
382:1416–1421, 2023.
2. K. Bi et al. Accurate medium-range global weather forecasting with 3d neural net-
works. Nature, 619:533–538, 2023.
3. I. Price, A. Sanchez-Gonzalez, F. Alet, et al. Probabilistic weather forecasting with
machine learning. Nature, 637:84–90, 2025.
4. T. Salimans and J. Ho. Progressive distillation for fast sampling of diffusion models.
In International Conference on Learning Representations, 2022.
