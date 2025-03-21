# L361 Project: Why go full, when you have momentum?

This repository contains implementations and experiments for federated learning with various momentum-based strategies, focusing on model accuracy, communication efficiency, and robustness to data heterogeneity and client dropout.

## Reproducibility Guidelines

To ensure reproducibility of our results, we have implemented the following measures:

### Environment Setup

```bash
# Create conda environment
conda create -n fedmom python=3.12
conda activate fedmom

# Install dependencies
pip install -r requirements.txt
```

### Running experiments

To run the experiments, navigate to the desired strategy folder and run the corresponding notebook:

1. For FedAvg with momentum experiments:
   ```bash
   cd experiments/fedmom_strategies/fedavgmom
   jupyter notebook fedavg_velocity.ipynb
   ```

2. For FedProx with momentum experiments:
   ```bash
   cd experiments/fedmom_strategies/fedaproxmom
   jupyter notebook fedaprox_velocity.ipynb
   ```

3. For FedMoon with momentum experiments:
   ```bash
   cd experiments/fedmom_strategies/fedmoonmom
   jupyter notebook fedmoon_velocity.ipynb
   ```

4. For partial training experiments:
   ```bash
   cd experiments/fedpart_strategies
   jupyter notebook original_strategies.ipynb
   ```

5. For baseline strategy experiments
    ```bash
    cd experiments/original_strategies
    jupyter notebook fedpart.ipynb
    ```

Each notebook contains detailed instructions and configurations for running the experiments. The results will be saved in the respective strategy folders.

It is important, if you want to save results make sure you have a folder named __results__ where they can be saved. 