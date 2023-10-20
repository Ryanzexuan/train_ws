# train_ws
# Installation
## Make sure your env is equivalent to settings in [neural-mpc](https://github.com/TUM-AAS/neural-mpc): Acados and ml-casadi
## Further recommended Requirements
- Anaconda env is provided: ```python = 3.9 ``` or
``` conda env create -f py39.yaml```
# Train
``` python src/model_fit/mlp_fitting.py --model_name simple_sim_mlp --hidden_size 64 --hidden_layers 4 --epochs 100```