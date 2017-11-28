#!/bin/sh

cd /home/ymkim
source /home/ysjang/.bashrc
cd /home/ymkim/sbse/sbse_parameter_optimization
python optimizer.py -f f_mnist -algo GA -n_samples 100 -n_evals 1
