#!/bin/bash

#SBATCH --time=20:00
#SBATCH --output=convert_dataset.out

module load eth_proxy #need to load this to access anything outside the ETH network on a cluster node
cd ~
source .venv/bin/activate
