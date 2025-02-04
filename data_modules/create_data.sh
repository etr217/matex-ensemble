#!/bin/bash

#SBATCH --job-name=create_data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3  # Adjust as needed

# source /etc/profile
# source $HOME/.bashrc

# aflow
dataset_name="aflow" # oliynyk
properties=("bulk_modulus_vrh" "debye_temperature" "Egap" "shear_modulus_vrh" "thermal_conductivity_300K" "thermal_expansion_300K")
for property in "${properties[@]}"; do
    echo "processing ${dataset_name} ${property}"
    python data_modules/data_process.py --dataset_name="${dataset_name}" --property="${property}"
done

# matbench
dataset_name="matbench" # magpie
properties=("band_gap" "refractive_ind" "yield_strength")
for property in "${properties[@]}"; do
    echo "processing ${dataset_name} ${property}"
    python data_modules/data_process.py --dataset_name="${dataset_name}" --property="${property}" --nan_strategy="drop_feat"
done

# mp
dataset_name="mp" # oliynyk
properties=("bulk_modulus" "elastic_anisotropy" "shear_modulus")
for property in "${properties[@]}"; do
    echo "processing ${dataset_name} ${property}"
    python data_modules/data_process.py --dataset_name="${dataset_name}" --property="${property}"
done

# molnet
dataset_name="molnet" # rdkit
properties=("bace" "delaney" "freesolv" "lipo")
for property in "${properties[@]}"; do
    echo "processing ${dataset_name} ${property}"
    python data_modules/data_process.py --dataset_name="${dataset_name}" --property="${property}" --nan_strategy="drop_sample"
done
