#!/bin/bash

# AFLOW
# python main.py --dataset_name=aflow --prop_type='bulk_modulus_vrh' --data_filename='oliynyk' \
#                 --hidden_layer_size=256 --hidden_depth=4 --embedding_dim=42 --batch_size=256
# python main.py --dataset_name=aflow --prop_type='debye_temperature' --data_filename='oliynyk' \
#                 --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=42 --batch_size=256
# python main.py --dataset_name=aflow --prop_type='Egap' --data_filename='oliynyk' \
#                 --hidden_layer_size=512 --hidden_depth=3 --embedding_dim=64 --batch_size=256
# python main.py --dataset_name=aflow --prop_type='shear_modulus_vrh' --data_filename='oliynyk' \
#                 --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=48 --batch_size=256
# python main.py --dataset_name=aflow --prop_type='thermal_conductivity_300K' --data_filename='oliynyk' \
#                 --hidden_layer_size=256 --hidden_depth=4 --embedding_dim=42 --batch_size=256
# python main.py --dataset_name=aflow --prop_type='thermal_expansion_300K' --data_filename='oliynyk' \
#                 --hidden_layer_size=256 --hidden_depth=4 --embedding_dim=48 --batch_size=256

# # Matbench
# python main.py --dataset_name=matbench --prop_type='band_gap' --data_filename='magpie' \
#                 --hidden_layer_size=512 --hidden_depth=3 --embedding_dim=64 --batch_size=256
# python main.py --dataset_name=matbench --prop_type='refractive_ind' --data_filename='magpie' \
#                 --hidden_layer_size=512 --hidden_depth=3 --embedding_dim=64 --batch_size=256
# python main.py --dataset_name=matbench --prop_type='yield_strength' --data_filename='magpie' \
#                 --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=32 --batch_size=64

# # Materials Project
# python main.py --dataset_name=mp --prop_type='bulk_modulus' --data_filename='oliynyk' \
#                 --hidden_layer_size=512 --hidden_depth=3 --embedding_dim=64 --batch_size=256
# python main.py --dataset_name=mp --prop_type='elastic_anisotropy' --data_filename='oliynyk' \
#                 --hidden_layer_size=512 --hidden_depth=3 --embedding_dim=64 --batch_size=256
# python main.py --dataset_name=mp --prop_type='shear_modulus' --data_filename='oliynyk' \
#                 --hidden_layer_size=512 --hidden_depth=3 --embedding_dim=64 --batch_size=256

# MoleculeNet
python main.py --dataset_name=molnet --prop_type='bace' --data_filename='rdkit' \
                --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=48 --batch_size=256
python main.py --dataset_name=molnet --prop_type='delaney' --data_filename='rdkit' \
                --hidden_layer_size=1024 --hidden_depth=3 --embedding_dim=64 --batch_size=256
python main.py --dataset_name=molnet --prop_type='freesolv' --data_filename='rdkit' \
                --hidden_layer_size=256 --hidden_depth=4 --embedding_dim=64 --batch_size=256
python main.py --dataset_name=molnet --prop_type='lipo' --data_filename='rdkit' \
                --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=32 --batch_size=256
