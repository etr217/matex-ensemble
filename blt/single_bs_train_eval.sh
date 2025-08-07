# python bs_main.py --dataset_name=molnet --prop_type='bace' --data_filename='rdkit' --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=48 --batch_size=256
# python bs_main.py --dataset_name=molnet --prop_type='delaney' --data_filename='rdkit' --hidden_layer_size=1024 --hidden_depth=3 --embedding_dim=64 --batch_size=256 --pct=.5
python bs_main.py --dataset_name=molnet --prop_type='freesolv' --data_filename='rdkit' --hidden_layer_size=256 --hidden_depth=4 --embedding_dim=64 --batch_size=256 --pct=.85
# python bs_main.py --dataset_name=molnet --prop_type='lipo' --data_filename='rdkit' --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=32 --batch_size=256
python plot_maker/plot_single.py