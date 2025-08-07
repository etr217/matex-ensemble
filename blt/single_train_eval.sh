# python data_modules/data_process.py --dataset_name="bace" --property="delaney" --nan_strategy="drop_sample"

cd blt

# python main.py --dataset_name=molnet --prop_type='bace' --data_filename='rdkit' --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=48 --batch_size=256
# python main.py --dataset_name=molnet --prop_type='delaney' --data_filename='rdkit' --hidden_layer_size=1024 --hidden_depth=3 --embedding_dim=64 --batch_size=256
python main.py --dataset_name=molnet --prop_type='freesolv' --data_filename='rdkit' --hidden_layer_size=256 --hidden_depth=4 --embedding_dim=64 --batch_size=256
# python main.py --dataset_name=molnet --prop_type='lipo' --data_filename='rdkit' --hidden_layer_size=256 --hidden_depth=3 --embedding_dim=32 --batch_size=256

#python plot_maker/plot_single.py