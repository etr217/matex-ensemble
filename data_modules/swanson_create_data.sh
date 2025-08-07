# molnet
dataset_name="molnet" # rdkit
properties=("freesolv") #("bace" "delaney" "freesolv" "lipo")
for property in "${properties[@]}"; do
    echo "processing ${dataset_name} ${property}"
    python data_modules/data_process.py --dataset_name="${dataset_name}" --property="${property}" --nan_strategy="drop_sample" --method="swanson" --test_size=.5
done