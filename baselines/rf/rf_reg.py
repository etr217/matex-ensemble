import pickle
import scipy
import os
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score 

BASE_PATH = '/home/gridsan/nsegal/tr-ext'

def save_pkl(data, logpath):
    with open(logpath, 'wb') as f:
        pickle.dump(data, f)

def random_forest_regression(pkl_file, save_path):
    # Load the data from the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract train, eval, and OOD datasets
    train_X, train_y = data['train_X'], data['train_Y'].ravel()
    eval_X, eval_y = data['eval_X'], data['eval_Y'].ravel()
    ood_X, ood_y = data['ood_X'], data['ood_Y'].ravel()

    # Initialize the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=42)
    
    # Train the model
    rf_model.fit(train_X, train_y)
    
    # Make predictions on the evaluation and OOD sets
    eval_preds = rf_model.predict(eval_X)
    ood_preds = rf_model.predict(ood_X)
    
    # Evaluate the model performance
    eval_mae = round(mean_absolute_error(eval_y, eval_preds), 3)
    ood_mae = round(mean_absolute_error(ood_y, ood_preds), 3)

    eval_sem = round(scipy.stats.sem(np.abs(eval_y-eval_preds)), 3)
    ood_sem = round(scipy.stats.sem(np.abs(ood_y-ood_preds)), 3)
    
    print(f"Evaluation Set MAE: {eval_mae} \u00b1 {eval_sem}")
    print(f"OOD Set MAE: {ood_mae} \u00b1 {ood_sem}")

    pearson_coeff_id_stat = round((scipy.stats.pearsonr(eval_preds.squeeze(), eval_y.squeeze())).statistic, 3)
    pearson_coeff_id_pval = max(round((scipy.stats.pearsonr(eval_preds.squeeze(), eval_y.squeeze())).pvalue, 3), 0.001)
    spearman_coeff_id_stat = round((scipy.stats.spearmanr(eval_preds.squeeze(), eval_y.squeeze())).statistic, 3)
    spearman_coeff_id_pval = max(round((scipy.stats.spearmanr(eval_preds.squeeze(), eval_y.squeeze())).pvalue, 3), 0.001)

    pearson_coeff_ood_stat = round((scipy.stats.pearsonr(ood_preds.squeeze(), ood_y.squeeze())).statistic, 3)
    pearson_coeff_ood_pval = max(round((scipy.stats.pearsonr(ood_preds.squeeze(), ood_y.squeeze())).pvalue, 3), 0.001)
    spearman_coeff_ood_stat = round((scipy.stats.spearmanr(ood_preds.squeeze(), ood_y.squeeze())).statistic, 3)
    spearman_coeff_ood_pval = max(round((scipy.stats.spearmanr(ood_preds.squeeze(), ood_y.squeeze())).pvalue, 3), 0.001)

    r2_id = round(r2_score(eval_y, eval_preds), 3)
    r2_ood = round(r2_score(ood_y, ood_preds), 3)

    print(f"Pearson id: {pearson_coeff_id_stat, pearson_coeff_id_pval}, ood: {pearson_coeff_ood_stat, pearson_coeff_ood_pval}")
    print(f"Spearman id: {spearman_coeff_id_stat, spearman_coeff_id_pval}, ood: {spearman_coeff_ood_stat, spearman_coeff_ood_pval}")
    print(f'r2 scores id: {r2_id}, ood: {r2_ood}')

    results = {
        'eval_preds': eval_preds,
        'ood_preds': ood_preds,
        'eval_gt': eval_y,
        'ood_gt': ood_y,
        'eval_mae': eval_mae,
        'eval_sem': eval_sem,
        'ood_mae': ood_mae,
        'ood_sem': ood_sem,
        'pearson_id': [pearson_coeff_id_stat, pearson_coeff_id_pval],
        'spearman_id': [spearman_coeff_id_stat, spearman_coeff_id_pval],
        'pearson_ood': [pearson_coeff_ood_stat, pearson_coeff_ood_pval],
        'spearman_ood': [spearman_coeff_ood_stat, spearman_coeff_ood_pval],
        'r2_id': r2_id,
        'r2_ood': r2_ood
    }

    save_pkl(results, os.path.join(save_path, 'rf_res.pkl'))


# Example usage
prop = 'bace'
pkl_file = f'/home/gridsan/nsegal/tr-ext/matex/blt/data/molnet/{prop}/demos_rdkit_normalized.pkl'
save_path = os.path.join(BASE_PATH, f'rf/{prop}')
random_forest_regression(pkl_file, save_path)
