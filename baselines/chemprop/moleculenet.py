import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn
import scipy
import argparse
import numpy as np
import os
import pdb
from scipy import stats

def plot_parity(y_true, y_pred, y_pred_unc=None, savepath=''):
    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))
    mae = mean_absolute_error(y_true, y_pred)
    sem = stats.sem(np.abs(y_true - y_pred))
    # pdb.set_trace()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    pearson_coeff = scipy.stats.pearsonr(y_pred, y_true)
    spearman_coeff = scipy.stats.spearmanr(y_pred, y_true)
    r_squared = sklearn.metrics.r2_score(y_true, y_pred)
    plt.plot([axmin, axmax], [axmin, axmax], '--k')
    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', alpha=1, elinewidth=1)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    ax = plt.gca()
    ax.set_aspect('equal')
    at = AnchoredText(
    f"MAE = {mae:.2f} ({sem:.4f})\nRMSE = {rmse:.2f}\nPearson = {pearson_coeff.statistic:.2f} ({pearson_coeff.pvalue:.2f})\nSpearman = {spearman_coeff.statistic:.2f} ({spearman_coeff.pvalue:.2f})\nRsquared = {r_squared}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    plt.xlabel('True')
    plt.ylabel('Chemprop Predicted')
    plt.savefig(savepath)

def main(prop):
    # #ID
    # #train
    # arguments = [
    #     '--data_path', f'/data/pulkitag/misc/avivn/matex/blt/data/molnet/{prop}/train.csv', #train
    #     '--dataset_type', 'regression',
    #     '--save_dir', f'test_checkpoints_{prop}',
    #     '--epochs', '10',
    #     '--save_smiles_splits'
    # ]
    # args = chemprop.args.TrainArgs().parse_args(arguments)
    # mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    # #pred 
    # arguments = [
    #     '--test_path', f'test_checkpoints_{prop}/fold_0/test_smiles.csv',
    #     '--preds_path', f'test_checkpoints_{prop}/fold_0/test_preds.csv',
    #     '--checkpoint_dir', f'test_checkpoints_{prop}'
    # ]
    # args = chemprop.args.PredictArgs().parse_args(arguments)
    # preds = chemprop.train.make_predictions(args=args)
    # df = pd.read_csv(f'test_checkpoints_{prop}/fold_0/test_full.csv')
    # df['preds'] = [x[0] for x in preds]
    # plot_parity(df['target'], df.preds, savepath=f'test_checkpoints_{prop}/training_parity.png')

    #OOD SPLIT
    #TRAIN (will split to train-validation. val required for early stopping)
    arguments = [
        '--data_path', f'/data/pulkitag/misc/avivn/matex/blt/data/molnet/{prop}/train.csv',
        '--dataset_type', 'regression',
        '--save_dir', f'test_checkpoints_{prop}',
        '--epochs', '30',
        '--save_smiles_splits',
        '--separate_test_path', f'/data/pulkitag/misc/avivn/matex/blt/data/molnet/{prop}/eval.csv'
    ]
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    #EVAL ID
    arguments = [
        '--test_path', f'moleculenet/{prop}_eval_id_smiles.csv', #just smiles
        '--preds_path', f'test_checkpoints_{prop}/eval_id_preds_{prop}.csv',
        '--checkpoint_dir', f'test_checkpoints_{prop}'
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
    df = pd.read_csv(f'/data/pulkitag/misc/avivn/matex/blt/data/molnet/{prop}/eval.csv') #gt ys
    df['preds'] = [x[0] for x in preds]
    plot_parity(df['target'].values, df.preds.values, savepath=f'test_checkpoints_{prop}/eval_id_parity.png')  
    #EVAL OOD
    arguments = [
        '--test_path', f'moleculenet/{prop}_eval_ood_smiles.csv', #just smiles
        '--preds_path', f'test_checkpoints_{prop}/eval_ood_preds_{prop}.csv',
        '--checkpoint_dir', f'test_checkpoints_{prop}'
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
    df = pd.read_csv(f'/data/pulkitag/misc/avivn/matex/blt/data/molnet/{prop}/ood.csv') #gt ys
    df['preds'] = [x[0] for x in preds]
    plt.figure()
    plot_parity(df['target'].values, df.preds.values, savepath=f'test_checkpoints_{prop}/eval_ood_parity.png')    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--property', choices=['delaney', 'freesolv', 'lipo', 'bace'], default='delaney')
    args = parser.parse_args()

    #DATA
    #save all data baseline format
    if not os.path.isfile(f'moleculenet/{args.property}_eval_id_smiles.csv'):
        eval_id = pd.read_csv(f'/data/pulkitag/misc/avivn/matex/blt/data/molnet/{args.property}/eval.csv')
        df = pd.DataFrame({'smiles': eval_id.formula})
        df.to_csv(f'moleculenet/{args.property}_eval_id_smiles.csv', index=False)
    if not os.path.isfile(f'moleculenet/{args.property}_eval_ood_smiles.csv'):
        eval_ood = pd.read_csv(f'/data/pulkitag/misc/avivn/matex/blt/data/molnet/{args.property}/ood.csv')
        df = pd.DataFrame({'smiles': eval_ood.formula})
        df.to_csv(f'moleculenet/{args.property}_eval_ood_smiles.csv', index=False)

    main(args.property)