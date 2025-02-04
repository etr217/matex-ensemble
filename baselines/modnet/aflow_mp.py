import pdb
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
import warnings
warnings.filterwarnings('ignore')
from matminer.datasets import load_dataset
from sklearn.model_selection import train_test_split
from pymatgen.core import Composition
import numpy as np
import pickle
import os
import json
import pandas as pd
from pymatgen.core import Structure
from sklearn.preprocessing import MinMaxScaler
import argparse
import pylab as plt
import scipy


def dist_corr(gt, preds, savepath):
    #dist
    plt.close('all')
    plt.figure()
    plt.hist(gt, density=True, bins=100, alpha=0.5, label='gt')
    plt.hist(preds, density=True, bins=100, alpha=0.5, label='pred')  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.legend()
    plt.savefig(savepath+'_preds.png', bbox_inches='tight', dpi=300) 
    #corr
    plt.close('all')
    plt.figure()
    plt.scatter(gt, preds)
    plt.axline((0, 0), (1, 1), linewidth=1, color='r')
    plt.xlabel('gt')
    plt.ylabel('pred')
    pearson_coeff = scipy.stats.pearsonr(preds.squeeze(), gt.squeeze())
    spearman_coeff = scipy.stats.spearmanr(preds.squeeze(), gt.squeeze())
    plt.title(f'pearson stat: {round(pearson_coeff.statistic,2)}, pval: {round(pearson_coeff.pvalue,2)}\n \
                spearman stat: {round(spearman_coeff.statistic,2)}, pval: {round(spearman_coeff.pvalue,2)}')
    plt.savefig(savepath+'_corr.png', bbox_inches='tight', dpi=300)


def get_split(n_data, test_size, split_type, targets=None):
    # n_data int, test_size %
    if split_type == 'iid':
        return train_test_split(range(n_data), test_size=test_size, random_state=1234)
    elif split_type == 'ood':
        idxs_low2high = np.argsort(targets.values.squeeze())
        cutoff_idx = round(test_size*n_data)
        ood_idxs = list(idxs_low2high[-cutoff_idx:])
        train_idxs = list(idxs_low2high[:-cutoff_idx])
        return [train_idxs, ood_idxs]


def run_modnet(prop_type, dataset_type):
    # data = MODData(
    #     materials=compos, # you can provide composition objects to MODData
    #     targets=ys, 
    #     target_names=["Y"]
    # )
    # data.featurize()
    # OOD split
    # split = get_split(data.get_featurized_df().shape[0], 0.05, 'ood', data.get_target_df()) # 5% ood
    # train, ood = data.split(split)
    # split = get_split(train.get_featurized_df().shape[0], 0.05, 'iid', train.get_target_df()) # 5% of what's left in training
    # train, test = train.split(split)

    data_path = f'/data/pulkitag/misc/avivn/matex/blt/data/{dataset_type}/{prop_type}/raw_data/train.csv'
    df = pd.read_csv(data_path)
    compos, ys = df['formula'].map(Composition), df['target'].values
    train = MODData(materials=compos, targets=ys, target_names=["Y"])
    train.featurize()
    data_path = f'/data/pulkitag/misc/avivn/matex/blt/data/{dataset_type}/{prop_type}/raw_data/eval.csv'
    df = pd.read_csv(data_path)
    compos, ys = df['formula'].map(Composition), df['target'].values
    test = MODData(materials=compos, targets=ys, target_names=["Y"])
    test.featurize()
    data_path = f'/data/pulkitag/misc/avivn/matex/blt/data/{dataset_type}/{prop_type}/raw_data/ood.csv'
    df = pd.read_csv(data_path)
    compos, ys = df['formula'].map(Composition), df['target'].values
    ood = MODData(materials=compos, targets=ys, target_names=["Y"])
    ood.featurize()

    train.feature_selection(n=-1)
    model = MODNetModel([[['Y']]], weights={'Y':1}, num_neurons = [[256], [128], [16], [16]], n_feat = 150, act =  "elu")
    train_X = model.fit(train, val_fraction = 0.1, lr = 0.0002, batch_size = 64, loss = 'mae', epochs = 100, verbose = 1)
    #in dist eval
    pred_in_dist, test_X = model.predict(test)
    test_diffs = np.absolute(pred_in_dist.values-test.df_targets.values)
    mae_test = test_diffs.mean()
    sem_test = np.std(test_diffs, ddof=1) / np.sqrt(np.size(test_diffs))
    print(f'mae in dist: {mae_test}, sem: {sem_test}')
    #ood eval
    pred_ood, ood_X = model.predict(ood)
    ood_diffs = np.absolute(pred_ood.values-ood.df_targets.values)
    mae_ood = ood_diffs.mean()
    sem_ood = np.std(ood_diffs, ddof=1) / np.sqrt(np.size(ood_diffs))
    print(f'mae ood: {mae_ood}, sem: {sem_ood}')
    dataset = {}
    dataset['train_X'] = np.array(train_X)
    dataset['train_Y'] = np.array(train.df_targets.values).reshape(-1, 1)
    dataset['train_formula'] = np.array([str(comp).replace(' ','') for comp in train.compositions])
    dataset['eval_X'] = np.array(test_X)
    dataset['eval_Y'] = np.array(test.df_targets.values).reshape(-1, 1)
    dataset['eval_formula'] = np.array([str(comp).replace(' ','') for comp in test.compositions])
    dataset['ood_X'] = np.array(ood_X)
    dataset['ood_Y'] = np.array(ood.df_targets.values).reshape(-1, 1)
    dataset['ood_formula'] = np.array([str(comp).replace(' ','') for comp in ood.compositions])
    data_dir = f'/data/pulkitag/misc/avivn/matex/blt/data/modnet/{dataset_type}/{prop_type}'
    if not os.path.isdir(data_dir): os.makedirs(data_dir)
    with open(f'{data_dir}/demos.pkl', 'wb') as f:
        pickle.dump([dataset, pred_in_dist, pred_ood], f)
    res_dir = f'/data/pulkitag/misc/avivn/matex/baselines/modnet/{dataset_type}/{prop_type}'
    if not os.path.isdir(res_dir): os.makedirs(res_dir)
    dist_corr(dataset['eval_Y'], pred_in_dist, f'{res_dir}/in_dist')
    dist_corr(dataset['ood_Y'], pred_ood, f'{res_dir}/ood')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', choices=['AFLOW', 'MP'], default='AFLOW')
    parser.add_argument('--prop_type', choices=['band_gap', 'bulk_modulus', 'log_debye', 'log_shear_modulus', 'log_thermal_conductivity', 'log_thermal_expansion', \
                                                'bulk_modulus', 'elastic_anisotropy', 'shear_modulus'], default='band_gap')
    args = parser.parse_args()
    run_modnet(args.prop_type, args.dataset_type)