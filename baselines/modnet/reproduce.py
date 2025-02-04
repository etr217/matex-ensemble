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

# srun --pty --partition vision-pulkitag-a100 --qos vision-pulkitag-debug --time=02:00:00 --cpus-per-task 1 --gres gpu:1 --mem=400G python reproduce.py

##########################################

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

# prop_type = 'matbench_steels' #matbench_expt_gap, matbench_steels, matbench_dielectric

def run_modnet(prop_type):

    if prop_type == "matbench_expt_gap": #band gap
        print('matbench_expt_gap')
        # EXPT GAP
        df = load_dataset(prop_type)
        df["composition"] = df["composition"].map(Composition)
        # df = df[df["gap expt"] != 0].reset_index() #TODO remove 0s
        data = MODData(
            materials=df["composition"], # you can provide composition objects to MODData
            targets=df["gap expt"], 
            target_names=["gap_expt_eV"]
        )
        data.featurize()
        # with open(f'featurized_data/{prop_type}.pkl', 'wb') as handle:
        #     pickle.dump([data.get_featurized_df()], handle) #feat:270
        
        # REPRODUCE
        # split = get_split(data.get_featurized_df().shape[0], 0.1, 'iid')
        # train, test = data.split(split)
        # OOD split
        split = get_split(data.get_featurized_df().shape[0], 0.05, 'ood', data.get_target_df()) # 5% ood
        train, ood = data.split(split)
        split = get_split(train.get_featurized_df().shape[0], 0.05, 'iid', train.get_target_df()) # 5% of what's left in training
        train, test = train.split(split)
        train.feature_selection(n=-1)
        model = MODNetModel([[['gap_expt_eV']]], weights={'gap_expt_eV':1}, num_neurons = [[256], [128], [16], [16]], n_feat = 150, act =  "elu")
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
        # with open(f'res/{prop_type}.pkl', 'wb') as f:
        #     pickle.dump([train, test, ood, test.df_targets.values, pred_in_dist, ood.df_targets.values, pred_ood], f)
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
        with open(f'/data/pulkitag/misc/avivn/matex/blt/data/modnet/band_gap/demos_egap_01.pkl', 'wb') as f:
            pickle.dump([dataset, pred_in_dist, pred_ood], f)
        dist_corr(dataset['eval_Y'], pred_in_dist, f'res/expt_gap_01_in_dist')
        dist_corr(dataset['ood_Y'], pred_ood, f'res/expt_gap_01_ood')


    elif prop_type == "matbench_steels": #yield strength
        print('matbench_steels')
        #STEEL STRENGTH
        df = load_dataset(prop_type)
        df["composition"] = df["composition"].map(Composition)
        data = MODData(
            materials=df["composition"], # you can provide composition objects to MODData
            targets=df["yield strength"], 
            target_names=["steels"]
        )
        data.featurize()
        # with open(f'featurized_data/{prop_type}.pkl', 'wb') as handle:
        #     pickle.dump([data.get_featurized_df()], handle) #feat:268

        # REPRODUCE
        # split = get_split(data.get_featurized_df().shape[0], 0.1, 'iid') 
        # train, test = data.split(split)
        # OOD split
        split = get_split(data.get_featurized_df().shape[0], 0.05, 'ood', data.get_target_df()) # 5% ood
        train, ood = data.split(split)
        split = get_split(train.get_featurized_df().shape[0], 0.05, 'iid', train.get_target_df()) # 5% of what's left in training
        train, test = train.split(split)
        train.feature_selection(n=-1)
        model = MODNetModel([[['steels']]], weights={'steels':1}, num_neurons = [[256], [128], [16], [16]], n_feat = 104, act =  "elu")
        train_X = model.fit(train, val_fraction = 0.1, lr = 0.0002, batch_size = 64, loss = 'mae', epochs = 5000, verbose = 1)
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
        # with open(f'res/{prop_type}.pkl', 'wb') as f:
        #     pickle.dump([train, test, ood, test.df_targets.values, pred_in_dist, ood.df_targets.values, pred_ood], f)
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
        with open(f'/data/pulkitag/misc/avivn/matex/blt/data/modnet/yield_strength/demos_yield_str_01.pkl', 'wb') as f:
            pickle.dump([dataset, pred_in_dist, pred_ood], f)
        

    elif prop_type == 'matbench_dielectric': #refractive ind
        print('matbench_dielectric')
        # #REFRACTIVE IDX (HAVE PRETRAINED MODEL)
        #TODO not loading
        #PRETRAINED
        # model = MODNetModel.load('pretrained/refractive_index.json') #something might have gone wrong downloading this
        # MP_data = MODData.load("../moddata/MP_2018.6") #not saved locally
        # df = model.predict(MP_data)

        #TRAIN
        # df = load_dataset(prop_type)
        # data = MODData(
        #     materials=df["structure"],
        #     targets=df["n"], 
        #     target_names=["refractive_idx"]
        # )
        # data.featurize()

        def extract_composition(container):
            return container.composition
        def get_comp_from_struc(structure_dict):
            structure = Structure.from_dict(structure_dict)
            return structure.composition    
        with open('/data/pulkitag/misc/avivn/matex/blt/data/matbench/refractive_ind/refractive_ind.json') as data_file:
            data=json.load(data_file)
        data_records = data["data"]
        df_og = pd.DataFrame(data_records)
        df_og.columns = ['composition', 'refractive_ind']
        df_og['composition'] = df_og['composition'].apply(get_comp_from_struc)
        df_og['composition'] = df_og['composition'].map(Composition)
        mod_data = MODData(
            materials=df_og['composition'], # you can provide composition objects to MODData
            targets=df_og['refractive_ind'], 
            target_names=['refractive_ind']
        )
        mod_data.featurize()
        # df_feat = mod_data.get_featurized_df()
        # if (
        #     df_feat.isna().any().any()
        # ):  # only preprocess if nans are present 
        #     scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        #     x = df_feat.values
        #     x = scaler.fit_transform(x)
        #     x = np.nan_to_num(x, nan=-1)
        #     df_feat = pd.DataFrame(x, index=df_feat.index, columns=df_feat.columns)
        # with open(f'featurized_data/{prop_type}.pkl', 'wb') as handle:
        #     pickle.dump([df_feat], handle)

        # REPRODUCE
        # split = get_split(mod_data.get_featurized_df().shape[0], 0.1, 'iid') 
        # train, test = mod_data.split(split)
        # OOD split
        split = get_split(mod_data.get_featurized_df().shape[0], 0.05, 'ood', mod_data.get_target_df()) # 5% ood
        train, ood = mod_data.split(split)
        split = get_split(train.get_featurized_df().shape[0], 0.05, 'iid', train.get_target_df()) # 5% of what's left in training
        train, test = train.split(split)
        train.feature_selection(n=-1)
        model = MODNetModel([[['refractive_ind']]], weights={'refractive_ind':1}, num_neurons = [[256], [128], [16], [16]], n_feat = 150, act =  "elu")
        train_X = model.fit(train, val_fraction = 0.1, lr = 0.0002, batch_size = 64, loss = 'mae', epochs = 200, verbose = 1)
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
        # with open(f'res/{prop_type}.pkl', 'wb') as f:
        #     pickle.dump([train, test, ood, test.df_targets.values, pred_in_dist, ood.df_targets.values, pred_ood], f)
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
        with open(f'/data/pulkitag/misc/avivn/matex/blt/data/modnet/refractive_ind/demos_ref_ind.pkl', 'wb') as f:
            pickle.dump([dataset, pred_in_dist, pred_ood], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prop_type', choices=['matbench_expt_gap','matbench_steels','matbench_dielectric'], default='matbench_steels')
    args = parser.parse_args()
    run_modnet(args.prop_type)