import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from utils.util import save_pkl

def get_UQ(path):
    folders = [osp.join(path, next) for next in os.listdir(path)]
    fnames = ['labelled_data.pkl', 'bilinear_eval_ood.pkl', 'bilinear_eval_in_dist.pkl']
    data_lists = {fname : {} for fname in fnames}
    actives = folders[:]
    for folder in folders:
        full_fnames = [osp.join(folder, fname) for fname in fnames]
        active = True
        for full_fname in full_fnames:
            # print(folder)
            # if folder == '/Users/elimeyers/matexes/matex_ensemble/blt/log/molnet/freesolv/rdkit_subtraction_bilinear_hsize256_hnum4_esize64_bsize256/20/25-07-25_15-04-55':
                # print(osp.exists(full_fname))
            if not osp.exists(full_fname):
                active = False

        if active:
            for full_fname, fname in zip(full_fnames, fnames):
                data_lists[fname][folder] = pd.read_pickle(full_fname)
        else:
            actives.remove(folder)
            print(f'removed {folder}')
    folders = actives
    datas = data_lists[fnames[0]]
    oods = data_lists[fnames[1]]
    in_dists = data_lists[fnames[2]]
    
    for folder in folders:
        for key in oods[folder].keys():
            if not isinstance(oods[folder][key], list):
                oods[folder][key] = oods[folder][key].squeeze()
                in_dists[folder][key] = in_dists[folder][key].squeeze()
        oods[folder]['formula'] = datas[folder]['ood_formula']
        in_dists[folder]['formula'] = datas[folder]['eval_formula']
        oods[folder] = pd.DataFrame(oods[folder])
        in_dists[folder] = pd.DataFrame(in_dists[folder])
    all_ood = pd.concat(oods)
    all_in_dist = pd.concat(in_dists)

    all_eval = pd.concat([all_ood, all_in_dist])

    all_eval['error'] = all_eval['preds'] - all_eval['gt']
    all_eval['abs_error'] = all_eval['error'].abs()
    all_eval['sq_error'] = all_eval['error']**2

    group_df = all_eval.groupby('formula').mean()
    group_df['pred_std'] = all_eval.groupby('formula')['preds'].std()
    group_df['total_abs_error'] = abs(group_df['error'])

    group_path = os.path.join(path, 'group_df.pkl')
    all_path = os.path.join(path, 'all_eval.pkl')
    data_path = os.path.join(path, 'data.pkl')
    if os.path.exists(group_path):
        os.remove(group_path)
        print('removed existing group df!')
    if os.path.exists(all_path):
        os.remove(all_path)
        print('removed existing eval df!')
    if os.path.exists(data_path):
        os.remove(data_path)
        print('removed existing data df!')
    group_df.to_pickle(group_path)
    all_eval.to_pickle(all_path)
    save_pkl(datas[actives[-1]], logpath = data_path)
    return all_eval, group_df

def test_rq():
    print('hi')

def get_sorted(UQ_df):
    ood_sorted = UQ_df.sort_values(by='pred_std').reset_index()
    ood_sorted['error_class'] = (ood_sorted.index//(len(ood_sorted)/10)).map(int)
    ood_sorted['error_class_str'] = ood_sorted['error_class'].map(str)
    ood_sorted['log_pred_std'] = np.log(ood_sorted['pred_std'])
    return ood_sorted


def plot_error_classes(ood_sorted):
    clist = [
        '#22ff00',
        '#70ee00',
        '#98db00',
        '#b2c800',
        '#c6b400',
        '#d89f00',
        '#e78600',
        '#f46900',
        '#fc4700',
        '#ff0000'
    ]
    cdict = {str(i) : clist[i] for i in range(10)}
    px.scatter(
        ood_sorted.iloc[-1::-1], 
        x = 'gt', 
        y = 'preds', 
        color = 'error_class_str',
        color_discrete_map= cdict
    )

def plot_mae_r2(ood_sorted, n = 10):
    sliceLen = len(ood_sorted)//n
    rs = []
    errors = []
    for i in range(n):
        df = ood_sorted.iloc[i * sliceLen : (i + 1)*sliceLen]
        x = df['gt']
        y = df['preds']
        r2 = pearsonr(x,y).statistic**2
        rs.append(r2)
        mae = np.mean(np.abs(x - y))
        errors.append(mae)
    plt.plot(rs, label = 'r2')
    plt.plot(errors, label = 'MAE')
    plt.legend()
    plt.show()

def compare_metrics(folders, metrics):
    outs = []
    x = []
    y = []
    for folder in folders:
        _, out = get_UQ(folder, metrics)
        outs.append(out)
        x.append(out[0])
        y.append(out[1])
    plt.scatter(x, y)
    plt.show()
    

def compare_metrics(folders, metrics):
    outs = []
    x = []
    y = []
    for folder in folders:
        _, out = get_UQ(folder, metrics)
        outs.append(out)
        x.append(out[0])
        y.append(out[1])
    plt.scatter(x, y)
    plt.show()

