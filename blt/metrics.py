import numpy as np
from scipy.stats import pearsonr, spearmanr

def spearman(df = None):
    if df is None:
        return 'spearman'
    else:
        return spearmanr(df['total_abs_error'], df['pred_std']).statistic
    
def MAE(df = None):
    if df is None:
        return 'MAE'
    else:
        return np.abs(df['preds'] - df['gt']).mean()
    
def MSE(df = None):
    if df is None:
        return 'MSE'
    else:
        return ((df['preds'] - df['gt'])**2).mean()
    
def r2(df = None):
    if df is None:
        return 'R2'
    else:
        return 1 - (
            MSE(df)/df['gt'].var()
        )
    
def r(df = None):
    if df is None:
        return 'r'
    else:
        return pearsonr(df['gt'], df['preds']).statistic
    
def nll(df = None):
    if df is None:
        return 'NLL'
    else:
        return (
            np.log( 2 * np.pi) + 
            np.log( df['pred_std']**2 ) + 
            ( (df['total_abs_error']**2) / (df['pred_std']**2) )
        ).mean()/ 2