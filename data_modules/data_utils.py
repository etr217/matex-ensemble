import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from pymatgen.core import Composition, Structure
import collections
import re
from typing import Tuple
import json
# import matex.blt.configs.config as config

from modnet.preprocessing import MODData
DATA_DIR = 'blt/data'


class CompositionError(Exception):
    """Exception class for composition errors"""
    pass

def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    # compile regex for speedup
    regex = r"([A-Z][a-z]*)\s*([-*\.\d]*)"
    r = re.compile(regex)
    for m in re.finditer(r, f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError(f'{f} is an invalid formula!')
    return sym_dict


def parse_formula(formula):
    '''
    Parameters
    ----------
        formula: str
            A string formula, e.g. Fe2O3, Li3Fe2(PO4)3.
    Return
    ----------
        sym_dict: dict
            A dictionary recording the composition of that formula.
    Notes
    ----------
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    '''
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace('@', '')
    formula = formula.replace('[', '(')
    formula = formula.replace(']', ')')
    # compile regex for speedup
    regex = r"\(([^\(\)]+)\)\s*([\.\d]*)"
    r = re.compile(regex)
    m = re.search(r, formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    sym_dict = get_sym_dict(formula, 1)
    return sym_dict


def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {key: elamt[key] / natoms for key in elamt}
    return comp_frac


def _fractional_composition_L(formula):
    comp_frac = _fractional_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    return elamt


def _element_composition_L(formula):
    comp_frac = _element_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def get_cbfv(path, pool_feat, elem_prop='mat2vec', scale=False):
    """
    Loads the compound csv file and featurizes it, then scales the features
    using StandardScaler.

    Parameters
    ----------
    path : str
        DESCRIPTION.
    elem_prop : str, optional
        DESCRIPTION. The default is 'oliynyk'.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    if 'formula' not in df.columns.values.tolist():
        df['formula'] = df['cif_id'].str.split('_ICSD').str[0]
    # elem_prop = 'mat2vec'
    # elem_prop = 'oliynyk'
    mini = False
    # mini = True
    X, y, formula, skipped = generate_features(df, elem_prop=elem_prop, pool_feat=pool_feat, mini=mini)
    if scale:
        # scale each column of data to have a mean of 0 and a variance of 1
        scaler = StandardScaler()
        # normalize each row in the data
        normalizer = Normalizer()

        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(normalizer.fit_transform(X_scaled),
                                columns=X.columns.values,
                                index=X.index.values)

        return np.array(X_scaled), np.array(y), np.array(formula), skipped
    else:
        return X, y, formula, skipped
    
def _assign_features(matrices, elem_info, formulae, pool_feat):
    formula_mat, count_mat, frac_mat, elem_mat, target_mat = matrices
    elem_symbols, elem_index, elem_missing = elem_info

    sum_feats = []
    avg_feats = []
    range_feats = []
    # var_feats = []
    dev_feats = []
    max_feats = []
    min_feats = []
    mode_feats = []
    targets = []
    formulas = []
    skipped_formula = []

    for h in tqdm(range(len(formulae)), desc='Assigning Features...'):
        elem_list = formula_mat[h]
        target = target_mat[h]
        formula = formulae[h]
        comp_mat = np.zeros(shape=(len(elem_list), elem_mat.shape[-1]))
        skipped = False

        for i, elem in enumerate(elem_list):
            if elem in elem_missing:
                skipped = True
            else:
                row = elem_index[elem_symbols.index(elem)]
                comp_mat[i, :] = elem_mat[row]

        if skipped:
            skipped_formula.append(formula)

        range_feats.append(np.ptp(comp_mat, axis=0))
        # var_feats.append(comp_mat.var(axis=0))
        max_feats.append(comp_mat.max(axis=0))
        min_feats.append(comp_mat.min(axis=0))

        comp_frac_mat = comp_mat.T * frac_mat[h]
        comp_frac_mat = comp_frac_mat.T
        avg_feats.append(comp_frac_mat.sum(axis=0))

        dev = np.abs(comp_mat - comp_frac_mat.sum(axis=0))
        dev = dev.T * frac_mat[h]
        dev = dev.T.sum(axis=0)
        dev_feats.append(dev)

        prominant = np.isclose(frac_mat[h], max(frac_mat[h]))
        mode = comp_mat[prominant].min(axis=0)
        mode_feats.append(mode)

        comp_sum_mat = comp_mat.T * count_mat[h]
        comp_sum_mat = comp_sum_mat.T
        sum_feats.append(comp_sum_mat.sum(axis=0))

        targets.append(target)
        formulas.append(formula)

    if len(skipped_formula) > 0:
        print('\nNOTE: Your data contains formula with exotic elements.',
              'These were skipped.')
    if pool_feat=='avg':
        conc_list = [avg_feats]
    elif pool_feat=='sum':
        conc_list = [sum_feats]
    elif pool_feat=='min':
        conc_list = [min_feats]
    elif pool_feat=='max':
        conc_list = [max_feats]
    else:
        # conc_list = [avg_feats, dev_feats,
        #              range_feats, max_feats, min_feats, mode_feats]
        print('non implemented method of pooling')
        
    
    feats = np.concatenate(conc_list, axis=1)

    return feats, targets, formulas, skipped_formula


def generate_features(df, 
                      elem_prop='oliynyk',
                      drop_duplicates=False,
                      extend_features=False,
                      pool_feat='avg',
                      mini=False):
    '''
    Parameters
    ----------
    df: Pandas.DataFrame()
        X column dataframe of form:
            df.columns.values = array(['formula', 'target',
                                       'extended1', 'extended2', ...],
                                      dtype=object)
    elem_prop: str
        valid element properties:
            'oliynyk',
            'jarvis',
            'magpie',
            'mat2vec',
            'onehot',
            'random_200'
    drop_duplicates: boolean
        Decide to keep or drop duplicate compositions
    extend_features: boolean
        Decide whether to use non ["formula", "target"] columns as additional
        features.
    pool_feat: choose between: 'avg', 'sum' ,'min', 'max'.
        how to pool atomic features to create the material representation
    Return
    ----------
    X: pd.DataFrame()
        Feature Matrix with NaN values filled using the median feature value
        for dataset
    y: pd.Series()
        Target values
    formulae: pd.Series()
        Formula associated with X and y
    '''
    if drop_duplicates:
        if df['formula'].value_counts()[0] > 1:
            df.drop_duplicates('formula', inplace=True)
            print('Duplicate formula(e) removed using default pandas function')

    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    
    cbfv_path = os.path.join('blt/data/element_properties', elem_prop+'.csv')

    elem_props = pd.read_csv(cbfv_path)

    elem_props.index = elem_props['element'].values
    elem_props.drop(['element'], inplace=True, axis=1)

    elem_symbols = elem_props.index.tolist()
    elem_index = np.arange(0, elem_props.shape[0], 1)
    elem_missing = list(set(all_symbols) - set(elem_symbols))

    elem_props_columns = elem_props.columns.values
    # column_names = np.concatenate(['avg_' + elem_props_columns,
    #                                'dev_' + elem_props_columns,
    #                                'range_' + elem_props_columns,
    #                                'max_' + elem_props_columns,
    #                                'min_' + elem_props_columns,
    #                                'mode_' + elem_props_columns])
    if pool_feat=='sum':
        column_names = np.array(['sum_' + elem_props_columns]).squeeze()
    elif pool_feat=='avg':
        column_names = np.array(['avg_' + elem_props_columns]).squeeze()
    elif pool_feat=='min':
        column_names = np.array(['min_' + elem_props_columns]).squeeze()
    elif pool_feat=='max':
        column_names = np.array(['max_' + elem_props_columns]).squeeze()
    else:
        print('non implemented method of pooling')
        exit()

    # make empty list where we will store the property value
    targets = []
    # store formula
    formulae = []
    # add the values to the list using a for loop

    elem_mat = elem_props.values

    formula_mat = []
    count_mat = []
    frac_mat = []
    target_mat = []

    if extend_features:
        features = df.columns.values.tolist()
        features.remove('target')
        extra_features = df[features]

    for index in tqdm(df.index.values, desc='Processing Input Data'):
        formula, target = df.loc[index, 'formula'], df.loc[index, 'target']
        if 'x' in formula:
            continue
        l1, l2 = _element_composition_L(formula)
        formula_mat.append(l1)
        count_mat.append(l2)
        _, l3 = _fractional_composition_L(formula)
        frac_mat.append(l3)
        target_mat.append(target)
        formulae.append(formula)

    print('\tfeaturizing compositions...'.title())

    matrices = [formula_mat, count_mat, frac_mat, elem_mat, target_mat]
    elem_info = [elem_symbols, elem_index, elem_missing]
    feats, targets, formulae, skipped = _assign_features(matrices,
                                                         elem_info,
                                                         formulae,
                                                         pool_feat)

    print('\tcreating pandas objects...'.title())

    # split feature vectors and targets as X and y
    X = pd.DataFrame(feats, columns=column_names, index=formulae)
    y = pd.Series(targets, index=formulae, name='target')
    formulae = pd.Series(formulae, index=formulae, name='formula')
    if extend_features:
        extended = pd.DataFrame(extra_features, columns=features)
        extended = extended.set_index('formula', drop=True)
        X = pd.concat([X, extended], axis=1)

    # reset dataframe indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    formulae.reset_index(drop=True, inplace=True)

    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of NaN values.
    X.dropna(inplace=True, how='all')
    y = y.iloc[X.index]
    formulae = formulae.iloc[X.index]

    # get the column names
    cols = X.columns.values
    # find the median value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the column's median value
    X[cols] = X[cols].fillna(median_values)

    # Only return the avg/sum of element properties.

    if mini:
        np.random.seed(42)
        booleans = np.random.rand(X.shape[-1]) <= 64/X.shape[-1]
        X = X.iloc[:, booleans]

    return X, y, formulae, skipped


def ood_split(self, ratio: float = 0.05) -> Tuple[MODData, MODData, MODData]:
        """Create three new MODData, one for training, one for in distribution evaluation, and one for 
        out of distribution (ood) evaluation.

        Arguments:
            ratio (float): Would be used to distinguish top % samples as ood set. Would also define % for
            in distribution evaluation set.

        Returns:
            The training MODData, in distribution MODData and the ood MODData as a tuple.

        """
        # Calculate the number of samples for ood and in-distribution evaluation sets
        num_samples = len(self.df_targets)
        num_ood_samples = int(ratio * num_samples)
        num_ind_samples = num_ood_samples

        # Sort indices by target values in descending order to find ood samples
        sorted_indices = np.argsort(self.targets[:, 0])[::-1]  # assuming single target
        ood_indices = sorted_indices[:num_ood_samples]
        remaining_indices = sorted_indices[num_ood_samples:]

        # Sample indices for in-distribution evaluation
        ind_indices = np.random.choice(remaining_indices, num_ind_samples, replace=False)
        train_indices = list(set(remaining_indices) - set(ind_indices))

        # Create MODData objects for each subset
        train_moddata = self.from_indices(train_indices)
        ind_moddata = self.from_indices(ind_indices)
        ood_moddata = self.from_indices(ood_indices)

        return train_moddata, ind_moddata, ood_moddata


def save_dataset_as_csv(dataset, filename, dataset_name, property):
    dir_path = os.path.join(DATA_DIR, dataset_name, property)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df = pd.DataFrame({
        'formula': dataset.ids,
        'target': dataset.y.ravel()
    })
    df.to_csv(os.path.join(dir_path, filename), index=False)
    df['rep'] = list(dataset.X)
    return df

def extract_composition(container):
    """extracts composition from a container."""
    return container.composition

def get_comp_from_struc(structure_dict):
    """gets composition from structure dictionary."""
    structure = Structure.from_dict(structure_dict)
    return structure.composition

def load_data(path, property_name):
    """Loads data from a JSON file and converts it into a DataFrame."""
    with open(os.path.join(path, f'{property_name}.json')) as data_file:
        data = json.load(data_file)
    return pd.DataFrame(data["data"], columns=['composition', property_name])

def preprocess_composition(df, property_name):
    """Processes the composition column if necessary."""
    if property_name == 'refractive_ind':
        df['composition'] = df['composition'].apply(get_comp_from_struc)
    df["composition"] = df["composition"].map(Composition)
    return df

def create_moddata(df, property_name, ood_ratio):
    """Creates MODData object, performs feature selection, and splits into sets."""
    MODData.ood_split = ood_split
    mod_data = MODData(
        materials=df["composition"], 
        targets=df[property_name], 
        target_names=[property_name]
    )
    mod_data.featurize()
    train_moddata, eval_moddata, ood_moddata = mod_data.ood_split(ood_ratio)
    train_moddata.feature_selection()
    return train_moddata, eval_moddata, ood_moddata

def get_featurized_sets(train_moddata, eval_moddata, ood_moddata):
    """Returns featurized and optimized datasets."""
    optimal_descriptors = train_moddata.get_optimal_descriptors()
    train_set = train_moddata.get_optimal_df()
    eval_set = eval_moddata.get_featurized_df()[optimal_descriptors].join(eval_moddata.get_target_df())
    ood_set = ood_moddata.get_featurized_df()[optimal_descriptors].join(ood_moddata.get_target_df())

    return train_set, eval_set, ood_set

def handle_nan_values(train_set, eval_set, ood_set, nan_strategy, dataset_name):
    """Handles NaN values based on the chosen strategy."""
    all_sets = [train_set, eval_set, ood_set]
    nan_features = {col for df in all_sets for col in df.columns[df.isna().any()]}

    if nan_strategy == 'drop_feat':
        for df in all_sets:
            df.drop(columns=nan_features, inplace=True)

    elif nan_strategy == 'drop_sample':
        if dataset_name == 'molnet':
            for i, df in enumerate(all_sets):
                has_nan_in_rep = df['rep'].apply(lambda x: np.isnan(x).any())
                all_sets[i] = df[~has_nan_in_rep].reset_index(drop=True)
            return all_sets[0], all_sets[1], all_sets[2]
        else:
            df.dropna(inplace=True)
    
    elif nan_strategy == 'const':
        for df in all_sets:
            df.fillna(value=-1, inplace=True)


def handle_zero_targets(train_set, eval_set, ood_set, property_name):
    """Removes rows with zero target values if the target property is band_gap."""
    for df in [train_set, eval_set, ood_set]:
        df.drop(df[df[property_name] == 0].index, inplace=True)

def extract_and_save_composition(train_moddata, eval_moddata, ood_moddata, train_set, eval_set, ood_set, path):
    """Extracts composition data and saves featurized sets to CSV."""
    updated_sets = []
    for set_name, moddata, df_set in zip(
        ['train', 'eval', 'ood'], [train_moddata, eval_moddata, ood_moddata], [train_set, eval_set, ood_set]
    ):
        df_comp = moddata.get_structure_df()
        df_comp['composition'] = df_comp['structure'].apply(extract_composition)
        df_comp.drop(columns=['structure'], inplace=True)
        df_set = df_set.join(df_comp)
        df_set.to_csv(os.path.join(path, f'{set_name}_featurized.csv'), index=False)
        updated_sets.append(df_set)
        
    return updated_sets[0], updated_sets[1], updated_sets[2]

def normalize_data(train_set, eval_set, ood_set, property_name, scaler_type):
    """Normalizes dataset features using MinMax or Standard scaler."""
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()

    feature_columns = train_set.drop(columns=[property_name, 'composition']).columns
    train_X = scaler.fit_transform(train_set[feature_columns])
    eval_X = scaler.transform(eval_set[feature_columns])
    ood_X = scaler.transform(ood_set[feature_columns])

    def create_normalized_df(X, df):
        return pd.concat([pd.DataFrame(X, columns=feature_columns), df[[property_name, 'composition']].reset_index(drop=True)], axis=1)

    return create_normalized_df(train_X, train_set), create_normalized_df(eval_X, eval_set), create_normalized_df(ood_X, ood_set)



