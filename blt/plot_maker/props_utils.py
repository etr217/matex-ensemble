
import os
import datetime

def get_prop_labels(prop):
    if prop == 'bulk_modulus_vrh' or prop == 'bulk_modulus': 
        x_label = 'Bulk Modulus (GPa)'
        title = 'Bulk Modulus'
    elif prop == 'shear_modulus_vrh' or prop == 'shear_modulus': 
        x_label = 'Log Shear Modulus (GPa)'
        title = 'Log Shear Modulus'
    elif prop == 'debye_temperature': 
        x_label = 'Log Debye T(K)'
        title = 'Debye Temperature'
    elif prop == 'thermal_conductivity_300K': 
        x_label = r'Log Thermal Conductivity $\left(\frac{W}{m \cdot K}\right)$'
        title = 'Log Thermal Conductivity'
    elif prop == 'thermal_expansion_300K': 
        x_label = r'Log Thermal Expansion $(K^{-1})$'
        title = 'Log Thermal Expansion'
    elif prop == 'Egap' or prop == 'band_gap': 
        x_label = 'Band Gap (eV)'
        title = 'Band Gap'
    elif prop == 'yield_strength':
        x_label = 'Yield Strength (MPa)'
        title = 'Yield Strength'
    elif prop == 'elastic_anisotropy':
        x_label = 'Elastic anisotropy'
        title = 'Elastic anisotropy'
    elif prop =='delaney':
        x_label = 'ESOL'
        title = 'ESOL'
    elif prop =='freesolv':
        x_label = 'Freesolv (kJ/mol)'
        title = 'Freesolv'
    elif prop =='lipo':
        x_label = 'Lipophilicity'
        title = 'Lipophilicity'
    elif prop =='bace':
        x_label = 'BACE Binding Affinity'
        title = 'BACE Binding Affinity'
    elif prop =='refractive_ind':
        x_label = 'Refractive Index'
        title = 'Refractive Index'

    return x_label, title

def get_prop_rep(benchmark):
    if benchmark == 'matbench':
        rep = 'magpie'
    elif benchmark == 'mp' or benchmark == 'aflow':
        rep = 'oliynyk'
    elif benchmark == 'molnet':
        rep = 'rdkit'
    return rep


def get_results_filename(prop, benchmark, model_type='bilinear'):
    """
    Generates a filename based on the dataset and property-specific hyperparameters.
    
    Args:
        benchmark (str): The dataset name (e.g., 'aflow', 'matbench', 'mp', 'molnet').
        prop (str): The property type.
    
    Returns:
        str: The formatted filename.
    """
    # Build the filename
    rep_type = get_prop_rep(benchmark)
    dist_type = 'subtraction'

    # Default hyperparameters
    hidden_layer_size = None
    hidden_depth = None
    embedding_dim = None
    batch_size = None
    
    if benchmark == "aflow":
        if prop == "bulk_modulus_vrh":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 4, 42, 512
        elif prop == "debye_temperature":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 3, 42, 256
        elif prop == "Egap":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 512, 3, 64, 256
        elif prop == "shear_modulus_vrh":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 3, 48, 256
        elif prop == "thermal_conductivity_300K":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 4, 42, 256
        elif prop == "thermal_expansion_300K":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 4, 48, 256
    
    elif benchmark == "matbench":
        if prop == "band_gap" or prop == "refractive_ind":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 512, 3, 64, 256
        elif prop == "yield_strength":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 3, 32, 64
    
    elif benchmark == "mp":
        if prop == "bulk_modulus" or prop == "elastic_anisotropy" or prop == "shear_modulus":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 512, 3, 64, 256
    
    elif benchmark == "molnet":
        if prop == "bace":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 3, 48, 256
        elif prop == "delaney":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 1024, 3, 64, 256
        elif prop == "freesolv":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 4, 64, 256
        elif prop == "lipo":
            hidden_layer_size, hidden_depth, embedding_dim, batch_size = 256, 3, 32, 256
    
    # Ensure that valid hyperparameters were assigned
    if None in (hidden_layer_size, hidden_depth, embedding_dim, batch_size):
        raise ValueError(f"Invalid dataset ({benchmark}) or property ({prop}) combination.")
    
    return f"{rep_type}_{dist_type}_{model_type}_hsize{hidden_layer_size}_hnum{hidden_depth}_esize{embedding_dim}_bsize{batch_size}"


def get_latest_datetime_dir(base_path):
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    latest_dir = max(dirs, key=lambda d: datetime.datetime.strptime(d[:17], '%y-%m-%d_%H-%M-%S'), default=None)
    return os.path.join(base_path, latest_dir) if latest_dir else None

