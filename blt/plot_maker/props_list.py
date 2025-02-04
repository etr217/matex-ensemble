
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