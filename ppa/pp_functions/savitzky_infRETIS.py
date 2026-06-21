from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as savgol
import glob
from collections import OrderedDict
    

def savitzky(opt_params, *args, optimize=True, return_cv_data=False):
    n_pieces = 16
    polyn = 2
    extendfactor = 1.5
    n_data_grid = 2000
    n_grid = int(n_data_grid*extendfactor)
    path_data = args[0]
    cv_data= args[1]
    polyn = 2
    n_start = int((n_grid-n_data_grid)/2)
    n_end = int(n_start + n_data_grid + 1 )
    if opt_params.sum() == 0:
        return 1
    else:
        opt_params /= np.abs(opt_params).sum()
        cvs_r, cvs_u = pd.DataFrame(), pd.DataFrame()

        cvs_r['comb_cv'] = cv_data.loc[:].dot(opt_params)
        cvs_u['comb_cv'] = cv_data.loc[:].dot(opt_params)
        txmin, txmax = cvs_r['comb_cv'].min(), cvs_r['comb_cv'].max()

        cvs_r.loc[path_data['reactive'], 'weight'] = path_data['weight']
        cvs_u.loc[~path_data['reactive'], 'weight'] = path_data['weight'] 
        if return_cv_data:
            return cvs_r, cvs_u
        cvs_r = cvs_r.dropna(subset=['weight'])
        cvs_u = cvs_u.dropna(subset=['weight'])

        cvs_r.sort_values(by=['comb_cv'], ascending=True, inplace=True)
        cvs_u.sort_values(by=['comb_cv'], ascending=True, inplace=True)

        cvs_r.loc[:,['weight']] = cvs_r.loc[:,['weight']].cumsum(axis=0)
        cvs_u.loc[:,['weight']] = cvs_u.loc[:,['weight']].cumsum(axis=0)

        cvs_r.drop_duplicates(subset='comb_cv', inplace=True, keep='last')
        cvs_u.drop_duplicates(subset='comb_cv', inplace=True, keep='last')
        # print('unique r: ', cvs_r.shape[0])
        # print('unique u: ', cvs_u.shape[0])

        cvs_r.loc[:,['weight']] = cvs_r.loc[:,['weight']].div(path_data['weight'].sum())
        cvs_u.loc[:,['weight']] = cvs_u.loc[:,['weight']].div(path_data['weight'].sum())
        # print('min max: ', txmin, txmax)
        # print('len r: ', cvs_r.shape[0], ' u: ', cvs_u.shape[0])
        txrange = txmax-txmin
        txmid = txmin+0.5*txrange
        extended_range = txrange*extendfactor
        extxmin = txmid-0.5*extended_range
        extxmax = txmid+0.5*extended_range
        extended_range = extxmax - extxmin
        dx=extended_range/n_grid
        new_idx = pd.Index(np.arange(extxmin, extxmax, dx))
        # new_idx = new_idx[n_start:n_end]
        # print(new_idx)
        # print('cvs_r', cvs_r)
        # print('cvs_u', cvs_u)
        r_linear = np.zeros(len(new_idx))
        u_linear = np.zeros(len(new_idx))
        # r_linear[-n_data_grid:] += np.interp(new_idx[-n_data_grid:], cvs_r['comb_cv'], cvs_r['weight'])
        # u_linear[-n_data_grid:] += np.interp(new_idx[-n_data_grid:], cvs_u['comb_cv'], cvs_u['weight'])
        r_linear[n_start:n_end] += np.interp(new_idx[n_start:n_end], cvs_r['comb_cv'], cvs_r['weight'])
        u_linear[n_start:n_end] += np.interp(new_idx[n_start:n_end], cvs_u['comb_cv'], cvs_u['weight'])
        if n_grid > n_data_grid:    
            r_linear[n_end:] += r_linear[n_end-1]
            u_linear[n_end:] += u_linear[n_end-1]
        # plt.plot(u_linear)
        # plt.plot(r_linear)
        # plt.show()
        P_val, U_val = cvs_r['weight'].max(), cvs_u['weight'].max()
        v_ratio = 1 / P_val 

        # find duplicates in CV and just keep the largest weight value per CV
        combined = np.vstack((new_idx, u_linear, r_linear)).T
        df = pd.DataFrame(combined, columns=["cvs", "u_int", "r_int"])

        # differentiate
        window_length=int(n_data_grid/n_pieces)
        if (window_length % 2) == 0: window_length += 1

        df = df.assign(r=lambda x: (savgol(x['r_int'], window_length, polyn, deriv=1,delta=dx)))
        df = df.assign(u=lambda x: (savgol(x['u_int'], window_length, polyn, deriv=1,delta=dx)))
        
        df[['u', 'r']] = df[['u', 'r']].apply(lambda x: round(x, 7))
        df[['u', 'r']] = df[['u', 'r']].replace(0, np.nan)
        df[['u', 'r']] = df[['u', 'r']].clip(lower=0)  # New replace negative values with NaN

        scale_r = P_val / np.nansum(df['r']*dx)         #scale factors to ensure proper normalization
        scale_u = U_val / np.nansum(df['u']*dx)
        df['r'] *= scale_r
        df['u'] *= scale_u

        # print('unre:',df['u'].lt(0).sum(), 're:',df['r'].lt(0).sum() )
        df = df.assign(overlap=lambda x: (v_ratio*x["u"]*x["r"]*dx/(x["u"]+x["r"])))
        df = df.assign(overlap_int=lambda g: (np.cumsum(g["overlap"])))
        neg_count = (df['overlap_int']-df['overlap_int'].shift()).lt(0).sum()
        # print(lamb_c, lamb_r)
        # print(1-df["overlap"].sum())
        S_val = df["overlap"].sum()
        T_val = 1 - S_val
   
        if optimize:
            return S_val
        else:
            return T_val, P_val, df


def savitzky_rare(
    opt_params,
    *args,
    optimize=True,
    return_cv_data=False,
    n_pieces=16,
    polyn=2,
    extendfactor=1.5,
    n_data_grid=2000,
):
    """
    Variant of savitzky() for very rare events.

    Changes compared to original:
      - reactive and unreactive CDFs are normalised separately to 1
      - no rounding to 7 decimals
      - only negative densities are set to NaN (positive tiny values are kept)
      - P_val is computed directly from weights
      - v_ratio is set to 1 (overlap scale no longer depends on P_val)
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter as savgol

    path_data = args[0]
    cv_data   = args[1]

    # --- handle trivial zero-vector case ---
    opt_params = np.asarray(opt_params, dtype=float)
    if np.allclose(opt_params, 0.0):
        if optimize:
            return 1.0
        else:
            empty = pd.DataFrame(
                columns=["cvs", "u_int", "r_int", "r", "u", "overlap", "overlap_int"]
            )
            return 1.0, 0.0, empty

    # normalise linear-combination weights
    opt_params /= np.abs(opt_params).sum()

    # --- build combined CV (same for r/u) ---
    comb_cv = cv_data.to_numpy(dtype=float) @ opt_params

    cvs_r = pd.DataFrame({"comb_cv": comb_cv})
    cvs_u = pd.DataFrame({"comb_cv": comb_cv})

    reactive_mask = path_data["reactive"].astype(bool).to_numpy()
    w_all         = path_data["weight"].to_numpy()

    cvs_r.loc[reactive_mask, "weight"] = w_all[reactive_mask]
    cvs_u.loc[~reactive_mask, "weight"] = w_all[~reactive_mask]

    # if user only wants raw CV data, return before smoothing
    if return_cv_data and not optimize:
        return cvs_r.copy(), cvs_u.copy()

    # drop NaNs / empty classes
    cvs_r = cvs_r.dropna(subset=["weight"])
    cvs_u = cvs_u.dropna(subset=["weight"])

    if len(cvs_r) == 0 or len(cvs_u) == 0:
        # no overlap possible
        total_w = w_all.sum()
        sum_r   = w_all[reactive_mask].sum()
        P_val   = (sum_r / total_w) if total_w > 0 else 0.0

        if optimize:
            return 1.0
        else:
            empty = pd.DataFrame(
                columns=["cvs", "u_int", "r_int", "r", "u", "overlap", "overlap_int"]
            )
            return 1.0, P_val, empty

    # sort and make cumulative weights
    cvs_r = cvs_r.sort_values(by="comb_cv")
    cvs_u = cvs_u.sort_values(by="comb_cv")

    cvs_r["weight"] = cvs_r["weight"].cumsum()
    cvs_u["weight"] = cvs_u["weight"].cumsum()

    cvs_r = cvs_r.drop_duplicates(subset="comb_cv", keep="last")
    cvs_u = cvs_u.drop_duplicates(subset="comb_cv", keep="last")

    # --- crossing probability from raw weights (as in original spirit) ---
    total_w = w_all.sum()
    sum_r   = w_all[reactive_mask].sum()
    P_val   = (sum_r / total_w) if total_w > 0 else 0.0

    # --- normalise EACH CDF separately to 1 ---
    if cvs_r["weight"].iloc[-1] > 0:
        cvs_r["weight"] /= cvs_r["weight"].iloc[-1]
    if cvs_u["weight"].iloc[-1] > 0:
        cvs_u["weight"] /= cvs_u["weight"].iloc[-1]

    # --- grid setup (same as original) ---
    n_grid = int(n_data_grid * extendfactor)
    n_start = int((n_grid - n_data_grid) / 2)
    n_end   = int(n_start + n_data_grid + 1)

    txmin = min(cvs_r["comb_cv"].min(), cvs_u["comb_cv"].min())
    txmax = max(cvs_r["comb_cv"].max(), cvs_u["comb_cv"].max())
    txrange = txmax - txmin
    txmid   = txmin + 0.5 * txrange

    extended_range = txrange * extendfactor
    extxmin = txmid - 0.5 * extended_range
    extxmax = txmid + 0.5 * extended_range
    extended_range = extxmax - extxmin

    dx = extended_range / n_grid
    new_idx = pd.Index(np.arange(extxmin, extxmax, dx))

    r_linear = np.zeros(len(new_idx))
    u_linear = np.zeros(len(new_idx))

    r_linear[n_start:n_end] = np.interp(
        new_idx[n_start:n_end],
        cvs_r["comb_cv"].to_numpy(),
        cvs_r["weight"].to_numpy(),
    )
    u_linear[n_start:n_end] = np.interp(
        new_idx[n_start:n_end],
        cvs_u["comb_cv"].to_numpy(),
        cvs_u["weight"].to_numpy(),
    )

    if n_grid > n_data_grid:
        r_linear[n_end:] = r_linear[n_end - 1]
        u_linear[n_end:] = u_linear[n_end - 1]

    combined = np.vstack((new_idx, u_linear, r_linear)).T
    df = pd.DataFrame(combined, columns=["cvs", "u_int", "r_int"])

    # --- Savitzky–Golay derivative ---
    window_length = int(n_data_grid / n_pieces)
    if (window_length % 2) == 0:
        window_length += 1

    df["r"] = savgol(df["r_int"], window_length, polyn, deriv=1, delta=dx)
    df["u"] = savgol(df["u_int"], window_length, polyn, deriv=1, delta=dx)

    # Only remove clearly unphysical NEGATIVE densities
    df.loc[df["r"] < 0, "r"] = np.nan
    df.loc[df["u"] < 0, "u"] = np.nan

    # --- overlap (v_ratio = 1 now) ---
    v_ratio = 1.0
    df["overlap"] = v_ratio * df["u"] * df["r"] * dx / (df["u"] + df["r"])
    df["overlap"] = df["overlap"].fillna(0.0)
    df["overlap_int"] = df["overlap"].cumsum()

    S_val = df["overlap"].sum()
    T_val = 1.0 - S_val

    if optimize:
        return S_val
    else:
        return T_val, P_val, df


def extract_check_col(file_loc, lamb_c, lamb_r, cv_cols, atom_cols, centers):
    raw_data = pd.DataFrame()
    ens_data = pd.DataFrame()
    path_labels = ['idx', 'weight', 'lambda_max']
    mat_labels, atom_sel = get_mat_labels(file_loc)
    c_files = glob.glob(file_loc+f"/crossings_pub/crossing_{lamb_c}_00*.feather")
    check_format = [is_feather(file) for file in c_files]
    if len(c_files) == 5 and all(check_format):
        for file in sorted(c_files):
            ensemble = file[-9]
            ens_data = pd.read_feather(file)
            ens_data.drop(columns=ens_data.columns.difference(path_labels+[*cv_cols]), inplace=True)
            if len(atom_cols) > 0:
                mat_vals = pd.DataFrame(columns=['idx'] , index=ens_data.index) 

                for center in list(OrderedDict.fromkeys(centers)):
                    check_atoms =  np.full(mat_labels.shape, False)
                    atoms = [atom_cols[i] for i in range(len(centers)) if centers[i] == center]
                    atoms_names = [atom + '@' + center for atom in atoms]


                    for atom in atoms:
                        # atom_sel |= np.char.find(mat_labels.astype(str), atom) >= 0  
                        check_atoms = np.logical_or(check_atoms, mat_labels == atom)
                    
                    for col in atoms_names:
                        mat_vals[col] = None

                    indices = ens_data['idx'].items()
                    for i, val in indices:
                        line = f'00{ensemble}/traj/traj-acc/{val}'
                        dist_mat = np.load(f'../../{line}/dist_mat/{center}_inv_{lamb_c}.npy')
                        dist_mat = dist_mat.flatten()
                        mat_vals.loc[i, 'idx'] = val
                        mat_vals.loc[i, atoms_names] = list(dist_mat[check_atoms] )
                    # ens_data = ens_data.filter(regex='|'.join([*path_labels, *cvars, *atoms]))                
                ens_data = pd.concat([ens_data.set_index('idx'), mat_vals.set_index('idx')], axis=1, join='inner')
            else:
                ens_data.drop(columns='idx', inplace=True)

            raw_data = pd.concat([raw_data, ens_data])
            # ens_data.drop(ens_data.index , inplace=True)
            raw_data = raw_data.reset_index(drop=True)
        lamb_r_val = 0.32 + 0.38 * float(lamb_r)/199
        raw_data['reactive'] = raw_data['lambda_max'] > lamb_r_val
        dr = ~raw_data.columns.isin(path_labels)
        dr[-1] = False
        raw_data.loc[:, dr]= raw_data.loc[:, dr].astype(float).div(raw_data.loc[:, dr].astype(float).abs().max())
    return raw_data, dr 



def get_name(atoms, factors=[]):
    import re
    cv_names = {"Ang-0": r'$\gamma_{g1}$', 
            "Ang-1": r'$\gamma_{g2}$',
            "Ang-2": r'$\gamma_{m1}$',
            "Ang-3": r'$\gamma_{m2}$',
             
            "Ang2-0": r'$\cos \alpha_{1}$', 
            "Ang2-1": r'$\cos \alpha_{2}$', 
            "Ang2-2": r'$\cos \beta_{1}$', 
            "Ang2-3": r'$\cos \beta_{2}$',

            "Dens-0": r'$\rho_{g0}$', 
            "Dens-1": r'$\rho_{g3}$', 
            "Dens-2": r'$\rho_{g1}$', 
            "Dens-3": r'$\rho_{m0}$', 
            "Dens-4": r'$\rho_{m3}$', 
            "Dens-5": r'$\rho_{m1}$', 
            "Dens2-0": r'$\rho_{g2}$', 
            "Dens2-1": r'$\rho_{g4}$', 
            "Dens2-2": r'$\rho_{m2}$', 
            "Dens2-3": r'$\rho_{m4}$', 
            
            "Weight2-0": r'$N_{i}$', 
            "Weight2-1": r'$N_{ip}$',
            "Weight2-2": r'$N_{B}$',
    }
    if factors:
        factors = [pf.strip('[]').split() for pf in factors][0]
        iterator = zip(atoms.split('_'), factors)
    else: 
        iterator = atoms.split('_')


    atom_cols = []
    for atom in iterator:
        if factors:
            atom, factor = atom
        if 'params' in atom:
            atom = str(atom).split('@')[0]
            atom = cv_names[atom]
        else: 
            atom = atom.replace('N0','Na').replace('C0', 'Cl').replace('-', '').replace('\'', '')
            atom = atom.replace('Na', r'Na$^+$').replace('Cl', r'Cl$^-$')
            atom, center = atom.split('@')
            if center[0] != atom[0]:
                # atom = atom + '@' + center
                atom = center + atom
            else:
                atom = atom

            index = re.findall(r'\d+', atom)
            for i in index:
                    atom = atom.replace(str(i), f'{str(int(i)+1)}')
        if factors:
            atom = str(np.round(float(factor),2)) + ' ' + atom
        atom_cols.append(atom)
    name = '\n'.join(atom_cols)
    return name


