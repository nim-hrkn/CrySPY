'''
Append structures by evolutionary algorithm
'''

import os

import pandas as pd

from .ea_child import child_gen
from ..gen_struc.EA.select_parents import Select_parents
from ..IO import out_results
from ..IO import change_input, io_stat, pkl_data
# from ..IO import read_input as rin
# from ..IO.rin_class import Rin
from ..common import aiida_major_version


def append_struc(rin, stat, init_struc_data, opt_struc_data, rslt_data):
    
    if aiida_major_version>=1:
        tot_struc = int(stat["basic"]["tot_struc"])
    else:
        tot_struc = rin.tot_struc
    # ---------- append structures by EA
    print('\n# ---------- Append structures by EA')
    with open('cryspy.out', 'a') as fout:
        fout.write('\n# ---------- Append structures by EA\n')

    # ---------- load data
    if aiida_major_version >= 1:
        if opt_struc_data is None:
            raise ValueError('opt_struc_data must not be None.')
        if rslt_data is None:
            raise ValueError('rslt_data must not be None.')
    else:
        opt_struc_data = pkl_data.load_opt_struc()
        rslt_data = pkl_data.load_rslt()

    # ---------- fitness
    fitness = rslt_data['E_eV_atom'].to_dict()    # {ID: energy, ..,}

    # ---------- instantiate Seclect_parents class
    print('# ------ select parents')
    sp = Select_parents(opt_struc_data, fitness, None, None,
                        rin.fit_reverse, rin.n_fittest,
                        rin.emax_ea, rin.emin_ea)
    if rin.slct_func == 'TNM':
        sp.set_tournament(t_size=rin.t_size)
    else:
        sp.set_roulette(a=rin.a_rlt, b=rin.b_rlt)

    # ---------- generate offspring by EA
    print('# ------ Generate structures')
    init_struc_data, eagen = child_gen(sp, init_struc_data)

    # ----------  ea_info
    if os.path.isfile('./data/pkl_data/EA_data.pkl'):
        _, _, ea_info, ea_origin = pkl_data.load_ea_data()
    else:
        # ------ initialize
        # -- ea_info
        ea_info = pd.DataFrame(columns=['Gen', 'Population',
                                        'Crossover', 'Permutation', 'Strain',
                                        'Random', 'Elite',
                                        'crs_lat', 'slct_func'])
        ea_info.iloc[:, 0:7] = ea_info.iloc[:, 0:7].astype(int)
        # -- ea_origin
        ea_origin = pd.DataFrame(columns=['Gen', 'Struc_ID',
                                          'Operation', 'Parent'])
        ea_origin.iloc[:, 0:2] = ea_origin.iloc[:, 0:2].astype(int)
    # ------ register ea_info
    tmp_info = pd.Series([tot_struc, rin.n_pop, rin.n_crsov,
                          rin.n_perm, rin.n_strain, rin.n_rand, 0,
                          rin.crs_lat, rin.slct_func],
                         index=ea_info.columns)
    ea_info = ea_info.append(tmp_info, ignore_index=True)
    # ------ out ea_info
    out_results.out_ea_info(ea_info)

    # ---------- ea_origin
    # ------ EA operation part
    for cid in range(tot_struc, tot_struc + rin.n_pop - rin.n_rand):
        tmp_origin = pd.Series([tot_struc, cid, eagen.operation[cid],
                                eagen.parents[cid]], index=ea_origin.columns)
        ea_origin = ea_origin.append(tmp_origin, ignore_index=True)
    # ------ random part
    for cid in range(tot_struc + rin.n_pop - rin.n_rand,
                     tot_struc + rin.n_pop):
        tmp_origin = pd.Series([tot_struc, cid, 'random', None],
                               index=ea_origin.columns)
        ea_origin = ea_origin.append(tmp_origin, ignore_index=True)
    # ------  out ea_origin
    out_results.out_ea_origin(ea_origin)

    # ---------- save ea_data
    ea_data = (None, None, ea_info, ea_origin)
    pkl_data.save_ea_data(ea_data)

    # ---------- change variables in cryspy.in
    config = change_input.config_read()
    print('# -- Changed cryspy.in')
    # ------ tot_struc
    change_input.change_basic(config, 'tot_struc', tot_struc + rin.n_pop)
    rin.tot_struc = tot_struc + rin.n_pop
    print('Changed tot_struc in cryspy.in from {} to {}'.format(
          tot_struc, tot_struc + rin.n_pop))
    tot_struc = tot_struc + rin.n_pop
    # ------ append_struc_ea: True --> False
    change_input.change_option(config, 'append_struc_ea', False)
    change_input.change_option(rin, 'append_struc_ea', False)
    print('Changed append_struc_ea in cryspy.in from {} to {}'.format(
          True, False))
    # ------ write
    # change_input.write_config(config)
    

    # ---------- status
    io_stat.set_input_common(stat, 'basic', 'tot_struc', tot_struc)
    io_stat.set_input_common(stat, 'option', 'append_struc_ea', False)
    io_stat.write_stat(stat)

    # ---------- return
    return init_struc_data, rin, stat, ea_data

