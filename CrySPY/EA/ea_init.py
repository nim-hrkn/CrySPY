'''
Initialize evolutionary algorithm
'''

import pandas as pd

from ..IO import out_results
from ..IO import io_stat, pkl_data
# from ..IO import read_input as rin
from ..common import aiida_major_version


def initialize(rin, stat, rslt_data):
    tot_struc = int(stat["basic"]["tot_struc"])
    # ---------- log
    print('\n# ---------- Initialize evolutionary algorithm')
    print('# ------ Generation 1')
    print('{} structures by random\n'.format(tot_struc))
    with open('cryspy.out', 'a') as fout:
        fout.write('\n# ---------- Initilalize evolutionary algorithm\n')
        fout.write('# ------ Generation 1\n')
        fout.write('{} structures by random\n\n'.format(tot_struc))

    # ---------- initialize
    gen = 1
    id_queueing = [i for i in range(tot_struc)]
    id_running = []
    # ------ ea_info
    ea_info = pd.DataFrame(columns=['Gen', 'Population',
                                    'Crossover', 'Permutation', 'Strain',
                                    'Random', 'Elite',
                                    'crs_lat', 'slct_func'])
    ea_info.iloc[:, 0:7] = ea_info.iloc[:, 0:7].astype(int)
    tmp_info = pd.Series([1, tot_struc, 0, 0, 0, tot_struc, 0,
                          rin.crs_lat, rin.slct_func],
                         index=ea_info.columns)
    ea_info = ea_info.append(tmp_info, ignore_index=True)
    out_results.out_ea_info(ea_info)
    # ------ ea_origin
    ea_origin = pd.DataFrame(columns=['Gen', 'Struc_ID',
                                      'Operation', 'Parent'])
    ea_origin.iloc[:, 0:2] = ea_origin.iloc[:, 0:2].astype(int)
    for cid in range(tot_struc):
        tmp_origin = pd.Series([1, cid, 'random', None],
                               index=ea_origin.columns)
        ea_origin = ea_origin.append(tmp_origin, ignore_index=True)
    # ------ elite
    elite_struc = None
    elite_fitness = None
    # ------ rslt_data
    rslt_data['Gen'] = pd.Series(dtype=int)
    rslt_data = rslt_data[['Gen', 'Spg_num',
                           'Spg_sym', 'Spg_num_opt',
                           'Spg_sym_opt', 'E_eV_atom', 'Magmom', 'Opt']]

    # ---------- save
    ea_id_data = (gen, id_queueing, id_running)
    pkl_data.save_ea_id(ea_id_data)
    ea_data = (elite_struc, elite_fitness, ea_info, ea_origin)
    pkl_data.save_ea_data(ea_data)
    pkl_data.save_rslt(rslt_data)

    # ---------- status
    io_stat.set_common(stat, 'generation', gen)
    io_stat.set_id(stat, 'id_queueing', id_queueing)
    io_stat.write_stat(stat)

    if aiida_major_version >= 1:
        return stat, ea_id_data, ea_data, rslt_data
