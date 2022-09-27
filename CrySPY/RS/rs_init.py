'''
Initialize random search
'''

from ..IO import io_stat, pkl_data
# from ..IO import read_input as rin


def initialize(rin):
    # ---------- initialize
    id_queueing = [i for i in range(rin.tot_struc)]
    id_running = []

    # ---------- save
    rs_id_data = (id_queueing, id_running)
    pkl_data.save_rs_id(rs_id_data)

    return rs_id_data
