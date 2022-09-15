'''
Initialize random search
'''

from ..IO import io_stat, pkl_data
# from ..IO import read_input as rin
from ..IO.rin_class import Rin


def initialize(cryspy_in, stat):
    rin = Rin(cryspy_in)
    # ---------- initialize
    id_queueing = [i for i in range(rin.tot_struc)]
    id_running = []

    # ---------- status
    io_stat.set_id(stat, 'id_queueing', id_queueing)
    io_stat.write_stat(stat)

    # ---------- save
    rs_id_data = (id_queueing, id_running)
    pkl_data.save_rs_id(rs_id_data)

    return stat, rs_id_data
