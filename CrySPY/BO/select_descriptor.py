'''
Select descriptors
    - FP: FingerPrint
'''

from .. import utility
from ..calc_dscrpt.FP.calc_FP import Calc_FP
#from ..IO import read_input as rin
from ..IO.rin_class import Rin

def select_descriptor(cryspy_in, struc_data):
    rin = Rin(cryspy_in)
    # ---------- fingerprint
    if rin.dscrpt == 'FP':
        print('Calculate descriptors: FingerPrint')
        # ------ check cal_fingerprint executable file
        fppath = utility.check_fppath()
        # ------ calc fingerprint
        cfp = Calc_FP(struc_data, rin.fp_rmin, rin.fp_rmax,
                      rin.fp_npoints, rin.fp_sigma, fppath)
        cfp.calc()
        return cfp.descriptors
    else:
        raise NotImplementedError('Now FP only')
