#!/usr/bin/env python3
'''
Main script
'''

import os

from CrySPY.interface import select_code
from CrySPY.job.ctrl_job import Ctrl_job
from CrySPY.IO import read_input as rin
from CrySPY.start import cryspy_init, cryspy_restart

from CrySPY.IO import pkl_data
from CrySPY.common import aiida_major_version

def main():
    # ---------- lock
    if os.path.isfile('lock_cryspy'):
        raise SystemExit('lock_cryspy file exists')
    else:
        with open('lock_cryspy', 'w') as f:
            pass    # create vacant file

    # ---------- initialize
    if not os.path.isfile('cryspy.stat'):
        if aiida_major_version>=1:
            init_struc_data, opt_struc_data, stat, rslt_data, ea_id_data, ea_data = cryspy_init.initialize()
        else:
            cryspy_init.initialize()
        os.remove('lock_cryspy')
        raise SystemExit()
    # ---------- restart
    else:
        init_struc_data = pkl_data.load_init_struc()
        stat, init_struc_data = cryspy_restart.restart(init_struc_data)

    # ---------- check point 1
    if rin.stop_chkpt == 1:
        print('Stop at check point 1')
        os.remove('lock_cryspy')
        raise SystemExit()

    # ---------- check calc files in ./calc_in
    select_code.check_calc_files()

    # ---------- mkdir work/fin
    os.makedirs('work/fin', exist_ok=True)

    # ---------- instantiate Ctrl_job class
    opt_struc_data = pkl_data.load_opt_struc()
    rslt_data = pkl_data.load_rslt()
    ea_id = pkl_data.load_ea_id()
    jobs = Ctrl_job(stat, init_struc_data, opt_struc_data, rslt_data, ea_id)

    # ---------- check job status
    tmp_running, tmp_queueing, job_stat, stage_stat = jobs.check_job()

    # ---------- handle job
    job_stat, work_path_dic = jobs.handle_job()

    # ---------- recheck for skip and done
    if jobs.id_queueing:
        cnt_recheck = 0
        while jobs.recheck:
            cnt_recheck += 1
            jobs.recheck = False    # True --> False
            print('\n\n recheck {}\n'.format(cnt_recheck))
            tmp_running, tmp_queueing, job_stat, stage_stat = jobs.check_job()
            job_stat, work_path_dic = jobs.handle_job()

    if not (jobs.id_queueing or jobs.id_running):
        # ---------- next selection or generation
        if rin.algo in ['BO', 'LAQA', 'EA']:
            ea_data = pkl_data.load_ea_data()
            stat, ea_id_data, ea_data, rslt_data = jobs.next_sg(ea_data)
        # ---------- for RS
        else:
            with open('cryspy.out', 'a') as fout:
                fout.write('\nDone all structures!\n')
                print('Done all structures!')

    # ---------- unlock
    os.remove('lock_cryspy')


if __name__ == '__main__':
    main()
