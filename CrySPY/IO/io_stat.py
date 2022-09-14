'''
I/O for cryspy.stat
'''
import configparser

# from . import read_input as rin
from ..IO.rin_class import Rin


def stat_init(cryspy_in='cryspy.in'):
    rin = Rin(cryspy_in)
    stat = configparser.ConfigParser()
    stat.add_section('basic')
    stat.add_section('structure')
    # ---------- algo
    if rin.algo == 'BO':
        stat.add_section('BO')
    if rin.algo == 'LAQA':
        stat.add_section('LAQA')
    if rin.algo == 'EA':
        stat.add_section('EA')
    # ---------- calc_code
    if rin.calc_code == 'VASP':
        stat.add_section('VASP')
    if rin.calc_code == 'QE':
        stat.add_section('QE')
    if rin.calc_code == 'soiap':
        stat.add_section('soiap')
    if rin.calc_code == 'LAMMPS':
        stat.add_section('LAMMPS')
    if rin.calc_code == 'OMX':
        stat.add_section('OMX')
    # ----------
    stat.add_section('option')
    stat.add_section('status')
    return stat


def stat_read():
    stat = configparser.ConfigParser()
    stat.read('cryspy.stat')
    return stat


def write_stat(stat):
    """delete 'None' and write them as 'cryspy.stat'

    Args:
        stat (configparser.ConfigParser): stat
    """
    if True:
        with open('cryspy.stat', 'w') as f:
            stat.write(f)
    else:
        from collections import OrderedDict
        newdict = OrderedDict()
        for name1 in stat._sections:
            # print(name1)
            name1dict = stat._sections[name1]
            dellist = []
            for name2 in name1dict:
                #print(name2, stat._sections[name1][name2])
                if stat._sections[name1][name2] == 'None':
                    dellist.append(name2)
            print("delete", name1, dellist)
            for name2 in dellist:
                name1dict.pop(name2)
            newdict[name1] = name1dict

        config = configparser.ConfigParser()
        config.read_dict(dictionary=newdict)

        with open('cryspy.stat', 'w') as f:
            config.write(f)


# ---------- input section
def set_input_common(stat, sec, var_str, var):
    stat.set(sec, var_str, '{}'.format(var))


# ---------- status section
def set_common(stat, var_str, var):
    stat.set('status', var_str, '{}'.format(var))


def set_id(stat, var_str, var_list):
    if len(var_list) > 30:
        stat.set('status', var_str, '{0} ... total {1} IDs'.format(
            ' '.join(str(a) for a in var_list[:5]), len(var_list)))
    else:
        stat.set('status', var_str, '{}'.format(
            ' '.join(str(a) for a in var_list)))


def set_stage(stat, current_id, current_stage):
    stat.set('status', 'ID {:>6}'.format(current_id), 'Stage {}'.format(
        current_stage))


def clean_id(stat, current_id):
    stat.remove_option('status', 'ID {:>6}'.format(current_id))
