'''
Select parents in evolutionary algorithm
'''

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher


class Select_parents:
    '''
    select parents

    # ---------- args
    struc_data (dict or list): structure data
        You may include None in struc_data
        if type of struc_data is list,
            struc_data is converted into dict type
            as {0: struc_data_0, 1: struc_data_1, ...}

    fitness (dict, list, or np.ndarray): fitness
        You may include None and np.nan in fitness
        if type of fitness is list or np.ndarray,
            fitness is converted into dict type
            as {0: fitness_0, 1: fitness_1, ...}

    elite_struc (dict or None): structure data of elite

    elite_fitness (dict or None): fitness of elite

    fit_reverse (bool): if you search minimum value,
                            set fit_reverse=False (default)
        False: smaller value is better
        True: larger value is better

    n_fittest (int): number of data which can survive
        if 0 (deault), all candidates can survive

    # ---------- instance methods
    self.set_tournament(self, t_size=3)

    self.set_roulette(self, a=2.0, b=1.0)

    self.get_parents(n_parent)
    '''

    def __init__(self, struc_data, fitness,
                 elite_struc=None, elite_fitness=None,
                 fit_reverse=False, n_fittest=0,
                 emax_ea=None, emin_ea=None):
        # ---------- check args
        # ------ data
        self.struc_data, self.fitness = self._check_data(struc_data,
                                                         fitness,
                                                         elite_struc,
                                                         elite_fitness)
        # ------ fit_reverse
        if not isinstance(fit_reverse, bool):
            raise TypeError('fit_reverse should be bool')
        else:
            self.fit_reverse = fit_reverse
        # ------ n_fittest
        if isinstance(n_fittest, int):
            if n_fittest < 0:
                raise ValueError('n_fittest must be zero or positive int')
            else:
                self.n_fittest = n_fittest
        else:
            raise TypeError('n_fittest must be int')
        # ------ None and np.nan --> inf or -inf for fitness
        for key, value in self.fitness.items():
            if value is None or np.isnan(value):
                self.fitness[key] = -np.inf if fit_reverse else np.inf
            # ---- emax_ea
            if emax_ea is not None:
                if value > emax_ea:
                    self.fitness[key] = -np.inf if fit_reverse else np.inf
                    print('Eliminate ID {}: {} > emax_ea'.format(
                          key, value))
            # ---- emin_ea
            if emin_ea is not None:
                if value < emin_ea:
                    self.fitness[key] = -np.inf if fit_reverse else np.inf
                    print('Eliminate ID {}: {} < emin_ea'.format(
                          key, value))
        # ---------- ranking of fitness: list of id
        self.ranking = sorted(self.fitness, key=self.fitness.get,
                              reverse=fit_reverse)
        #print('ranking', self.ranking)
        # ---------- remove duplicated structures and
        #                cut by survival of the fittest
        self._dedupe()    # get self.ranking_dedupe

    def _check_data(self, struc_data, fitness, elite_struc, elite_fitness):
        # ---------- struc_data and fitness
        if isinstance(struc_data, dict) and isinstance(fitness, dict):
            pass    # if dict, allow len(struc_data) != len(fitness)
        elif isinstance(struc_data, dict) and (
                isinstance(fitness, list) or isinstance(fitness, np.ndarray)):
            raise TypeError('struc_data is dict,'
                            ' so fitness should also be dict')
        elif isinstance(fitness, dict) and (
                isinstance(struc_data, list)
                or isinstance(struc_data, np.ndarray)):
            raise TypeError('fitness is dict,'
                            ' so struc_data should also be dict')
        elif isinstance(struc_data, list) and (
                isinstance(fitness, list) or isinstance(fitness, np.ndarray)):
            # ------ check number of data
            if not len(struc_data) == len(fitness):
                raise ValueError('not len(struc_data) == len(fitness)')
            # ------ convert
            struc_data = {i: struc_data[i] for i in range(len(struc_data))}
            fitness = {i: fitness[i] for i in range(len(fitness))}
        else:
            raise TypeError('Type of struc_data and fitness is wrong')
        # ---------- elite
        if isinstance(elite_struc, dict) and isinstance(elite_fitness, dict):
            # ------ check number of data
            if not len(elite_struc) == len(elite_fitness):
                raise ValueError('not len(elite_struc) == len(elite_fitness)')
            if None in elite_struc.values():
                raise ValueError('elite_struc includes None')
            if (None in elite_fitness.values()) or (
                    np.nan in elite_fitness.values()):
                raise ValueError('elite_fitness includes None or np.nan')
            # ------ add elite to data
            struc_data.update(elite_struc)
            fitness.update(elite_fitness)
        elif elite_struc is elite_fitness is None:
            pass
        else:
            raise TypeError('elite_struc and elite_fitness'
                            ' must be dict or None')
        # ---------- return
        return struc_data, fitness

    def _dedupe(self):
        # ---------- remove duplicated structure data
        # ------ initialize
        print("in dedupe, self.struc_data")
        print(self.struc_data)
        ncheck = 5
        self.ranking_dedupe = []
        smatcher = StructureMatcher()    # instantiate
        # ------ register not dupulicated data
        for i_id in self.ranking:
            # -- init dupl_flag
            dupl_flag = False
            # -- for structure is None
            if self.struc_data[i_id] is None:
                continue    # next i_id
            # -- duplication check
            for j_id in self.ranking_dedupe[:-(ncheck+1):-1]:
                if smatcher.fit(self.struc_data[i_id], self.struc_data[j_id]):
                    dupl_flag = True
                    break
            # -- register or skip
            if dupl_flag:
                continue    # next i_id
            else:
                self.ranking_dedupe.append(i_id)
                # n_fittest
                if self.n_fittest > 0:
                    if len(self.ranking_dedupe) == self.n_fittest:
                        break
        # ---------- log
        print('Remove duplicated data')
        if self.n_fittest > 0:
            print('Survival of the fittest: top {0} structures survive'.format(
                self.n_fittest))

    def set_tournament(self, t_size=3):
        '''
        setting for tournament selection

        # ---------- args
        t_size (int): tournament size
        '''
        # ---------- check args
        # ------ t_size
        if isinstance(t_size, int):
            if t_size < 2:
                raise ValueError('t_size must be more than or equal to 2')
        else:
            raise TypeError('t_size must be int')
        self.t_size = t_size
        # ---------- set self.get_parents()
        self.get_parents = self._tournament

    def _tournament(self, n_parent):
        '''
        tournament selection

        # ---------- args
        n_parent (int): number of parents

        # ---------- return
        parent_id (list): ID of selected parents
        '''
        parent_id = []
        while len(parent_id) < n_parent:
            t_indx = np.random.choice(len(self.ranking_dedupe), self.t_size,
                                      replace=False)
            if parent_id:    # not allow the same parent in crossover
                if parent_id[0] == self.ranking_dedupe[min(t_indx)]:
                    continue
            parent_id.append(self.ranking_dedupe[min(t_indx)])
        return parent_id

    def set_roulette(self, a=2.0, b=1.0):
        '''
        setting for roulette selection

        # ---------- args
        a (float): parameter for linear scaling, 0 < b < a
        b (float): parameter for linear scaling, 0 < b < a
        '''
        # ---------- check args
        if not 0 < b < a:
            raise ValueError('must be 0 < b < a')
        # ---------- calculate cumulative fitness
        fitness_dedupe = np.array(
            [self.fitness[i] for i in self.ranking_dedupe])
        fitness_dedupe = self._linear_scaling(fitness_dedupe, a, b)
        self.cum_fit = np.cumsum(fitness_dedupe/fitness_dedupe.sum())
        # ---------- set self.get_parents()
        self.get_parents = self._roulette

    def _roulette(self, n_parent):
        '''
        roulette selection

        # ---------- args
        n_parent (int): number of parents

        # ---------- return
        parent_id (list): ID of selected parents
        '''
        # ---------- select parents
        parent_id = []
        while len(parent_id) < n_parent:
            # indx_array: if all false, array is vacant
            indx_array = np.where(self.cum_fit < np.random.rand())[0]
            # select_indx: consider vacant or not
            select_indx = indx_array[-1] + 1 if indx_array.size != 0 else 0
            if parent_id:    # not allow same parent for crossover
                if parent_id[0] == self.ranking_dedupe[select_indx]:
                    continue
            parent_id.append(self.ranking_dedupe[select_indx])
        return parent_id

    def _linear_scaling(self, fitness, a, b):
        # ---------- check args
        # ------ fitness
        if isinstance(fitness, np.ndarray):
            pass
        elif isinstance(fitness, list):
            fitness = np.array(fitness)
        else:
            raise TypeError('fitness must be list or np.array')
        # ------ in case of int
        a = float(a)
        b = float(b)
        # ---------- for fit_reverse
        if not self.fit_reverse:
            fitness = -fitness
        # ---------- scaling
        fmax = float(fitness.max())
        fmin = float(fitness.min())
        # ------ in case the same values
        if fmax == fmin:
            return fitness
        fitness = (a - b)/(fmax - fmin)*fitness + (
            (b*fmax - a*fmin)/(fmax - fmin))
        # ---------- return
        return fitness
