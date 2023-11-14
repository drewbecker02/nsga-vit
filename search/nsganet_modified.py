
import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.core.individual import Individual
from pymoo.core.survival import Survival
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput as disp_multi_objective
from pymoo.util.dominator import Dominator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
# =========================================================================================================
# Implementation
# based on nsga2 from https://github.com/msu-coinlab/pymoo
# =========================================================================================================


class NSGANet(GeneticAlgorithm):

    def __init__(self, **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------
class RankAndCrowding(Survival):

    def __init__(self, nds=None, crowding_func="cd"):
        """
        A generalization of the NSGA-II survival operator that ranks individuals by dominance criteria
        and sorts the last front by some user-specified crowding metric. The default is NSGA-II's crowding distances
        although others might be more effective.
        For many-objective problems, try using 'mnn' or '2nn'.
        For Bi-objective problems, 'pcd' is very effective.
        Parameters
        ----------
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.
        crowding_func : str or callable, optional
            Crowding metric. Options are:
                - 'cd': crowding distances
                - 'pcd' or 'pruning-cd': improved pruning based on crowding distances
                - 'ce': crowding entropy
                - 'mnn': M-Neaest Neighbors
                - '2nn': 2-Neaest Neighbors
            If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)``
            in which F (n, m) and must return metrics in a (n,) array.
            The options 'pcd', 'cd', and 'ce' are recommended for two-objective problems, whereas 'mnn' and '2nn' for many objective.
            When using 'pcd', 'mnn', or '2nn', individuals are already eliminated in a 'single' manner. 
            Due to Cython implementation, they are as fast as the corresponding 'cd', 'mnn-fast', or '2nn-fast', 
            although they can singnificantly improve diversity of solutions.
            Defaults to 'cd'.
        """

        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)
        print("F: ", F)
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )
                print("Front:", front)
                print("Crowding of front:", crowding_of_front)
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]

def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")
        print('Crowding distance:', cd_a, cd_b)
        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='larger_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------

# =========================================================================================================
# Interface
# =========================================================================================================


def nsganet(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=PointCrossover(n_points=2),
        mutation=PolynomialMutation(eta=3, vtype=np.int32),
        eliminate_duplicates=True,
        n_offsprings=None,
        **kwargs):
    """

    Parameters
    ----------
    pop_size : {pop_size}
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}
    n_offsprings : {n_offsprings}

    Returns
    -------
    nsganet : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGANet algorithm object.


    """

    return NSGANet(pop_size=pop_size,
                   sampling=sampling,
                   selection=selection,
                   crossover=crossover,
                   mutation=mutation,
                   survival=RankAndCrowding(crowding_func="cd"),
                   eliminate_duplicates=eliminate_duplicates,
                   n_offsprings=n_offsprings,
                   advance_after_initial_infill=True,
                   **kwargs)


parse_doc_string(nsganet)
