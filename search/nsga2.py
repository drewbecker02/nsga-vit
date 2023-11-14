import numpy as np
import warnings

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling, IntegerRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.core.problem import Problem
from pymoo.core.population import Population



# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


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

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

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


class RankAndCrowdingSurvival(RankAndCrowding):
    
    def __init__(self, nds=None, crowding_func="cd"):
        warnings.warn(
                "RankAndCrowdingSurvival is deprecated and will be removed in version 0.8.*; use RankAndCrowding operator instead, which supports several and custom crowding diversity metrics.",
                DeprecationWarning, 2
            )
        super().__init__(nds, crowding_func)

# =========================================================================================================
# Implementation
# =========================================================================================================

class InitViT(IntegerRandomSampling):
    def __init__(self) -> None:
        """
        This abstract class represents any sampling strategy that can be used to create an initial population or
        an initial search point.
        """
        super().__init__()


    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        # Define the values for initialization
        # This is a vit for 5 blocks
        initial_values = [0, 1, 0, 1, 
                      9, 2, 0, 2,
                      11, 3, 0, 3,
                      9, 4, 0, 4,
                      11, 5,0, 5]
        X = np.tile(initial_values, (n_samples//2, 1))
        # X = np.full((n, n_samples), initial_values)
        # pop = Population.new(X=X)
        # mutation = PolynomialMutation(prob=0.02, eta=3, vtype=np.int32)
        # off = pop
        # for i in range(self.diversity):
        #     if i == 0:
        #         off = mutation(problem, pop)
        #     else:
        #         off = mutation(problem, off)
        # result = off.get("X")
        result = X
        return result

class NSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 # sampling=IntegerRandomSampling(),
                 sampling=InitViT(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=PointCrossover(n_points=2),
                 mutation=PolynomialMutation(eta=3, vtype=np.int32),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 init_vit=False,
                 **kwargs):
        if init_vit:
            self.sampling = InitViT()
        
        super().__init__(
            pop_size=pop_size,
            sampling=self.sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            init_vit=init_vit,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        if init_vit:
            self.sampling = InitViT()

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]
            
def main():
    
    n_var = int(4 * 5)
                        # * args.n_cells)
    lb = np.zeros(n_var)
    ub = np.ones(n_var)
    h = 1
    diversity = 200
    if True: #change lower bound from 0 to 1 for indices
        for i in range(1, len(lb)-1, 2):
            lb[i] = 1
    for b in range(0, n_var, 4):
        ub[b] =  12
        ub[b + 1] = h
        ub[b + 2] = 12
        ub[b + 3] = h
        h += 1
    n_samples = 20
    problem = Problem(n_var=n_var, xl=lb, xu=ub)    
    initial_values = [0, 1, 0, 1, 
                      9, 2, 0, 2,
                      11, 3, 0, 3,
                      9, 4, 0, 4,
                      11, 5,0, 5]
    X = np.full((20, n_samples), initial_values)
    pop = Population.new(X=X)
    mutation = PolynomialMutation(prob=0.02, eta=3, vtype=np.int32)
    for i in range(diversity):
        if i == 0:
            off = mutation(problem, pop)
        else:
            off = mutation(problem, off)
    Xp = off.get("X")

    print(Xp)
    
if __name__ == "__main__":
    main()

parse_doc_string(NSGA2.__init__)
