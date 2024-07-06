import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.soo.nonconvex.ga import comp_by_cv_and_fitness
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import replications_of_sim as ros
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MyProblem(ElementwiseProblem):
    def __init__(self, T, product_size, item_size, upper_bound):
        super().__init__(n_var=T*item_size, n_obj=1, n_constr=0, xl=np.zeros(T*item_size), xu=np.ones(T*item_size) * upper_bound)
        self.parameters=[T, product_size, item_size]
        self.count = 0
        self.fitness_list = []

    def _evaluate(self, x, out, *args, **kwargs):
        # print(">> Case %d" %(self.count+1))
        self.count += 1
        T, product_size, item_size = self.parameters[0], self.parameters[1], self.parameters[2]
        f1 = ros.replications_of_sim(T, product_size, item_size, x.reshape(T,item_size).astype('int'))
        self.fitness_list.append(f1)
        out["F"] = [f1]

def ga_fun(T, product_size, item_size, MaxIteration, pop_size, upper_bound, initial_sol):
    problem = MyProblem(T, product_size, item_size, upper_bound)
    termination = get_termination("n_gen", MaxIteration)
    algorithm = GA(
    pop_size,
    sampling=FloatRandomSampling(),
    selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
    crossover=SimulatedBinaryCrossover(prob=0.9),
    mutation=PolynomialMutation(prob=0.1),
    survival=FitnessSurvival(),
    n_offsprings=None,
    eliminate_duplicates=True)
    res = minimize(problem, algorithm, termination, seed=99, verbose=False)
    # print(problem.fitness_list)


    # Store best result
    every_best_value = [initial_sol]
    # print(MaxIteration, pop_size)
    for i in range(MaxIteration*pop_size):
        if every_best_value[i] >= problem.fitness_list[i]:
            every_best_value.append(problem.fitness_list[i])
        elif every_best_value[i] <= problem.fitness_list[i]:
            every_best_value.append(every_best_value[i])


    # best_solution = res.X.astype('int')
    print('The best fitness: %d' %res.F[0])
    # print("Best solution found: \nX = %s" %res.X.astype('int'))
    
    return res.F[0], every_best_value

'''# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (200, 40, 30)
	import time
	time.clock = time.time
	
	# genetic algorithm new

	Max_measurements = 4500*20
	upper_bound = product_size*1000
	initial_sol = 400000000
 
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga = ga_fun(T, product_size, item_size, int(Max_measurements/ga_pop_size), ga_pop_size, upper_bound, initial_sol)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)

    # visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(bl_ga,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
'''
