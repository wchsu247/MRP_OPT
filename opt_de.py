import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import replications_of_sim as ros
import matplotlib.pyplot as plt

class MyProblem(ElementwiseProblem):
    def __init__(self, T, product_size, item_size, upper_bound = 1024):
        super().__init__(n_var=T*item_size, n_obj=1, n_constr=0, xl=np.zeros(T*item_size), xu=np.ones(T*item_size) * upper_bound)
        self.parameters=[T, product_size, item_size]
        self.count = 0
        self.fitness_list = []

    def _evaluate(self, x, out, *args, **kwargs):
        print(">> Case %d" %(self.count+1))
        self.count += 1
        T, product_size, item_size = self.parameters[0], self.parameters[1], self.parameters[2]
        f1 = ros.replications_of_sim(T, product_size, item_size, x.reshape(T,item_size).astype('int'))
        self.fitness_list.append(f1)
        out["F"] = [f1]

def de_fun(T, product_size, item_size, MaxIteration, pop_size):
    problem = MyProblem(T, product_size, item_size)
    termination = get_termination("n_gen", MaxIteration)

    algorithm = DE(
        pop_size,
        sampling=LHS(),
        variant="DE/rand/1/bin",
        CR=0.3,
        dither="vector",
        jitter=False
    )

    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    # print(problem.fitness_list)
    
    # Store best result
    every_best_value = [problem.fitness_list[0]]
    for i in range(MaxIteration):
        if every_best_value[i] >= problem.fitness_list[i+1]:
            every_best_value.append(problem.fitness_list[i+1])
        elif every_best_value[i] <= problem.fitness_list[i+1]:
            every_best_value.append(every_best_value[i])
    
    # best_solution = res.X.astype('int')
    print('The best fitness: ', res.F[0])
    # print("Best solution found: \nX = %s\nF = %s" % (res.X, res.X.astype('int')))
    
    return res.F[0], every_best_value

'''# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (200, 40, 30)
	import time
	time.clock = time.time
	
	tic = time.clock()
	best_de, bl_de = de_fun(T, product_size, item_size, 2, 50)
	time_spsa = time.clock()-tic
	print(">> DE in %.5f sec." %time_spsa)

    # visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(bl_de,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
'''