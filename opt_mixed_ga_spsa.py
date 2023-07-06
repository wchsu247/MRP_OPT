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


def spsa_fun(T, product_size, item_size, opt_count_limit, upper_bound, arrival, lower_bound = 0):
	'''
		Input: initial solution of arrival
		opt_count_limit: # iterations for the SPSA algorithm
	'''
	# -----------------------------------------------------------
	# index setting (1)
	alpha = .602 # .602 from (Spall, 1998)
	gamma = .167 # .167 default
	a = .0101 # .101 found empirically using HyperOpt
	A = .193 # .193 default
	c = 1 # .0277 default # T * product_size *item_size
	u = arrival*upper_bound/20
	d_k = 100
	# sample_std = normalization(T, product_size, item_size, upper_bound)
	# print(sample_mean)
	# scalar_u = ros.replications_of_sim(T, product_size, item_size, u)
	# print(u)

	# best_obj = MAX_INT
	best_obj_list = []

	for k in range(opt_count_limit):

		# print(">> Case %d" %(k+1))
		# index setting (2)

		a_k = a / (A + k + 1)**alpha 	# a_k = 1 / (k+1)
		c_k = c / (k + 1)**gamma		# c_k = 1 / ((1 + k) ** (1 / 6))

		# Step 2: Generation of simultaneous perturbation vector
		# choose each component from a bernoulli +-1 distribution with
		# probability of .5 for each +-1 outcome.
		delta_k = np.random.choice([-d_k,d_k], size=(T, item_size), p=[.5, .5])
		# print(c_k*delta_k[0][0])

		# Step 3: Function evaluations
		thetaplus = np.where(u + c_k*delta_k < lower_bound, lower_bound, u + c_k*delta_k)
		thetaplus = np.where(thetaplus > upper_bound, upper_bound, thetaplus).astype('int')
		y_thetaplus = ros.replications_of_sim(T, product_size, item_size, thetaplus)
		
		thetaminus = np.where(u - c_k*delta_k < lower_bound, lower_bound, u - c_k*delta_k)
		thetaminus = np.where(thetaminus > upper_bound, upper_bound, thetaminus).astype('int')
		y_thetaminus = ros.replications_of_sim(T, product_size, item_size, thetaminus)

		# print(thetaplus.min(), thetaplus.max())

		# Step 4: Gradient approximation
		g_k = np.dot((y_thetaplus - y_thetaminus) / (2.0*c_k*d_k**2), delta_k)
		# print(c_k*delta_k[0][0], a_k * g_k[0][0])

		# Step 5: Update u estimate
		# u = np.asarray(np.where((u-a_k*g_k<0, 0, u-a_k*g_k) & (u-a_k*g_k>64, 64, u-a_k*g_k)), dtype = 'int')
		u = np.where(u - a_k * g_k < lower_bound, lower_bound, u - a_k * g_k)
		u = np.where(u > upper_bound, upper_bound, u).astype('int')
		best_obj_list.append(min(ros.replications_of_sim(T, product_size, item_size, u),y_thetaplus,y_thetaminus))
		# print(u)
	return best_obj_list


class MyProblem(ElementwiseProblem):
    def __init__(self, T, product_size, item_size, spsa_round, upper_bound):
        super().__init__(n_var=T*item_size, n_obj=1, n_constr=0, xl=np.zeros(T*item_size), xu=np.ones(T*item_size) * 2)
        self.parameters=[T, product_size, item_size]
        self.count = 0
        self.upper_bound = upper_bound
        self.spsa_round = spsa_round
        self.fitness_list = []

    def _evaluate(self, x, out, *args, **kwargs):
        # print(">> Case %d" %(self.count+1))
        self.count += 1
        T, product_size, item_size = self.parameters[0], self.parameters[1], self.parameters[2]
        arrival = x.reshape(T,item_size).astype('int')
        # print(arrival)
        
        f1 = spsa_fun(T, product_size, item_size, self.spsa_round, self.upper_bound, arrival)
        # f1 = ros.replications_of_sim(T, product_size, item_size, arrival)
        self.fitness_list.extend(f1)
        # print(self.fitness_list)
        out["F"] = [min(f1)]

def mix_fun(T, product_size, item_size, MaxIteration, pop_size, spsa_round, upper_bound, initial_sol):
    problem = MyProblem(T, product_size, item_size, spsa_round, upper_bound)
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
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    # print(problem.fitness_list)
    
    # Store best result
    every_best_value = [initial_sol]
    # print(MaxIteration, pop_size)
    for i in range(MaxIteration*pop_size*spsa_round):
        if every_best_value[i] >= problem.fitness_list[i]:
            every_best_value.append(problem.fitness_list[i])
        elif every_best_value[i] <= problem.fitness_list[i]:
            every_best_value.append(every_best_value[i])
    
    mix_ans_list = [initial_sol]
    spsa_measurment_per_iteration = 3
    for i in range(len(every_best_value)-1):
        for k in range(spsa_measurment_per_iteration): mix_ans_list.append(every_best_value[i+1])
    
    # best_solution = res.X.astype('int')
    print('The best fitness: %d' %min(mix_ans_list))
    # print("Best solution found: \nX = %s\nF = %s" % (res.X, res.X.astype('int')))
    
    return min(mix_ans_list), mix_ans_list

'''# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (200, 40, 30)
	import time
	time.clock = time.time
	
	tic = time.clock()
	best_de, bl_de = mix_fun(T, product_size, item_size, 3, 25, 6, product_size*1000, 1316006610)
	time_spsa = time.clock()-tic
	print(">>  in %.5f sec." %time_spsa)
	print(len(bl_de))
    # visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(bl_de,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
'''