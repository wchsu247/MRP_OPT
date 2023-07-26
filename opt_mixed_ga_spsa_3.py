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

def spsa_fun(T, product_size, item_size, opt_count_limit, upper_bound, initial_sol, ga_best_solution, lower_bound = 0):
	'''
		Input: initial solution of arrival
		opt_count_limit: # iterations for the SPSA algorithm
	'''
	# -----------------------------------------------------------
	# index setting (1)
	alpha = .602 # .602 from (Spall, 1998)
	gamma = .167 # .167 default
	a = .000101 # .101 found empirically using HyperOpt
	A = .193 # .193 default
	c = 1 # .0277 default # T * product_size *item_size
	u = ga_best_solution
	d_k = 100
	# sample_std = normalization(T, product_size, item_size, upper_bound)
	# print(sample_mean)
	# scalar_u = ros.replications_of_sim(T, product_size, item_size, u)
	# print(u)
	best_solution = ga_best_solution
	best_obj = initial_sol
	best_obj_list = [initial_sol]

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

		obj_list = [ros.replications_of_sim(T, product_size, item_size, u), y_thetaplus, y_thetaminus]
		sol_list = [u, thetaplus, thetaminus]
		obj_value = min(obj_list)
		obj_solution = sol_list[obj_list.index(min(obj_list))]

		# print(obj_value)

		# Step 6: Check for convergence
		if obj_value < best_obj:
			best_obj = obj_value
			best_solution = obj_solution
		best_obj_list.append(best_obj)

	# print("The best fitness:   %d" %(best_obj))
	spsa_ans_list = []
	# print(len(best_obj_list),len(spsa_ans_list))
	spsa_measurment_per_iteration = 3
	for i in range(len(best_obj_list)-1):
		for k in range(spsa_measurment_per_iteration): spsa_ans_list.append(best_obj_list[i+1])
	return best_obj, best_solution, spsa_ans_list

def ga_sol_list_order(upper_bound, fitness_list):
    every_best_value = [upper_bound]
    for i in range(len(fitness_list)-1):
        if every_best_value[i] >= fitness_list[i+1]:
            every_best_value.append(fitness_list[i+1])
        elif every_best_value[i] <= fitness_list[i+1]:
            every_best_value.append(every_best_value[i])
    return every_best_value


class MyProblem(ElementwiseProblem):
    def __init__(self, T, product_size, item_size, upper_bound):
        super().__init__(n_var=T*item_size, n_obj=1, n_constr=0, xl=np.zeros(T*item_size), xu=np.ones(T*item_size) * upper_bound) # xl=np.zeros(T*item_size)
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

def mix3_fun(T, product_size, item_size, MaxIteration, pop_size, upper_bound, initial_sol, check_range = 1500):
    problem = MyProblem(T, product_size, item_size, upper_bound)
    termination = get_termination("n_gen", int(check_range/pop_size))
    algorithm = GA(
    pop_size,
    sampling=FloatRandomSampling(),
    selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
    crossover=SimulatedBinaryCrossover(prob=0.9),
    mutation=PolynomialMutation(prob=0.1),
    survival=FitnessSurvival(),
    n_offsprings=None,
    eliminate_duplicates=True)
    spsa_measurements_per_iteration = 3
    every_best_value = [initial_sol]

    flag = 1
    current_best_obj = initial_sol
    print("GA", end="")
    res = minimize(problem, algorithm, termination, seed=43, verbose=False)
    best_obj = int(res.F[0])
    ga_best_solution = res.X.reshape(T,item_size).astype('int')
    every_best_value.extend(ga_sol_list_order(current_best_obj, problem.fitness_list))
    problem.fitness_list = []

    for i in range(int(MaxIteration/check_range)-1):

        if best_obj < current_best_obj and flag == 1: # remain GA
            current_best_obj = best_obj
            print("->GA", end="")
            res = minimize(problem, algorithm, termination, seed=1, verbose=False)
            every_best_value.extend(ga_sol_list_order(current_best_obj, problem.fitness_list))
            problem.fitness_list = []
            best_obj = int(res.F[0])
            ga_best_solution = res.X.reshape(T,item_size).astype('int')
            every_best_value.extend(problem.fitness_list)

        elif best_obj >= current_best_obj and flag == 1:   # switch to SPSA
            print("->SPSA", end="")
            best_obj, spsa_best_solution, spsa_ans_list = spsa_fun(T, product_size, item_size, int(check_range/spsa_measurements_per_iteration), upper_bound, current_best_obj, ga_best_solution)
            every_best_value.extend(spsa_ans_list)
            flag = 2

        elif best_obj < current_best_obj and flag == 2:    # remain SPSA
            current_best_obj = best_obj
            print("->SPSA", end="")
            best_obj, spsa_best_solution, spsa_ans_list = spsa_fun(T, product_size, item_size, int(check_range/spsa_measurements_per_iteration), upper_bound, current_best_obj, spsa_best_solution)
            every_best_value.extend(spsa_ans_list)

        elif best_obj >= current_best_obj and flag == 2:   # switch to GA
            print("->GA", end="")
            res = minimize(problem, algorithm, termination, seed=52, verbose=False)
            every_best_value.extend(ga_sol_list_order(current_best_obj, problem.fitness_list))
            problem.fitness_list = []
            best_obj = int(res.F[0])
            ga_best_solution = res.X.reshape(T,item_size).astype('int')
            flag = 1
            
    print("")


    print('The best fitness: %d' %min(best_obj, current_best_obj))
    # print("Best solution found: \nX = %s" %res.X.astype('int'))
    
    return min(best_obj, current_best_obj), every_best_value

'''# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (200, 40, 500)
	import time
	time.clock = time.time
	
	# mixed ga and spsa algorithm 3
	Max_measurements = 4500*3
	upper_bound = 38400/5
	initial_sol = 31713756827
	mix3_pop_size = 25
	tic = time.clock()
	best_mix3, bl_mix3 = mix3_fun(T, product_size, item_size, Max_measurements, mix3_pop_size, upper_bound, initial_sol)
	time_mix3 = time.clock()-tic
	print(">> MIX3 in %.5f sec." %time_mix3)

	# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Measurements",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)
	plt.axvline(x=5600, c="r", ls="--", lw=2)
	plt.axvline(x=9200, c="r", ls="--", lw=2)
	plt.axvline(x=15200, c="r", ls="--", lw=2)
	plt.axvline(x=17700, c="r", ls="--", lw=2)
	plt.plot(bl_mix3,linewidth = 2, label = "MIX3 best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
'''