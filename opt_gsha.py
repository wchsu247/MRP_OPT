import numpy
numpy.random.seed(1)
import matplotlib.pyplot as plt
import replications_of_sim as ros
import cost_evaluation as ce
import sys
MAX_INT=sys.maxsize
import warnings
warnings.filterwarnings('ignore')

# GA function --------------------------------------------------------------------------------
def cal_pop_fitness(T, product_size, item_size, bom, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = []
    rc = 0
    for i in pop:
        fit, random_count = ros.replications_of_sim(T, product_size, item_size, bom, i.reshape(T,item_size).astype('int'))
        fitness.append(fit)
        rc += random_count
    return fitness, rc

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    temp_fit = fitness.copy()
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(temp_fit == numpy.min(temp_fit))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        temp_fit[max_fitness_idx] = MAX_INT
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        numpy.random.seed(1)
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover

def ga_fun(T, product_size, item_size, bom, MaxIteration, pop_size, upper_bound, initial_fit, initial_sol, measurement, ga_check_count, num_parents_mating = 6):
	print(" GA ", end="")
 
	# random count
	rc = 0
 
	# Number of the weights we are looking to optimize.
	num_weights = T*item_size	

	# Defining the population size.
	sol_size = (pop_size, num_weights) # The population will have pop_size chromosome where each chromosome has num_weights genes.
	#Creating the initial population.
	numpy.random.seed(1)
	new_population = numpy.random.uniform(low=0, high=upper_bound, size=(pop_size-1, num_weights))
	new_population = numpy.append(new_population, initial_sol, axis = 0)

	fitness_list = []
	current_best_fit = initial_fit
	current_best_sol = new_population[0]

	check_count = 0
	while measurement < MaxIteration and check_count < ga_check_count:
		
		
		# print("Generation : ", measurement)
		# Measing the fitness of each chromosome in the population.
		fitness, random_count = cal_pop_fitness(T, product_size, item_size, bom, new_population)
		fitness_list.extend(fitness)
		rc += random_count
  
		if min(fitness) < current_best_fit:
			check_count = 0
			current_best_fit = min(fitness)
			current_best_sol = new_population[fitness.index(current_best_fit)]
		else:	check_count += pop_size

		# Selecting the best parents in the population for mating.
		parents = select_mating_pool(new_population, fitness, 
										num_parents_mating)

		# Generating next generation using crossover.
		offspring_crossover = crossover(parents,
										offspring_size=(sol_size[0]-parents.shape[0], num_weights))

		# Adding some variations to the offsrping using mutation.
		offspring_mutation = mutation(offspring_crossover)

		# Creating the new population based on the parents and offspring.
		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation

		# The best result in the current iteration.
		# print("Best result : ", min(fitness))

		measurement += pop_size

	# Store best result
	every_best_value = []
 
	# print(MaxIteration, pop_size)
 
	for i in range(len(fitness_list)):
		if fitness_list[i] < initial_fit:
			initial_fit = fitness_list[i]
			every_best_value.append(fitness_list[i])
		else:	every_best_value.append(initial_fit)

	# print('The best fitness: %d' %current_best_fit)
	return current_best_fit, every_best_value, current_best_sol, measurement, rc
# GA function --------------------------------------------------------------------------------

# SPSA function ------------------------------------------------------------------------------
def spsa_fun(T, product_size, item_size, bom, MaxIteration, upper_bound, initial_fit, ga_best_solution, measurement, spsa_check_count, lower_bound = 0):

	# random count
	rc = 0

	print("SPSA", end="")
	'''
		Input: initial solution of arrival
		opt_count_limit: # iterations for the SPSA algorithm
	'''
	# -----------------------------------------------------------
	# index setting (1)
	alpha = .602 # .602 from (Spall, 1998)
	gamma = .167 # .167 default
	a = 101 # .101 found empirically using HyperOpt
	A = .193 # .193 default
	c = 2.77 # .0277 default # T * product_size *item_size
	u = ga_best_solution.reshape(T,item_size).astype('int')
	d_k = 10


	spsa_measurment_per_iteration = 3
 
	best_solution = u
	best_obj = initial_fit
	best_obj_list = [initial_fit]

	check_count = 0
	k = 0
	while measurement < MaxIteration and check_count < spsa_check_count:

		# print(">> Case %d" %(measurement))
		# index setting (2)

		a_k = a / (A + k + 1)**alpha 	# a_k = 1 / (k+1)
		c_k = c / (k + 1)**gamma		# c_k = 1 / ((1 + k) ** (1 / 6))

		# Step 2: Generation of simultaneous perturbation vector
		delta_k = numpy.random.choice([-d_k,d_k], size=(T, item_size), p=[.5, .5])

		# Step 3: Function evaluations
		thetaplus = numpy.where(u + c_k*delta_k < lower_bound, lower_bound, u + c_k*delta_k)
		thetaplus = numpy.where(thetaplus > upper_bound, upper_bound, thetaplus).astype('int')
		y_thetaplus, random_count = ros.replications_of_sim(T, product_size, item_size, bom, thetaplus)
		rc += random_count

		thetaminus = numpy.where(u - c_k*delta_k < lower_bound, lower_bound, u - c_k*delta_k)
		thetaminus = numpy.where(thetaminus > upper_bound, upper_bound, thetaminus).astype('int')
		y_thetaminus, random_count = ros.replications_of_sim(T, product_size, item_size, bom, thetaminus)
		rc += random_count

		# print(thetaplus.min(), thetaplus.max())

		# Step 4: Gradient approximation
		g_k = numpy.dot((y_thetaplus - y_thetaminus) / (2.0*c_k*d_k**2), delta_k)
		# print(c_k*delta_k[0][0], a_k * g_k[0][0])

		# Step 5: Update u estimate
		u = numpy.where(u - a_k * g_k < lower_bound, lower_bound, u - a_k * g_k)
		u = numpy.where(u > upper_bound, upper_bound, u).astype('int')

		fit, random_count = ros.replications_of_sim(T, product_size, item_size, bom, u)
		rc += random_count
		obj_list = [fit, y_thetaplus, y_thetaminus]
		sol_list = [u, thetaplus, thetaminus]
		obj_value = min(obj_list)
		obj_solution = sol_list[obj_list.index(min(obj_list))]

		# print(obj_value)

		# Step 6: Check for convergence
		if obj_value < best_obj:
			best_obj = obj_value
			best_solution = obj_solution
			check_count = 0
		else: check_count += spsa_measurment_per_iteration
		best_obj_list.append(best_obj)
  
		measurement += spsa_measurment_per_iteration
		k += 1


	# print("The best fitness:   %d" %(best_obj))
	spsa_ans_list = []
	# print(len(best_obj_list),len(spsa_ans_list))
	
	for i in range(len(best_obj_list)-1):
		for k in range(spsa_measurment_per_iteration): spsa_ans_list.append(best_obj_list[i+1])

	return best_obj, spsa_ans_list, best_solution.reshape(1, T*item_size).astype('int'), measurement, rc
# SPSA function ------------------------------------------------------------------------------

def gsha_fun(T, product_size, item_size, bom, MaxIteration, pop_size, upper_bound, initial_fit, initial_sol, ga_check_count=1000, spsa_check_count=1000):
    
    # measurement_count: current measurement
    measurement_count = 0
    rc = 0
    flag = 1
    fitness_list = [initial_fit]
    switching_timing = []

    while measurement_count < MaxIteration:

        if flag == 1:
            ga_best_fitnes, ga_fitness_list, ga_best_sol, ga_measurment, random_count = ga_fun(T, product_size, item_size, bom, MaxIteration, pop_size, upper_bound, initial_fit, initial_sol, measurement_count, ga_check_count)
            fitness_list.extend(ga_fitness_list)
            rc += random_count
            measurement_count = ga_measurment
            initial_fit, initial_sol =  ga_best_fitnes, ga_best_sol
            switching_timing.append(measurement_count)
            flag = 2
        elif flag == 2: 
            spsa_best_fitnes, spsa_fitness_list, spsa_best_sol, spsa_measurment, random_count = spsa_fun(T, product_size, item_size, bom, MaxIteration, upper_bound, initial_fit, initial_sol, measurement_count, spsa_check_count)
            fitness_list.extend(spsa_fitness_list)
            rc += random_count
            measurement_count = spsa_measurment
            initial_fit, initial_sol = spsa_best_fitnes, spsa_best_sol
            switching_timing.append(measurement_count)
            flag = 1

    print("")
    if flag == 1:	best_fitnes, best_sol = spsa_best_fitnes, spsa_best_sol
    else:	best_fitnes, best_sol = ga_best_fitnes, ga_best_sol
    print('The best fitness: %d' %best_fitnes)

    return best_fitnes, fitness_list[0:MaxIteration+1], best_sol.reshape(T,item_size).astype('int'), switching_timing, rc
    

'''
# test ------------------------------------------------------------------------------------
if __name__ == '__main__' :
	import time
	time.clock = time.time
 
	# index setting
	T, product_size, item_size = (5, 4, 3)
	MaxIteration = 4500
	ga_pop_size = 30
	upper_bound = product_size*20

	# update initial solution
	initial_sol = numpy.ones((1, T*item_size))*upper_bound
	initial_fit = ros.replications_of_sim(T, product_size, item_size, initial_sol.reshape(T,item_size))
	print(f'initial fitness = {initial_fit}')

	tic = time.clock()
	a, b, c, d = gsha_fun(T, product_size, item_size, MaxIteration, ga_pop_size, upper_bound, initial_fit, initial_sol)
	time_ga = time.clock()-tic
	print(">> GSHA in %.5f sec." %time_ga)

	# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(b,linewidth = 2, label = "Best fitness convergence", color = 'b')
	for i in range(len(d)-1):	plt.axvline(x=d[i], c="r", ls="--", lw=2)
	plt.legend()
	plt.show()
'''