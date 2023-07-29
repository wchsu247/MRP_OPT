import numpy
import matplotlib.pyplot as plt
import replications_of_sim as ros
import cost_evaluation as ce
import sys
MAX_INT=sys.maxsize
import warnings
warnings.filterwarnings('ignore')

def cal_pop_fitness(T, product_size, item_size, bom, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = []
    for i in pop:
        fitness.append(ros.replications_of_sim(T, product_size, item_size, bom, i.reshape(T,item_size).astype('int')))
    return fitness

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
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover
#--------------------------------------------------------------------------------
def ga_fun(T, product_size, item_size, bom, MaxIteration, pop_size, upper_bound, initial_fit, initial_sol, num_parents_mating = 6):

	# Number of the weights we are looking to optimize.
	num_weights = T*item_size

	"""
	Genetic algorithm parameters:
		Mating pool size
		Population size
	"""
	# sol_per_pop = 50
	

	# Defining the population size.
	sol_size = (pop_size, num_weights) # The population will have pop_size chromosome where each chromosome has num_weights genes.
	#Creating the initial population.
	new_population = numpy.random.uniform(low=0, high=upper_bound, size=(pop_size-1, num_weights))
	new_population = numpy.append(new_population, initial_sol, axis = 0)
 
 
	fitness_list = []
	current_best_fit = initial_fit
	current_best_sol = new_population[0]

	measurement = 0
	while measurement < MaxIteration:
		
		
		# print("Generation : ", measurement)
		# Measing the fitness of each chromosome in the population.
		fitness = cal_pop_fitness(T, product_size, item_size, bom, new_population)
		fitness_list.extend(fitness)
  
		if min(fitness) < current_best_fit:
			current_best_fit = min(fitness)
			current_best_sol = new_population[fitness.index(current_best_fit)]

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
	every_best_value = [initial_fit]
 
	# print(MaxIteration, pop_size)
	for i in range(MaxIteration):
		if every_best_value[i] >= fitness_list[i]:	every_best_value.append(fitness_list[i])
		elif every_best_value[i] <= fitness_list[i]:	every_best_value.append(every_best_value[i])

	print('The best fitness: %d' %current_best_fit)
	return current_best_fit, every_best_value, current_best_sol.reshape(T,item_size).astype('int')

'''# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (5, 4, 3)
	import time
	time.clock = time.time

	Max_measurements = 100
	upper_bound = product_size*20
 
	# update initial solution
	initial_sol = numpy.ones((1, T*item_size))*upper_bound
	initial_fit = ros.replications_of_sim(T, product_size, item_size, initial_sol.reshape(T,item_size))
	# print(initial_sol, initial_fit)

	ga_pop_size = 8
	tic = time.clock()
	best_ga, bl_ga, ans_ga = ga_fun(T, product_size, item_size, Max_measurements, ga_pop_size, upper_bound, initial_fit, initial_sol)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)

	
	# print(ans_ga)
	# a, b, c = ce.cost_evaluation(T, product_size, item_size, ans_ga.reshape(T,item_size).astype('int'))
	# print(a, b, c)
	

	# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(bl_ga,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
'''