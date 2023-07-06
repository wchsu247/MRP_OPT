import numpy as np
import time
time.clock = time.time
import opt_ga, opt_ga_new, opt_random, opt_spsa, opt_de, opt_mixed_ga_spsa, visualization, replications_of_sim as ros

if __name__ == '__main__':
	
	#=============================index setting==============================
	T, product_size, item_size =  (5, 4, 3) # product_size should be power of 2
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	upper_bound = product_size*1000
	# MaxIteration = 30
	Max_measurements = 4500 # This value should be a multiple of 'pop_size = 50' and 'spsa_measurements_per_iteration = 3'
	initial_sol = ros.replications_of_sim(T, product_size, item_size, np.random.randint(0, upper_bound/20, size=(T, item_size)))
	print(f'initial fitness = {initial_sol}')
	#========================================================================
	
	
	'''# genetic algorithm
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga = opt_ga.ga_fun(T, product_size, item_size, int(Max_measurements/ga_pop_size), ga_pop_size, upper_bound)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)
	'''
	
	# genetic algorithm 2
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga = opt_ga_new.ga_fun(T, product_size, item_size, int(Max_measurements/ga_pop_size), ga_pop_size, upper_bound, initial_sol)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)

	
	# fully random search
	tic = time.clock()
	best_random, bl_random = opt_random.random_fun(T, product_size, item_size, Max_measurements, upper_bound, initial_sol)
	time_random = time.clock()-tic
	print(">> Random in %.5f sec." %time_random)


	# spsa algorithm
	spsa_measurements_per_iteration = 3
	tic = time.clock()
	best_spsa, bl_spsa = opt_spsa.spsa_fun(T, product_size, item_size, int(Max_measurements/spsa_measurements_per_iteration), upper_bound, initial_sol)
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)


	# differential evolution algorithm
	de_pop_size = 50
	tic = time.clock()
	best_de, bl_de = opt_de.de_fun(T, product_size, item_size, int(Max_measurements/de_pop_size), de_pop_size, upper_bound, initial_sol)
	time_de = time.clock()-tic
	print(">> DE in %.5f sec." %time_de)
	

	# mixed ga and spsa algorithm
	mixed_pop_size = 25
	spsa_round = 6
	spsa_measurements_per_iteration = 3
	tic = time.clock()
	best_mix, bl_mix = opt_mixed_ga_spsa.mix_fun(T, product_size, item_size, int(Max_measurements/(mixed_pop_size*spsa_round*spsa_measurements_per_iteration)), mixed_pop_size, spsa_round, upper_bound, initial_sol)
	time_mix = time.clock()-tic
	print(">> MIX in %.5f sec." %time_mix)


	print(len(bl_ga), len(bl_random), len(bl_spsa), len(bl_de), len(bl_mix))
	# conclusion
	print("The best ans of GA: %.5f for %.5f sec." % (best_ga, time_ga))
	print("The best ans of random: %.5f for %.5f sec." % (best_random, time_random))
	print("The best ans of SPSA: %.5f for %.5f sec." % (best_spsa, time_spsa))
	print("The best ans of DE: %.5f for %.5f sec." % (best_de, time_de))
	print("The best ans of MIX: %.5f for %.5f sec." % (best_mix, time_mix))
	
	# visualization
	visualization.vis(bl_ga, bl_random, bl_spsa, bl_de, bl_mix)
	
