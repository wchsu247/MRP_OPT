import time
time.clock = time.time
import opt_ga, opt_random, opt_spsa, opt_de, visualization

if __name__ == '__main__':
	
	#=============================index setting==============================
	T, product_size, item_size =  (5, 4, 3)
	# MaxIteration = 30
	Max_measurements = 1500 # This value should be a multiple of 'pop_size = 50' and 'spsa_measurements_per_iteration = 3'
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	#========================================================================
	
	# genetic algorithm
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga = opt_ga.ga_fun(T, product_size, item_size, int(Max_measurements/ga_pop_size) , ga_pop_size)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)

	
	# fully random search
	tic = time.clock()
	best_random, bl_random = opt_random.random_fun(T, product_size, item_size, Max_measurements)
	time_random = time.clock()-tic
	print(">> Random in %.5f sec." %time_random)


	# spsa algorithm
	spsa_measurements_per_iteration = 3
	tic = time.clock()
	best_spsa, bl_spsa = opt_spsa.spsa_fun(T, product_size, item_size, int(Max_measurements/spsa_measurements_per_iteration))
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)

	# differential evolution algorithm
	de_pop_size = 50
	tic = time.clock()
	best_de, bl_de = opt_de.de_fun(T, product_size, item_size, int(Max_measurements/de_pop_size) , de_pop_size)
	time_de = time.clock()-tic
	print(">> DE in %.5f sec." %time_de)


	# conclusion
	print("The best ans of GA: %.5f for %.5f sec." % (best_ga, time_ga))
	print("The best ans of random: %.5f for %.5f sec." % (best_random, time_random))
	print("The best ans of SPSA: %.5f for %.5f sec." % (best_spsa, time_spsa))
	print("The best ans of DE: %.5f for %.5f sec." % (best_de, time_de))
	
	# visualization
	visualization.vis(bl_ga, bl_random, bl_spsa, bl_de)
	
