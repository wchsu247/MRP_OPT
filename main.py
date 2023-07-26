import numpy as np
import time
time.clock = time.time
import opt_ga_new, old_code.opt_random as opt_random, opt_spsa, old_code.opt_de as opt_de, visualization, visualization_og
import  opt_mixed_ga_spsa, old_code.opt_mixed_ga_spsa_2 as opt_mixed_ga_spsa_2, opt_mixed_ga_spsa_3

if __name__ == '__main__':
	
	#=============================index setting==============================
	T, product_size, item_size =  (200, 40, 30) # product_size should be power of 2
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	upper_bound = 40000
	# MaxIteration = 30
	Max_measurements = 4500*4 # This value should be a multiple of 'pop_size = 50' and 'spsa_measurements_per_iteration = 3'
	# initial_sol = ros.replications_of_sim(T, product_size, item_size, np.random.randint(0, upper_bound/20, size=(T, item_size)))
	initial_sol = 940000000
	print(f'initial fitness = {initial_sol}')
	#========================================================================
	
	
	'''# genetic algorithm
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga = opt_ga.ga_fun(T, product_size, item_size, int(Max_measurements/ga_pop_size), ga_pop_size, upper_bound)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)
	'''
	
	# genetic algorithm new
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga = opt_ga_new.ga_fun(T, product_size, item_size, int(Max_measurements/ga_pop_size), ga_pop_size, upper_bound, initial_sol)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)

	'''
	# fully random search
	tic = time.clock()
	best_random, bl_random = opt_random.random_fun(T, product_size, item_size, Max_measurements, upper_bound, initial_sol)
	time_random = time.clock()-tic
	print(">> Random in %.5f sec." %time_random)
	'''


	# spsa algorithm
	spsa_measurements_per_iteration = 3
	tic = time.clock()
	best_spsa, bl_spsa = opt_spsa.spsa_fun(T, product_size, item_size, int(Max_measurements/spsa_measurements_per_iteration), upper_bound, initial_sol)
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)


	'''# differential evolution algorithm
	de_pop_size = 50
	tic = time.clock()
	best_de, bl_de = opt_de.de_fun(T, product_size, item_size, int(Max_measurements/de_pop_size), de_pop_size, upper_bound, initial_sol)
	time_de = time.clock()-tic
	print(">> DE in %.5f sec." %time_de)
	'''

	# mixed ga and spsa algorithm
	mixed_pop_size = 15
	spsa_round = 10
	spsa_measurements_per_iteration = 3
	tic = time.clock()
	best_mix, bl_mix = opt_mixed_ga_spsa.mix_fun(T, product_size, item_size, int(Max_measurements/(mixed_pop_size*spsa_round*spsa_measurements_per_iteration)), mixed_pop_size, spsa_round, upper_bound, initial_sol)
	time_mix = time.clock()-tic
	print(">> MIX in %.5f sec." %time_mix)
	

	# mixed ga and spsa algorithm 2
	mix2_pop_size = 50
	tic = time.clock()
	best_mix2, bl_mix2 = opt_mixed_ga_spsa_2.mix2_fun(T, product_size, item_size, Max_measurements, mix2_pop_size, upper_bound, initial_sol)
	time_mix2 = time.clock()-tic
	print(">> MIX2 in %.5f sec." %time_mix2)
	
	
	# mixed ga and spsa algorithm 3
	mix3_pop_size = 25
	tic = time.clock()
	best_mix3, bl_mix3 = opt_mixed_ga_spsa_3.mix3_fun(T, product_size, item_size, Max_measurements, mix3_pop_size, upper_bound, initial_sol)
	time_mix3 = time.clock()-tic
	print(">> MIX3 in %.5f sec." %time_mix3)
	
	# print(len(bl_ga), len(bl_spsa))

	# conclusion
	print("The best ans of GA:   %.5f for %.5f sec." % (best_ga, time_ga))
	# print("The best ans of RS:   %.5f for %.5f sec." % (best_random, time_random))
	print("The best ans of SPSA: %.5f for %.5f sec." % (best_spsa, time_spsa))
	# print("The best ans of DE:   %.5f for %.5f sec." % (best_de, time_de))
	# print("The best ans of MIX:  %.5f for %.5f sec." % (best_mix, time_mix))
	# print("The best ans of MIX2: %.5f for %.5f sec." % (best_mix2, time_mix2))
	# print("The best ans of MIX3: %.5f for %.5f sec." % (best_mix3, time_mix3))
 
	# visualization
	visualization.vis(bl_ga, bl_spsa)
	visualization_og.vis(bl_ga, bl_spsa, bl_mix, bl_mix2, bl_mix3)
