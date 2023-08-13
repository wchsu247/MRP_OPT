import numpy as np
import time
time.clock = time.time
import opt_ga, opt_ga_new, opt_spsa, old_code.opt_de as opt_de, visualization, visualization_og
import  opt_mixed_ga_spsa, opt_mixed_ga_spsa_3,replications_of_sim as ros, opt_gsha, cost_evaluation as ce

if __name__ == '__main__':
	
	#=============================index setting==============================
	T, product_size, item_size =  (5, 4, 3) # product_size should be power of 2
	bom = np.random.randint(2, size=(product_size, item_size))
	
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	print(bom)
 
	upper_bound = product_size*40
	# MaxIteration = 30
	Max_measurements = 9000 # This value should be a multiple of 'pop_size = 50' and 'spsa_measurements_per_iteration = 3'
	# initial_sol = ros.replications_of_sim(T, product_size, item_size, np.random.randint(0, upper_bound/20, size=(T, item_size)))
	# initial_sol = 940000000
 
	# update initial solution
	initial_sol = np.ones((1, T*item_size))*upper_bound
	initial_fit = ros.replications_of_sim(T, product_size, item_size, bom, initial_sol.reshape(T,item_size))
	print(f'initial fitness = {initial_fit}')
	#========================================================================
	
	
	# genetic algorithm
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga, ans_ga = opt_ga.ga_fun(T, product_size, item_size, bom, Max_measurements, ga_pop_size, upper_bound, initial_fit, initial_sol)
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
	'''
	# fully random search
	tic = time.clock()
	best_random, bl_random = opt_random.random_fun(T, product_size, item_size, Max_measurements, upper_bound, initial_sol)
	time_random = time.clock()-tic
	print(">> Random in %.5f sec." %time_random)
	'''


	# spsa algorithm
	tic = time.clock()
	best_spsa, bl_spsa, ans_spsa = opt_spsa.spsa_fun(T, product_size, item_size, bom, Max_measurements, upper_bound, initial_fit, initial_sol)	
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)


	'''# differential evolution algorithm
	de_pop_size = 50
	tic = time.clock()
	best_de, bl_de = opt_de.de_fun(T, product_size, item_size, int(Max_measurements/de_pop_size), de_pop_size, upper_bound, initial_sol)
	time_de = time.clock()-tic
	print(">> DE in %.5f sec." %time_de)
	'''

	'''# mixed ga and spsa algorithm
	mixed_pop_size = 15
	spsa_round = 10
	spsa_measurements_per_iteration = 3
	tic = time.clock()
	best_mix, bl_mix = opt_mixed_ga_spsa.mix_fun(T, product_size, item_size, int(Max_measurements/(mixed_pop_size*spsa_round*spsa_measurements_per_iteration)), mixed_pop_size, spsa_round, upper_bound, initial_sol)
	time_mix = time.clock()-tic
	print(">> MIX in %.5f sec." %time_mix)
	'''

	'''# mixed ga and spsa algorithm 2
	mix2_pop_size = 50
	tic = time.clock()
	best_mix2, bl_mix2 = opt_mixed_ga_spsa_2.mix2_fun(T, product_size, item_size, Max_measurements, mix2_pop_size, upper_bound, initial_sol)
	time_mix2 = time.clock()-tic
	print(">> MIX2 in %.5f sec." %time_mix2)
	'''
	
	'''# mixed ga and spsa algorithm 3
	mix3_pop_size = 25
	tic = time.clock()
	best_mix3, bl_mix3 = opt_mixed_ga_spsa_3.mix3_fun(T, product_size, item_size, Max_measurements, mix3_pop_size, upper_bound, initial_sol)
	time_mix3 = time.clock()-tic
	print(">> MIX3 in %.5f sec." %time_mix3)
	'''
	

	# Switching-type GSHA
	tic = time.clock()
	best_gsha, bl_gsha, ans_gsha, d = opt_gsha.gsha_fun(T, product_size, item_size, bom, Max_measurements, ga_pop_size, upper_bound, initial_fit, initial_sol)
	time_gsha = time.clock()-tic
	print(">> GSHA in %.5f sec." %time_ga)


	# print(len(bl_ga), len(bl_spsa), len(bl_gsha))
 
	# conclusion
	print("The best ans of GA:   %.5f for %.5f sec." % (best_ga, time_ga))
	print("The best ans of SPSA: %.5f for %.5f sec." % (best_spsa, time_spsa))
	print("The best ans of GSHA: %.5f for %.5f sec." % (best_gsha, time_gsha))
	
	# cost evaluation
	sample_mean_initial, sample_std_initial, sample_size_initial = ce.cost_evaluation(T, product_size, item_size, bom, initial_sol.reshape(T,item_size))
	sample_mean_ga, sample_std_ga, sample_size_ga = ce.cost_evaluation(T, product_size, item_size, bom, ans_ga)
	sample_mean_spsa, sample_std_spsa, sample_size_spsa = ce.cost_evaluation(T, product_size, item_size, bom, ans_spsa)
	sample_mean_gsha, sample_std_gsha, sample_size_gsha = ce.cost_evaluation(T, product_size, item_size, bom, ans_gsha)

	# improve rate
	ir_target = initial_fit*0.2
	ir_ga = sample_mean_ga/ sample_mean_initial
	ir_spsa = sample_mean_spsa/ sample_mean_initial
	ir_gsha = sample_mean_gsha/ sample_mean_initial
 
	# visualization
	visualization.vis(bl_ga, bl_spsa, bl_gsha, d, ir_target)
	# visualization_og.vis(bl_ga, bl_spsa, bl_mix, bl_mix2, bl_mix3)
