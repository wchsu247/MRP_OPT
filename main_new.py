import numpy as np
import time
time.clock = time.time
import opt_ga, opt_spsa, opt_gsha, visualization
import  replications_of_sim as ros, cost_evaluation as ce

if __name__ == '__main__':
	
	#=============================index setting==============================
	T, product_size, item_size =  (3, 4, 5)
	bom = np.random.randint(2, size=(product_size, item_size))
	
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	print(bom)
 
	upper_bound = product_size*800
	Max_measurements = 50 
 
	# update initial solution
	initial_sol = np.ones((1, T*item_size))*400*20
	initial_fit, y = ros.replications_of_sim(T, product_size, item_size, bom, initial_sol.reshape(T,item_size).astype('int'))
	print(f'initial fitness = {initial_fit}, rc = {y}')
	#========================================================================
 
	# genetic algorithm
	ga_pop_size = 50
	tic = time.clock()
	best_ga, bl_ga, ans_ga, rc_ga = opt_ga.ga_fun(T, product_size, item_size, bom, Max_measurements, ga_pop_size, upper_bound, initial_fit, initial_sol)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)


	# spsa algorithm
	tic = time.clock()
	best_spsa, bl_spsa, ans_spsa, rc_spsa = opt_spsa.spsa_fun(T, product_size, item_size, bom, Max_measurements, upper_bound, initial_fit, initial_sol)	
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)


	# Switching-type GSHA
	tic = time.clock()
	best_gsha, bl_gsha, ans_gsha, d, rc_gsha = opt_gsha.gsha_fun(T, product_size, item_size, bom, Max_measurements, ga_pop_size, upper_bound, initial_fit, initial_sol)
	time_gsha = time.clock()-tic
	print(">> GSHA in %.5f sec." %time_ga)
 
 	#========================================================================
	# conclusion
	print("The best ans of GA:   %.5f for %.5f sec." % (best_ga, time_ga))
	print("The best ans of SPSA: %.5f for %.5f sec." % (best_spsa, time_spsa))
	print("The best ans of GSHA: %.5f for %.5f sec." % (best_gsha, time_gsha))
	
	# cost evaluation
	target = np.ones((1, T*item_size))*400
	sample_mean_initial, sample_std_initial, sample_size_initial = ce.cost_evaluation(T, product_size, item_size, bom, target.reshape(T,item_size))
	sample_mean_ga, sample_std_ga, sample_size_ga = ce.cost_evaluation(T, product_size, item_size, bom, ans_ga)
	sample_mean_spsa, sample_std_spsa, sample_size_spsa = ce.cost_evaluation(T, product_size, item_size, bom, ans_spsa)
	sample_mean_gsha, sample_std_gsha, sample_size_gsha = ce.cost_evaluation(T, product_size, item_size, bom, ans_gsha)

	# improve rate
	ir_target = sample_mean_initial
 
	# visualization
	visualization.vis(bl_ga, bl_spsa, bl_gsha, d, ir_target)
