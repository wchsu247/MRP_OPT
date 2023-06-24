import time
time.clock = time.time
import opt_ga, opt_random, opt_spsa, visualization

if __name__ == '__main__':
    
	#=============================index setting==============================
	T, product_size, item_size =  (5, 4, 3)
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	#========================================================================
	
	# genetic algorithm
	tic = time.clock()
	best_ga, bl_ga = opt_ga.ga_fun(T, product_size, item_size)
	time_ga = time.clock()-tic
	print(">> GA in %.5f sec." %time_ga)


	# fully random search
	tic = time.clock()
	best_random, bl_random = opt_random.random_fun(T, product_size, item_size)
	time_random = time.clock()-tic
	print(">> Random in %.5f sec." %time_random)


	# spsa algorithm
	tic = time.clock()
	best_spsa, bl_spsa = opt_spsa.spsa_fun(T, product_size, item_size)
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)


	# conclusion
	print('The best ans of GA: ', best_ga)
	print('The best ans of random: ', best_random)
	print('The best ans of SPSA: ', best_spsa)

	# visualization
	visualization.vis(bl_ga, bl_random, bl_spsa)
