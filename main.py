import time
time.clock = time.time
import opt_ga, opt_random

def main():
	
	#=============================index setting==============================
	T, product_size, item_size =  (52, 4, 3)
	simulation_times = 4
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	#========================================================================
	
	# genetic algorithm
	tic = time.clock()
	best_ga = opt_ga.ga_fun(T, product_size, item_size, simulation_times)
	print(">> GA in %.5f sec." %(time.clock()-tic))


	# fully random
	tic = time.clock()
	best_random = opt_random.random_fun(T, product_size, item_size)
	print(">> Random in %.5f sec." %(time.clock()-tic))


	# conclusion
	print('The best ans of GA: ', best_ga)
	print('The best ans of random: ', best_random)


if __name__ == '__main__':
	main()