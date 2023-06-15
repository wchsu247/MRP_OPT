import time
time.clock = time.time
import opt_ga, opt_random

def main():
	
	#=============================index setting==============================
	T, product_size, item_size =  (5, 4, 3)
	print(f'T={T},  product_size={product_size}, item_size={item_size}')
	#========================================================================
	
	# genetic algorithm
	tic = time.clock()
	best_ga = opt_ga.ga_fun(T, product_size, item_size)
	print(">> GA in %.5f sec." %(time.clock()-tic))


	# fully random search
	tic = time.clock()
	best_random = opt_random.random_fun(T, product_size, item_size)
	print(">> Random in %.5f sec." %(time.clock()-tic))


	# conclusion
	print('The best ans of GA: ', best_ga)
	print('The best ans of random: ', best_random)


if __name__ == '__main__':
	main()