import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def vis(bl_ga, bl_random, bl_spsa):

	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(bl_ga, 'r')
	plt.plot(bl_random, 'k')
	plt.plot(bl_spsa, 'b')

	plt.legend(['GA', 'Random', 'SPSA'])
	plt.title('MRP Problem')
	plt.savefig("Plot.png")
	# plt.show()
	return 