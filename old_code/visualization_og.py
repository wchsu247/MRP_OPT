import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random



def vis(bl_ga, bl_spsa, bl_mix, bl_mix2, bl_mix3):

	random_number = np.random.randint(1000, 9999)
	plt.figure(figsize = (15,8))
	plt.xlabel("# Measurements",fontsize = 15)
	plt.ylabel("Total Cost",fontsize = 15)

	# plt.plot(bl_random, 'k')
	plt.plot(bl_ga, 'r')
	plt.plot(bl_spsa, 'b')
	# plt.plot(bl_de, 'g')
	plt.plot(bl_mix, 'm')
	plt.plot(bl_mix2, 'c')
	plt.plot(bl_mix3, 'g')

	plt.legend(['GA', 'SPSA', 'Parallel GSHA', 'Two-phase GSHA', 'GSHA'])
	plt.title('MRP Problem',fontsize = 15)
	plt.savefig('c:/Users/MB608/Desktop/theis_MRP/theis_MRP/Reusult_Plot/'+str(random_number)+'.png')
	print(random_number)
	# plt.savefig(f'c:/Users/MB608/Desktop/theis_MRP/theis_MRP/Reusult_Plot/{random_number}.png')
	# plt.savefig(f'/Users/user/Desktop/MRP-2/Reusult_Plot/{random_number}.png')
	# plt.show()
	return 