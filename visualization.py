import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import MultipleLocator
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')



def vis(bl_ga, bl_spsa, bl_gsha, d, ir_target):

	random_number = np.random.randint(1000, 9999)
	plt.figure(figsize = (15,8))
	plt.xlabel("評估次數",fontsize = 15)
	plt.ylabel("總成本評估值",fontsize = 15)

	# plt.plot(bl_random, 'k')
	plt.plot(bl_ga, 'r')
	plt.plot(bl_spsa, 'b')
	# plt.plot(bl_de, 'g')
	# plt.plot(bl_mix, 'm')
	plt.plot(bl_gsha, 'c')
	# plt.plot(bl_mix3, 'g')

	# for i in range(len(d)-1):	plt.axvline(x=d[i], c="r", ls="--", lw=1)

	plt.axhline(y=ir_target, c="m", ls="--", lw=1)


	
	plt.legend(['GA', 'SPSA', 'GSHA'])

 
	# plt.title('MRP Problem',fontsize = 15)
	plt.savefig('/Users/j1999c4/Desktop/SCHOOL/論文專區/論文相關檔案/MRP_code/Reusult_Plot'+str(random_number)+'.png')
	print(random_number)
	# plt.savefig(f'c:/Users/MB608/Desktop/theis_MRP/theis_MRP/Reusult_Plot/{random_number}.png')
	# plt.savefig(f'/Users/user/Desktop/MRP-2/Reusult_Plot/{random_number}.png')
	# plt.show()
	return 