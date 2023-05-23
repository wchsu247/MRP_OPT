import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings
import numpy as np
np.set_printoptions(suppress=True)
# %matplotlib inline
warnings.filterwarnings("ignore")
import MRP_0301_hsu

class GeneticAlgorithm:
	def __init__(self, Nnumber=10, Dimension=3, Bitnum=4, Elite_num=4, CrossoverRate=0.7, MutationRate=0.3, MaxIteration=10):
		self.N = Nnumber
		self.D = Dimension
		self.B = Bitnum
		self.n = Elite_num
		self.cr = CrossoverRate
		self.mr = MutationRate
		self.max_iter = MaxIteration

	def generatePopulation(self):
		population = []
		for number in range(self.N):
			chrom_list = []
			for run in range(self.D):
				element = (np.zeros((1,self.B))).astype(int)
				for i in range(1):
					for j in range(self.B):
						element[i,j] = np.random.randint(0,2)
				chromosome = list(element[0])
				chrom_list.append(chromosome)
			population.append(chrom_list)
		return population

	def B2D(self, pop):
		dec = str(pop[0])+str(pop[1])+str(pop[2])+str(pop[3])
		return int(str(dec),2)
	
	def D2B(self, num):
		return [int(i) for i in (bin(10)[2:])]

	# Rastrigin function
	def fun(self, pop):
		X = np.array(pop)
		# print(X)
		funsum = 0
		for i in range(self.D):
			x = X[:,i]
			# print(x)
			funsum += x**2 - 10*np.cos(2*np.pi*x)
		# print(list(funsum))
		funsum += 10*self.D
		# print(list(funsum))
		return list(funsum)
#============================================================
	# New Objective Function
#============================================================
	def fun_2(self, pop, demand, bom):
		arrival_set = np.array(pop)
		funsum = []
		

		# data for testing
		demand = np.array([[5,5,5,5]])
		# demand = np.array([[5,5,5,5],[5,5,5,5]])
		bom = np.array([[1,1,1],[1,0,1],[1,1,0],[0,0,1]])

		for i in arrival_set:
			print(np.array([i]))
			df_production, df_stock, df_backlog = MRP_0301_hsu.MRP_abstract(np.array([i]), demand, bom)
			obj_value = MRP_0301_hsu.obj_function(df_production['production'].sum(),df_stock['stock'].sum(),df_backlog['backlog_qty'].sum())
			funsum.append(obj_value)

		return list(funsum)
#============================================================
	# Selection method 1
	def Selection1(self, n, pop_bin, fitness):
		select_bin = pop_bin.copy()
		fitness1 = fitness.copy()
		Parents = []
		if sum(fitness1) == 0:
			for i in range(self.n):
				parent = select_bin[random.randint(0,(self.N)-1)]
				Parents.append(parent)
		else:
			for i in range(self.n):
				arr = fitness1.index(min(fitness1))
				Parents.append(select_bin[arr])
				del select_bin[arr]
				del fitness1[arr]
		return Parents

	# Selection method 2
	def Selection(self, n, pop_bin, fitness):
		select_bin = pop_bin.copy()
		fitness1 = fitness.copy()
		Parents = []
		if sum(fitness1) == 0:
			for i in range(self.n):
				parent = select_bin[random.randint(0,(self.N)-1)]
				Parents.append(parent)
		else: 
			NorParent = [(1 - indivi/sum(fitness1))/((self.N-1)) for indivi in fitness1]
			tep = 0
			Cumulist = []
			for i in range(len(NorParent)):
				tep += NorParent[i]
				Cumulist.append(tep)
			#Find parents
			for i in range(self.n):
				z1 = random.uniform(0,1)
				for pick in range(len(Cumulist)):
					if z1<=Cumulist[0]:
						parent = select_bin[NorParent.index(NorParent[0])]
					elif Cumulist[pick] < z1 <=Cumulist[pick+1]:
						parent = select_bin[NorParent.index(NorParent[pick+1])]
				Parents.append(parent)
		return Parents
#============================================================
	# Crossover & Mutation
	def Crossover_Mutation(self, parent1, parent2):
		def swap_machine(element_1, element_2):
			temp = element_1
			element_1 = element_2
			element_2 = temp
			return element_1, element_2
		child_1 = []
		child_2 = []
		for i in range(len(parent1)):
			#隨機生成一數字，用以決定是否進行Crossover
			z1 = random.uniform(0,1)
			if z1 < self.cr:
				z2 = random.uniform(0,1)
				#決定要交換的位置點
				cross_location = math.ceil(z2*(len(parent1[i])-1))
				#Crossover
				parent1[i][:cross_location],parent2[i][:cross_location] = swap_machine(parent1[i][:cross_location],parent2[i][:cross_location])
				p_list = [parent1[i], parent2[i]]
				#隨機生成一數字，用以決定是否進行mutation
				for i in range(len(p_list)):
					z3 = random.uniform(0,1)
					if z3 < self.mr:
						#決定要mutate的數字
						z4 = random.uniform(0,1)
						temp_location = z4*(len(p_list[i])-1)
						mutation_location = 0 if temp_location < 0.5 else math.ceil(temp_location)
						p_list[i][mutation_location] = 0 if p_list[i][mutation_location] == 1 else 1
				child_1.append(p_list[0])
				child_2.append(p_list[1])
			else:
				child_1.append(parent1[i])
				child_2.append(parent2[i])
		return child_1,child_2

def main():
	
	T, product_size, item_size =  (1, 4, 3)
	demand, bom = MRP_0301_hsu.data_gen(T, product_size, item_size)
	
	ga = GeneticAlgorithm()
	print(ga.N, ga.D, ga.B)
	pop_bin = ga.generatePopulation()
	pop_dec = []
	for i in range(ga.N):
		chrom_rv = []
		for j in range(ga.D):
			chrom_rv.append(ga.B2D(pop_bin[i][j]))
		pop_dec.append(chrom_rv)
	fitness = ga.fun_2(pop_dec, demand, bom)
	
	best_fitness = min(fitness)
	arr = fitness.index(best_fitness)
	best_dec = pop_dec[arr]
	
	best_rvlist = []
	best_valuelist = []

	it = 0
	while it < ga.max_iter:
		
		print("\n>> Iteration %d" %(it+1))
		
		Parents_list = ga.Selection(ga.n, pop_bin, fitness)
		Offspring_list = []
		for i in range(int((ga.N-ga.n)/2)):
			candidate = [Parents_list[random.randint(0,len(Parents_list)-1)] for i in range(2)]
			after_cr_mu = ga.Crossover_Mutation(candidate[0], candidate[1])
			offspring1, offspring2 = after_cr_mu[0], after_cr_mu[1]
			Offspring_list.append(offspring1)
			Offspring_list.append(offspring2)

		final_bin = Parents_list + Offspring_list
		final_dec = []
		for i in range(ga.N):
			rv = []
			for j in range(ga.D):
				rv.append(ga.B2D(final_bin[i][j]))
			final_dec.append(rv)

		# Final fitness
		final_fitness = ga.fun_2(final_dec, demand, bom)


		#Take the best value in this iteration
		smallest_fitness = min(final_fitness)
		index = final_fitness.index(smallest_fitness)
		smallest_dec = final_dec[index]

		#Store the best fitness in the list
		best_rvlist.append(smallest_dec)
		best_valuelist.append(smallest_fitness)

		#Parameters back to the initial
		pop_bin = final_bin 
		pop_dec = final_dec
		fitness = final_fitness

		it += 1
	
	#Store best result
	every_best_value = []
	every_best_value.append(best_valuelist[0])
	for i in range(ga.max_iter-1):
		if every_best_value[i] >= best_valuelist[i+1]:
			every_best_value.append(best_valuelist[i+1])

		elif every_best_value[i] <= best_valuelist[i+1]:
			every_best_value.append(every_best_value[i])

	print('The best fitness: ', min(best_valuelist))
	best_index = best_valuelist.index(min(best_valuelist))
	print('Arrival list: ')
	print(best_rvlist[best_index])

	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(every_best_value,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
	
if __name__ == '__main__':
	main()