import numpy as np
import matplotlib.pyplot as plt

''' SPSA method 1
# Defining the seed to have same results
np.random.seed(42)

def polynomial(a, x):
	N = len(a)
	S = 0
	for k in range(N):
		S += a[k]*x**k
	return S

def Loss(parameters, X, Y):
# Predictions of our model
	Y_pred = polynomial(parameters, X)
	
	# mse (mean square error)
	L = ((Y_pred - Y)**2).mean()
	
	# Noise in range: [-5, 5]
	noise = 5*np.random.random()
	return L + noise

def grad(L, w, ck):
	
	# number of parameters
	p = len(w)
	
	# bernoulli-like distribution
	deltak = np.random.choice([-1, 1], size=p)
	
	# simultaneous perturbations
	ck_deltak = ck * deltak

	# gradient approximation
	DELTA_L = L(w + ck_deltak) - L(w - ck_deltak)

	return (DELTA_L) / (2 * ck_deltak)

def initialize_hyperparameters(alpha, lossFunction, w0, N_iterations):

	c = 1e-2 # a small number

	# A is <= 10% of the number of iterations
	A = N_iterations*0.1

	# order of magnitude of first gradients
	magnitude_g0 = np.abs(grad(lossFunction, w0, c).mean())

	# the number 2 in the front is an estimative of
	# the initial changes of the parameters,
	# different changes might need other choices
	a = 2*((A+1)**alpha)/magnitude_g0

	return a, A, c

# optimization algorithm
def SPSA(LossFunction, parameters, alpha=0.602,\
		gamma=0.101, N_iterations=int(1e5)):
	
	# model's parameters
	w = parameter

	a, A, c = initialize_hyperparameters(
	alpha, LossFunction, w, N_iterations)

	for k in range(1, N_iterations):

		# update ak and ck
		ak = a/((k+A)**(alpha))
		ck = c/(k**(gamma))

		# estimate gradient
		gk = grad(LossFunction, w, ck)

		# update parameters
		w -= ak*gk

	return w

# Y is the polynomial to be approximated
X = np.linspace(0, 10, 100)
Y = 1*X**2 - 4*X + 3

print(X)
print(Y)

noise = 3*np.random.normal(size=len(X))
Y += noise

# plot polynomial
plt.title("polynomial with noise")
plt.plot(X, Y, 'go')
plt.show()

# Initial parameters are randomly
# choosing in the range: [-10,10]
parameters = (2*np.random.random(3) - 1)*10

plt.title("Before training")

# Compare true and predicted values before
# training
plt.plot(X, polynomial(parameters, X), "bo")
plt.plot(X, Y, 'go')
plt.legend(["predicted value", "true value"])
plt.show()

# Training with SPSA
parameters = SPSA(LossFunction = lambda parameters: Loss(parameters, X, Y),
				parameters = parameters)

plt.title("After training")
plt.plot(X, polynomial(parameters, X), "bo")
plt.plot(X, Y, 'go')
plt.legend(["predicted value", "true value"])
plt.show()
'''


def cost(self, x, u): 
    dt = .1 if self.arm.DOF == 3 else .01
    next_x = self.plant_dynamics(x, u, dt=dt)
    vel_gain = 100 if self.arm.DOF == 3 else 10
    return (np.sqrt(np.sum((self.arm.x - self.target)**2)) * 1000 \
        + np.sum((next_x[self.arm.DOF:])**2) * vel_gain)

def spsa(self):
	# Step 1: Initialization and coefficient selection
	max_iters = 5
	converge_thresh = 1e-5

	alpha = 0.602 # from (Spall, 1998)
	gamma = 0.101
	a = .101 # found empirically using HyperOpt
	A = .193
	c = .0277

	delta_K = None
	delta_J = None
	u = np.copy(self.u) if self.u is not None \
			else np.zeros(self.arm.DOF)
	for k in range(max_iters):
		ak = a / (A + k + 1)**alpha
		ck = c / (k + 1)**gamma

		# Step 2: Generation of simultaneous perturbation vector
		# choose each component from a bernoulli +-1 distribution with
		# probability of .5 for each +-1 outcome.
		delta_k = np.random.choice([-1,1], size=(4,3), p=[.5, .5])

		# Step 3: Function evaluations
		inc_u = np.copy(u) + ck * delta_k
		cost_inc = self.cost(np.copy(state), inc_u)
		dec_u = np.copy(u) - ck * delta_k
		cost_dec = self.cost(np.copy(state), dec_u)

		# Step 4: Gradient approximation
		gk = np.dot((cost_inc - cost_dec) / (2.0*ck), delta_k)

		# Step 5: Update u estimate
		old_u = np.copy(u)
		u -= ak * gk

		# Step 6: Check for convergence
		if np.sum(abs(u - old_u)) < converge_thresh:
			break

