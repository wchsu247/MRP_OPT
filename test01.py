import numpy as np
import matplotlib.pyplot as plt
import simulation_model_new as simulation_model

# 給定隨機種子，使每次執行結果保持一致
np.random.seed(1)
    
def getdata(n):
    # n為產生資料量
    # x = np.arange(-5, 5.1, 10/(n-1))
    T, product_size, item_size =  (52, 4, 3)
    x = np.random.randint(2, 50, size=(T, item_size))
    
    # 給定一個固定的參數，再加上隨機變動值作為雜訊，其變動值介於 +-10 之間
    # y = 3*x + 2 + (np.random.rand(len(x))-0.5)*20
    y = simulation_model.ans_fun(x, T, product_size, item_size)
    
    return x, y

def plot_error(x, y):
    a = np.arange(-10, 16, 1)
    b = np.arange(-10, 16, 1)
    mesh = np.meshgrid(a, b)
    sqr_err = 0
    for xs, ys in zip(x, y):
        sqr_err += ((mesh[0]*xs + mesh[1]) - ys) ** 2
    loss = sqr_err/len(x)
    
    plt.contour(mesh[0], mesh[1], loss, 20, cmap=plt.cm.jet)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.axis('scaled')
    plt.title('function loss')
    plt.colorbar()
    plt.show()

    sqr_err = 0
    for xs, ys in zip(x, y):
        sqr_err += ((mesh[0]*xs + mesh[1]) - ys) ** 2
    loss = sqr_err/len(x)
    
    plt.contour(mesh[0], mesh[1], loss, 20, cmap=plt.cm.jet)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.axis('scaled')
    plt.title('function loss')

class my_BGD:    
    def __init__(self, a, b, x, y, alpha):
        self.a = a
        self.b = b
        self.x = x
        self.y = y
        self.alpha = alpha
        
        self.a_old = a
        self.b_old = b
        
        self.loss = self.mse();
    
    # Loss function
    def mse(self):
        sqr_err = ((self.a*self.x + self.b) - self.y) ** 2
        return np.mean(sqr_err)
    
    def gradient(self):
        grad_a = 2 * np.mean((self.a*self.x + self.b - self.y) * (self.x))
        grad_b = 2 * np.mean((self.a*self.x + self.b - self.y) * (1))
        return grad_a, grad_b

    def update(self):
        # 計算梯度
        grad_a, grad_b = self.gradient()
        
        # 梯度更新
        self.a_old = self.a
        self.b_old = self.b
        self.a = self.a - self.alpha * grad_a
        self.b = self.b - self.alpha * grad_b
        self.loss = self.mse();

# =============================================== #
# 隨機產生一組資料
x, y = getdata(51)
print(x, y)
# 繪製誤差空間底圖
plot_error(x, y)

# =============================================== #

# 初始設定
alpha = 0.1

# 從 -9 開始，能看得更明顯
a = -9; b = -9

# 初始化
mlclass = my_BGD(a, b, x, y, alpha)

plt.plot(a, b, 'ro-')
plt.title('Initial, loss='+'{:.2f}'.format(mlclass.loss)+'\na='+
        '{:.2f}'.format(a)+', b='+'{:.2f}'.format(b))
# plt.show()

# 開始迭代
for i in range(1, 11):
    mlclass.update()
    print('iter='+str(i)+', loss='+'{:.2f}'.format(mlclass.loss))
    plt.plot((mlclass.a_old, mlclass.a), (mlclass.b_old, mlclass.b), 'ro-')
    plt.title('iter='+str(i)+', loss='+'{:.2f}'.format(mlclass.loss)+'\na='+
            '{:.2f}'.format(mlclass.a)+', b='+'{:.2f}'.format(mlclass.b))
    # plt.show()


from numpy import random
import numpy as np
x = random.poisson(lam=15, size=(5,5))
bom = np.random.randint(2, size=(5, 5))
demand = np.random.randint(8, 20, size=(5, 5))
print(x)
print(bom)
print(demand)