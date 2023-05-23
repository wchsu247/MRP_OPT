import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from numpy.random import randint
from numpy.random import rand

# index setting
m_limit = 10000
total_count = 0
c = []
spm_best = [0,9999]

# --------------------------------------------------------------------------------------------------------------------------#
def simulation(s, S, round):
    global total_count, m_limit
    infeasible_counter = 0
    ave_cost = 0
    r = round
    c = 0
    while (r > 0 and total_count <= m_limit):
        # print(total_count)
        total_count = total_count + 1
        if total_count == m_limit:
            # print (total_count)
            return 0
        # 預設條件
        c = c + 1
        Days = 14
        h = 1
        f = 36
        c = 2
        flag = 0
        flag2 = 0
        op = 0
        total_demand = 0
        lose = 0
        Total_cost = 0
        i = 0
        Demand = []
        SI = []
        EI = []
        OR = []
        IP = []
        OP = []
        LD = []
        PC = []
        l = []
        # 產生 Dt~Exp(μ) 亂數
        for i in range(Days):
            μ = 100
            D = random.expovariate(1 / μ)
            D = int(D)
            Demand.append(D)
        # 變數歸零
        for i in range(10 * Days):
            OR.append(0)
            OP.append(0)
            LD.append("-")
            l = 0
        for i in range(Days):
            total_demand += Demand[i]
            flag2 = 0
            if i == 0:
                SI.append(s)
                ei = SI[i] + OR[i] - Demand[i]
                EI.append(ei)
                IP.append(ei)
                if ei < 0:
                    PC.append(0)
                    lose -= ei
                else:
                    PC.append(h * ei)
            else:
                SI.append(ei)
                ei = SI[i] + OR[i] - Demand[i]
                EI.append(ei)
                ip = IP[i - 1] + op - Demand[i]
                IP.append(ip)
                op = 0
                if flag == 1:
                    op = S - IP[i]
                    # 產生 l~Poi(theta) 亂數
                    ll = np.random.poisson(6, 1)
                    l = ll[0]
                    OR[i + l + 1] = op
                    flag = 0
                    flag2 = 1
                    if ei < 0:
                        PC.append(f + op * c)
                        lose -= ei
                    else:
                        PC.append(ei + f + op * c)
                else:
                    if ei < 0:
                        PC.append(0)
                        lose -= ei
                    else:
                        PC.append(h * ei)
                    l = "-"
            if IP[i] < s and flag2 == 0:
                flag = 1
            Total_cost += PC[i]
        if 1 - lose / total_demand < 0.9:
            continue
        r = r - 1
        ave_cost = ave_cost + Total_cost / Days
        infeasible_counter = infeasible_counter + 1
    if infeasible_counter < round:
        return 0
    return ave_cost / round
# --------------------------------------------------------------------------------------------------------------------------#
def simu_spm(s, S, round, Demand, lead_time):
    global total_count, m_limit
    infeasible_counter = 0
    ave_cost = 0
    r = round
    c = 0
    while (r > 0 and total_count <= m_limit):
        # print(total_count)
        total_count = total_count + 1
        if total_count == m_limit:
            # print (total_count)
            return 0
        # 預設條件
        c = c + 1
        Days = 14
        h = 1
        f = 36
        c = 2
        flag = 0
        flag2 = 0
        op = 0
        total_demand = 0
        lose = 0
        Total_cost = 0
        i = 0
        SI = []
        EI = []
        OR = []
        IP = []
        OP = []
        LD = []
        PC = []
        l = []
        # 變數歸零
        for i in range(10 * Days):
            OR.append(0)
            OP.append(0)
            LD.append("-")
            l = 0
        for i in range(Days):
            total_demand += Demand[i]
            flag2 = 0
            if i == 0:
                SI.append(s)
                ei = SI[i] + OR[i] - Demand[i]
                EI.append(ei)
                IP.append(ei)
                if ei < 0:
                    PC.append(0)
                    lose -= ei
                else:
                    PC.append(h * ei)
            else:
                SI.append(ei)
                ei = SI[i] + OR[i] - Demand[i]
                EI.append(ei)
                ip = IP[i - 1] + op - Demand[i]
                IP.append(ip)
                op = 0
                if flag == 1:
                    op = S - IP[i]    
                    OR[i + lead_time[i] + 1] = op
                    flag = 0
                    flag2 = 1
                    if ei < 0:
                        PC.append(f + op * c)
                        lose -= ei
                    else:
                        PC.append(ei + f + op * c)
                else:
                    if ei < 0:
                        PC.append(0)
                        lose -= ei
                    else:
                        PC.append(h * ei)
                    l = "-"
            if IP[i] < s and flag2 == 0:
                flag = 1
            Total_cost += PC[i]
        if 1 - lose / total_demand < 0.9:
            return 1200
        r = r - 1
        ave_cost = ave_cost + Total_cost / Days
        infeasible_counter = infeasible_counter + 1
    if infeasible_counter < round:
        return 0
    return ave_cost / round
# --------------------------------------------------------------------------------------------------------------------------#
def FDSA(s, S, f_ave_cost):
    global total_count, m_limit, c
    g_round = 4
    c_round = 10
    s_limit = 650
    k = 1
    cost = []
    theta_s = s
    theta_S = S
    ξ = 1
    A = 1
    cost.append(f_ave_cost)
    c.append(total_count)
    while (total_count < m_limit):
        a = A / (1 + k)
        cc = 1 / ((1 + k) ** (1 / 6))
        g = (simulation(theta_s + cc * ξ, theta_S + cc * ξ, g_round) - simulation(theta_s - cc * ξ, theta_S - cc * ξ,
                                                                                  g_round)) / (2 * cc)
        theta_s = max(theta_s - a * g, s_limit)
        theta_S = theta_S - a * g
        k = k + 1
        ave_cost = simulation(theta_s, theta_S, c_round)

        if f_ave_cost > ave_cost and ave_cost != 0:
            f_ave_cost = ave_cost
            best_s = theta_s
            best_S = theta_S
        if total_count < m_limit and ave_cost != 0 and ave_cost != 0.0:
            c.append(total_count)
            cost.append(ave_cost)
        else:
            cost.pop()
            c.pop()
    print('FDSA Strategy', [best_s, best_S], '=', f_ave_cost)
    return cost
# --------------------------------------------------------------------------------------------------------------------------#
def SPSA(s, S, f_ave_cost):
    global total_count, m_limit, c
    cost = []
    k = 1
    g_round = 4
    c_round = 10
    theta_s = s
    theta_S = S
    A = 1
    cost.append(f_ave_cost)
    best_s = 1000
    best_S = 2000
    c.append(total_count)
    while (total_count < m_limit):
        a = A / (1 + k)
        delta = 2 * np.random.binomial(1, .5, 1) - 1
        ck = 1 / ((1 + k) ** (1 / 6))
        g = (simulation(theta_s + ck * delta[0], theta_S + ck * delta[0], g_round) - simulation(theta_s - ck * delta[0],
                                                                                                theta_S - ck * delta[0],
                                                                                                g_round)) / (
                    2 * ck * delta[0])
        theta_s = theta_s - a * g
        theta_S = theta_S - a * g
        k = k + 1
        ave_cost = simulation(theta_s, theta_S, c_round)
        if f_ave_cost > ave_cost and ave_cost != 0:
            f_ave_cost = ave_cost
            best_s = theta_s
            best_S = theta_S
        if total_count < m_limit and ave_cost != 0 and ave_cost != 0.0:
            c.append(total_count)
            cost.append(ave_cost)
        else:
            cost.pop()
            c.pop()
    print('SPSA Strategy', [best_s, best_S], '=', f_ave_cost)
    return cost
# --------------------------------------------------------------------------------------------------------------------------#
class ga:
    # objective function
    def objective(x, fixed_rv, Demand, lead_time):
        if fixed_rv == 0:
            round = 10
            return simulation(x[0], x[1], round)
        else:
            return simu_spm(x[0], x[1], 1, Demand, lead_time)
    
    # decode bitstring to numbers
    def decode(bounds, n_bits, bitstring):
        decoded = list()
        largest = 2 ** n_bits
        for i in range(len(bounds)):
            # extract the substring
            start, end = i * n_bits, (i * n_bits) + n_bits
            substring = bitstring[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
            # store
            decoded.append(value)
        return decoded

    # tournament selection(隨機比較，輸的被贏的取代)
    def selection(pop, scores, k=3):
        # first random selection
        selection_ix = randint(len(pop))
        for ix in randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    # crossover two parents to create two children
    def crossover(p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # mutation operator
    def mutation(bitstring, r_mut):
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]

    # genetic algorithm
    def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, fixed_rv, Demand, lead_time):
        global c,spm_best
        cost = []
        # n_pop組解的基因序列
        pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
        best, best_eval = 0, objective(ga.decode(bounds, n_bits, pop[0]),fixed_rv, Demand, lead_time)
        for gen in range(n_iter):
            # decode population
            decoded = [ga.decode(bounds, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective(d, fixed_rv, Demand, lead_time) for d in decoded]
            group_leader = min(scores)
            if group_leader == 0:
                break
            if fixed_rv == 0:
                cost.append(group_leader)
                c.append(total_count)

            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    
            # select parents(經隨機兩兩比較，挑選較好的n_pop組基因序列)
            selected = [ga.selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for cc in ga.crossover(p1, p2, r_cross):
                    # mutation
                    ga.mutation(cc, r_mut)
                    # store for next generation
                    children.append(cc)
            # replace population
            pop = children
        if best == 0:
            return 0
        decoded = ga.decode(bounds, n_bits, best)
        if fixed_rv==0:
            print('GA Strategy  ', (decoded), '=', best_eval)
            return cost
        else:
            cost = objective(decoded,0,0,0)
            if cost < spm_best[1] and cost != 0:
                spm_best = [decoded,cost]
            return cost


def GA(fixed_rv, Demand, lead_time):
    global m_limit
    # define range for input
    bounds = [[800, 1200], [1600, 2200]]
    # define the total iterations
    if fixed_rv == 0:
        n_iter = m_limit
    else:
        n_iter = 60
    n_bits = 16
    n_pop = 6
    r_cross = 0.9
    r_mut = 1.0 / (float(n_bits) * len(bounds))
    # perform the genetic algorithm search
    cost = ga.genetic_algorithm(ga.objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, fixed_rv, Demand, lead_time)
    return cost
# --------------------------------------------------------------------------------------------------------------------------#
def rsm(s, S, f_ave_cost):
    global total_count, m_limit, c
    cost = []
    cost.append(f_ave_cost)
    c.append(total_count)
    r = 10
    best_cost = 9999
    x1 = []
    x2 = []
    y = []
    center = [s, S]
    best_s = s
    best_S = S
    cx1 = [-1, -1, 1, 1, 0, 0, 0]
    cx2 = [-1, 1, -1, 1, 0, 0, 0]
    delta_x1 = 5
    delta_x2 = 5
    while total_count < m_limit:
        x1 = []
        x2 = []
        y = []
        for i in cx1:
            x1.append(center[0] + i * delta_x1)
        for i in cx2:
            x2.append(center[1] + i * delta_x2)
        for i in range(0, len(cx1)):
            y.append(simulation(x1[i], x2[i], r))
        if min(y) == 0:
            break
        series_dict = {'s': cx1, 'S': cx2, 'cost': y}
        df = pd.DataFrame(series_dict)
        X = df[['s', 'S']]
        Y = df['cost']
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        if abs(regr.coef_[0]) > abs(regr.coef_[1]):
            if regr.coef_[0] > 0:
                move_cx1 = 1
            else:
                move_cx1 = -1
            move_x1 = move_cx1 * delta_x1
            move_cx2 = regr.coef_[1] / regr.coef_[0] * move_cx1
            move_x2 = min(move_cx2 * delta_x2, 40)
        else:
            if regr.coef_[1] > 0:
                move_cx2 = 1
            else:
                move_cx2 = -1
            move_cx2 = 1
            move_x2 = move_cx2 * delta_x2
            move_cx1 = regr.coef_[0] / regr.coef_[1] * move_cx2
            move_x1 = move_cx1 * delta_x1
        current = simulation(center[0] - move_x1, center[1] - move_x2, r)
        if total_count >= m_limit:
            break
        center[0] = center[0] - move_x1
        center[1] = center[1] - move_x2
        while 1:
            next_ = simulation(center[0] - move_x1, center[1] - move_x2, r)
            if next_ > current or next_ == 0:
                break
            else:
                current = next_
                best_s = center[0]
                best_S = center[1]
                center[0] = center[0] - move_x1
                center[1] = center[1] - move_x2
        if total_count >= m_limit:
            break
        c.append(total_count)
        cost.append(current)
        if current < best_cost and total_count < m_limit:
            best_center = [best_s, best_S]
            best_cost = current
    print('RSM Strategy ', best_center, '=', best_cost)
    return cost
# --------------------------------------------------------------------------------------------------------------------------#
def spm(s, S):
    global total_count, m_limit, c
    Days = 14
    fixed_rv=1
    cost = []
    while (total_count < m_limit):
        Demand = []
        lead_time = []
        for i in range(Days):
            mu = 100
            D = random.expovariate(1 / mu)
            D = int(D)
            ll = np.random.poisson(6, 1)
            Demand.append(D)
            lead_time.append(ll[0])
        # print(simu_spm(s, S, 1, Demand, lead_time))
        temp = GA(fixed_rv, Demand, lead_time)
        if temp != 0:
            cost.append(temp)
            c.append(total_count)
    return cost
# --------------------------------------------------------------------------------------------------------------------------#
def main():
    global total_count, c, spm_best
    # two original strategys
    s = 1000
    S = 2000
    f = simulation(s, S, 10)
    print('Oringinal Strategy', [s, S], '=', f)
    total_count = 0
    c.clear()
    cost1 = FDSA(s, S, f)
    x1 = np.array(c)
    y1 = np.array(cost1)
    total_count = 0
    c.clear()
    # # --------------------------------------------------------------------------------------------------------------------------#
    cost2 = SPSA(s, S, f)
    x2 = np.array(c)
    y2 = np.array(cost2)
    # # --------------------------------------------------------------------------------------------------------------------------#
    total_count = 0
    c.clear()
    cost3 = GA(Demand=0, lead_time=0,fixed_rv=0)
    x3 = np.array(c)
    y3 = np.array(cost3)
    # --------------------------------------------------------------------------------------------------------------------------#
    total_count = 0
    c.clear()
    cost4 = rsm(s, S, f)
    x4 = np.array(c)
    y4 = np.array(cost4)
    # --------------------------------------------------------------------------------------------------------------------------#
    total_count = 0
    c.clear()
    cost5 = spm(s, S)
    print("SPM Strategy ",spm_best[0],"=",spm_best[1])
    x5 = np.array(c)
    y5 = np.array(cost5)
    # --------------------------------------------------------------------------------------------------------------------------#
    # result plot
    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2, 'k')
    plt.plot(x3, y3, 'b')
    plt.plot(x4, y4, 'm')
    plt.plot(x5, y5, 'g')

    plt.legend(['FDSA', 'SPSA', 'GA', 'RSM', 'SPM'])
    plt.xlabel('Measurements')
    plt.ylabel('Average Costs')
    plt.title('Inventory Problem')
    plt.show()
    #
    plt.plot(x3, y3, 'b')
    plt.plot(x5, y5, 'g')
    plt.legend(['GA','SPM'])
    plt.xlabel('Measurements')
    plt.ylabel('Average Costs')
    plt.title('Inventory Problem')
    plt.show()
    #
    data1 = np.array(cost1)
    d1 = pd.Series(data1)
    data2 = np.array(cost2)
    d2 = pd.Series(data2)
    plt.plot(d1.rolling(50).mean(), 'r')
    plt.plot(d2.rolling(50).mean(), 'k')
    plt.legend(['FDSA', 'SPSA'])
    plt.xlabel('Measurements')
    plt.ylabel('Average Costs')
    plt.title('Inventory Problem (MA)')
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    main()