import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

def vis(bl_ga, bl_spsa, bl_gsha, d, ir_target):

    random_number = np.random.randint(1000, 9999)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)
    #--------------------------------------------------------------------
    # subplot 1 - outliers only
    ax1.plot(bl_ga, 'r', label='GA')
    ax1.plot(bl_spsa, 'b', label='SPSA')
    ax1.plot(bl_gsha, 'c', label='GSHA')
    # for i in range(len(d) - 1):	ax1.axvline(x=d[i], c="r", ls="--", lw=1)
    ax1.axhline(y=ir_target, c="m", ls="--", lw=1)

    # subplot 2 - most of the data
    ax2.plot(bl_ga, 'r', label='GA')
    ax2.plot(bl_spsa, 'b', label='SPSA')
    ax2.plot(bl_gsha, 'c', label='GSHA')
    # for i in range(len(d) - 1):	ax2.axvline(x=d[i], c="r", ls="--", lw=1)
    ax2.axhline(y=ir_target, c="m", ls="--", lw=1)
    #--------------------------------------------------------------------
    ax1.set_ylim(7.55e10, 8.05e10) # 設定 y 軸範圍
    ax2.set_ylim(0.63e10, 0.85e10)   # 設定 y 軸範圍
    #--------------------------------------------------------------------
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.setp(ax1.get_xticklines(), visible=False)
    #--------------------------------------------------------------------
    g = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-g, +g), (-g, +g), **kwargs)        # top-left diagonal
    ax1.plot((1 - g, 1 + g), (-g, +g), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-g, +g), (1 - g, 1 + g), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - g, 1 + g), (1 - g, 1 + g), **kwargs)  # bottom-right diagonal
    #--------------------------------------------------------------------
    plt.xlabel("評估次數",fontsize = 15)
    plt.ylabel("總成本評估值",fontsize = 15)
    plt.legend(['GA', 'SPSA', 'GSHA'])
    #--------------------------------------------------------------------
    plt.savefig('c:/Users/MB608/Desktop/theis_MRP/theis_MRP/Reusult_Plot/' + str(random_number) + '.png')
    print(random_number)
    return