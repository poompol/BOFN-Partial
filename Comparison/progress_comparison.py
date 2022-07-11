import matplotlib.pyplot as plt
import numpy as np
problem = 'ackley'
script_dir = "/home/pb482/BOPartial/Clean-code"
algo = ['EIFN-NoCL_qEI','EIFN-CLMAX_qEI','EIFN-CLMIN_qEI','EIFN-CLMEAN_qEI',
        'EIFN-NoCL_TS','EIFN-CLMAX_TS','EIFN-CLMIN_TS','EIFN-CLMEAN_TS',
        'TSFN-NoCL_qEI','TSFN-CLMAX_qEI','TSFN-CLMIN_qEI','TSFN-CLMEAN_qEI',
        'TSFN-NoCL_TS','TSFN-CLMAX_TS','TSFN-CLMIN_TS','TSFN-CLMEAN_TS','TSWH','RAND']
no_iter = 20
no_rep = 10
progress = [[[] for j in range(no_rep)] for i in range(len(algo))]
mean_progress = []
std_progress = []
index_algo=0
for alg in algo:
    results_folder = script_dir + "/results/" + problem + "/" + alg + "/"
    for trial in range(1,no_rep+1):
        result_file = results_folder+ "best_obs_vals_"+str(trial)+".txt"
        f = open(result_file,'r')
        data = f.read().split('\n')
        data.pop()
        for y in data:
            progress[index_algo][trial-1].append(float(y))
        f.close()
    mean_progress.append(np.mean(progress[index_algo],axis=0))
    std_progress.append(np.std(progress[index_algo],axis=0))
    index_algo = index_algo+1

x = list(range(0,no_iter+1))
plt.errorbar(x, mean_progress[0], std_progress[0] , marker='o',label='EINo_qEI')
plt.errorbar(x, mean_progress[1], std_progress[1] , marker='o',label='EIMAX_qEI')
plt.errorbar(x, mean_progress[2], std_progress[2] , marker='o',label='EIMIN_qEI')
plt.errorbar(x, mean_progress[3], std_progress[3], marker='o',label='EIMEAN_qEI')
plt.errorbar(x, mean_progress[4], std_progress[4] , marker='o',label='EINo_TS')
plt.errorbar(x, mean_progress[5], std_progress[5] , marker='o',label='EIMAX_TS')
plt.errorbar(x, mean_progress[6], std_progress[6] , marker='o',label='EIMIN_TS')
plt.errorbar(x, mean_progress[7], std_progress[7], marker='o',label='EIMEAN_TS')
plt.errorbar(x, mean_progress[8], std_progress[8] , marker='o',label='TSNo_qEI')
plt.errorbar(x, mean_progress[9], std_progress[9] , marker='o',label='TSMAX_qEI')
plt.errorbar(x, mean_progress[10], std_progress[10] , marker='o',label='TSMIN_qEI')
plt.errorbar(x, mean_progress[11], std_progress[11], marker='o',label='TSMEAN_qEI')
plt.errorbar(x, mean_progress[12], std_progress[12] , marker='o',label='TSNo_TS')
plt.errorbar(x, mean_progress[13], std_progress[13] , marker='o',label='TSMAX_TS')
plt.errorbar(x, mean_progress[14], std_progress[14] , marker='o',label='TSMIN_TS')
plt.errorbar(x, mean_progress[15], std_progress[15], marker='o',label='TSMEAN_TS')
plt.errorbar(x, mean_progress[16], std_progress[16] , marker='o',label='TSWH')
plt.errorbar(x, mean_progress[17], std_progress[17] , marker='o',label='RAND')

plt.title("Progress Curve_"+problem)
plt.xlabel("Iteration")
plt.ylabel("Obj func val")
plt.legend(loc='lower right', shadow=True, fontsize='small', prop={'size': 6})
fig_file = '/home/pb482/BOPartial/Clean-code/Comparison/'+problem+'_progress-plot-compare-COMBI-EIFN-TSFN-qEI-TS-TSWH-RAND-Botorch.png'
plt.savefig(fig_file)
plt.close()
