import matplotlib.pyplot as plt
import numpy as np
problem = 'rosenbrock'
script_dir = "/home/pb482/BOPartial/Clean-code"
algo = ['NoCL','CLMAX','CLMIN','CLMEAN','TS','RAND']
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
plt.errorbar(x, mean_progress[0], std_progress[0] , marker='o',label='NoCL')
plt.errorbar(x, mean_progress[1], std_progress[1] , marker='o',label='CLMAX')
plt.errorbar(x, mean_progress[2], std_progress[2] , marker='o',label='CLMIN')
plt.errorbar(x, mean_progress[3], std_progress[3], marker='o',label='CLMEAN')
plt.errorbar(x, mean_progress[4], std_progress[4] , marker='o',label='TS')
plt.errorbar(x, mean_progress[5], std_progress[5] , marker='o',label='RAND')
plt.title("Progress Curve_"+problem)
plt.xlabel("Iteration")
plt.ylabel("Obj func val")
plt.legend(loc='lower right', shadow=True, fontsize='x-large')
fig_file = '/home/pb482/BOPartial/Clean-code/Comparison/'+problem+'_progress-plot-compare-NoCL-CL3-TS-RAND-Botorch.png'
plt.savefig(fig_file)
plt.close()
