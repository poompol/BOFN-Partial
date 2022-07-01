import matplotlib.pyplot as plt
import numpy as np
problem = 'rosenbrock'
script_dir = "/home/pb482/BOPartial/Clean-code"
algo = ['NoCL','CLMAX','CLMIN','CLMEAN','TS','RAND']
no_iter = 20
no_rep = 10
avg_runtime = []
index_algo=0
for alg in algo:
    results_folder = script_dir + "/results/" + problem + "/" + alg + "/"
    temp_time = 0
    for trial in range(1,no_rep+1):
        result_file = results_folder+ "runtimes/runtimes_"+str(trial)+".txt"
        f = open(result_file,'r')
        data = f.read().split('\n')
        data.pop()
        data = [float(x) for x in data]
        temp_time = temp_time + np.sum(data)
    avg_time = temp_time/no_rep
    avg_runtime.append(avg_time)
    f.close()
    index_algo = index_algo+1

print(f"Problem_"+problem)
print(f"Avg Running time NoCL: "+str(avg_runtime[0]/60)+" mins")
print(f"Avg Running time CLMAX: "+str(avg_runtime[1]/60)+" mins")
print(f"Avg Running time CLMIN: "+str(avg_runtime[2]/60)+" mins")
print(f"Avg Running time CLMEAN: "+str(avg_runtime[3]/60)+" mins")
print(f"Avg Running time TS: "+str(avg_runtime[4]/60)+" mins")
print(f"Avg Running time RAND: "+str(avg_runtime[5]/60)+" mins")

