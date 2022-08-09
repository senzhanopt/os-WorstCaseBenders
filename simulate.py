import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee'])
from sklearn.cluster import KMeans

model = 'ben' # functions to import, centralized or Benders
conti_mp = False # continuous or discrete master problems


def annual_cost(t:float, r:float = 0.02):
    return r/(1-1/(1+r)**t)
c_pv = 700*(annual_cost(25) + 0.005)
c_es = 260*(annual_cost(15) + 0.005) 
c_inv = 75*(annual_cost(15) + 0.005)

C1 = np.sqrt(c_pv**2+c_es**2+c_inv**2)
m = 3
beta_up = 2377
beta_down =- 701
delta = 1
d_max = 75.12
C2 = np.sqrt(m) * (beta_up-beta_down)/delta


if model == 'cen':
    from cen_po import portf_opt
else:
    from ben_po import portf_opt

dir_data = "data\\data.xlsx"
dir_fig = "fig\\"
df_pvgen = pd.read_excel(dir_data, sheet_name = "gen_normalized")
df_load = pd.read_excel(dir_data, sheet_name = "load_normalized")
nparr_pvgen = df_pvgen.to_numpy()
nparr_load = df_load.to_numpy()
nparr_pvgen_reshape = np.reshape(nparr_pvgen, (-1, 96))
nparr_load_reshape = np.reshape(nparr_load, (-1, 96))
nparr_pvgen_load = np.append(nparr_pvgen_reshape, nparr_load_reshape, axis = 1)

list_format = [['k', '-'],['r', '--'],['b', ':'],['g', '--'],['k', '--'],
               ['m', '--'],['r', '-.']]

dict_inv = {}
dict_gap = {}

for idx, n_S in enumerate([10,20,40,80,160,320,366]):
    dict_in = {}
    dict_in["conti_mp"] = conti_mp
    # Pieter van Eig, End-user economics of PV-coupled residential battery 
    # systems in the Netherlands, master's thesis, Universiteit Utrecht, 2021
    dict_in["n_S"] = n_S 
    # sklearn k clustering
    kmeans = KMeans(n_clusters=n_S, random_state=0).fit(nparr_pvgen_load)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_   
    dict_in["norm_pv_gen"] = centers[:,0:96]
    dict_in["p_d"] = 4.9 * centers[:,96:]
    dict_in["weight"] = [labels.tolist().count(s) for s in range(n_S)]
    dict_out = portf_opt(dict_in)
    print(f"Scenario {n_S}: PV {dict_out['P_pv']}kW, ES {dict_out['E_es']}kWh,\
          Inv {dict_out['P_inv']}kW, cost {dict_out['objVal']}EUR, \
              time {dict_out['time']}s.")
    print(f"{dict_out['P_pv']:.2f}/{dict_out['E_es']:.2f}/{dict_out['P_inv']:.2f}&{dict_out['objVal']:.2f}&{dict_out['time']:.2f}")
    if model == 'ben':
        dict_inv[f"{n_S}"] = dict_out["list_inv"]
        dict_gap[f"{n_S}"] = dict_out["list_gap"]

if model == 'ben':
    for idx, n_S in enumerate([10,20,40,80,160,320,366]):  
        list_gap = dict_gap[f"{n_S}"]
        plt.plot(range(1,len(list_gap)), list_gap[1:], label=f"{n_S} scenarios",
                    color=list_format[idx][0], linestyle=list_format[idx][1])
        plt.yscale('symlog')
    plt.xlabel("No. of iterations")
    plt.ylabel("Optimality gap (EUR)")
    plt.plot(range(2,20), [2*d_max*(C1+C2)/n**(1/3) for n in range(2,20)], 
             label = "worst-case", color = 'g', linestyle='-.')
    plt.xticks(np.arange(0,22, step = 2))
    plt.legend(ncol = 1, loc = "center right", prop={'size': 7})
    if dict_in["conti_mp"]:
        plt.savefig(dir_fig+"conti.pdf")
    else:
        plt.savefig(dir_fig+"discre.pdf")
    plt.show()
    
    
    list_marker = ['.', '+', '1', "2", "4", "d", "+"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, n_S in enumerate([10,40,80,366]): 
        list_inv = dict_inv[f"{n_S}"]
        ndarr_inv = np.array(list_inv)
        x = ndarr_inv[:,0]
        y = ndarr_inv[:,1]
        z = ndarr_inv[:,2]
        ax.scatter(x,y,z,label=f'{n_S} scenarios', s = 2.5, marker = list_marker[idx])
    ax.set_xlabel("PV (kW)",labelpad = -10)
    ax.set_ylabel("ES (kWh)",labelpad = -10)
    ax.set_zlabel("Inverter (kW)",labelpad = -12)
    ax.tick_params(axis='both', which='major', pad=-4.5)
    ax.legend(loc="upper left", prop={'size': 7})
    plt.tight_layout()
    if dict_in["conti_mp"]:
        plt.savefig(dir_fig+'conti_scatt.pdf',bbox_inches='tight')
    else:
        plt.savefig(dir_fig+'discre_scatt.pdf',bbox_inches='tight')
    plt.show()
