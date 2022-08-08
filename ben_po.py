import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use(['science','ieee'])

def annual_cost(t:float, r:float = 0.02):
    return r/(1-1/(1+r)**t)

def master(dict_cut: dict, conti_mp: bool = True, n_S: int = 366, 
           c_pv: float = 700*(annual_cost(25) + 0.005), 
           c_es: float = 260*(annual_cost(15) + 0.005), 
           c_inv: float = 75*(annual_cost(15) + 0.005),
           P_max_pv: float = 50, E_max_es: float = 50, 
           P_max_inv: float = 30, P_min_pv: float = 1, E_min_es: float = 1,
           P_min_inv: float = 1) -> list:
    '''

    Parameters
    ----------
    Unit in kW, kWh, EUR/kw, EUR/kWh, hour.
    
    dict_cut : dict
        for i th subproblem: ndarray
            obj, lam_pos, lam_neg, P_pv, ... ES, inv
        (1)
        (2)
        ...
        (k)
    '''
    
    cap_pv_module = 0.3 # kW
    cap_inv_module = 2 # kW
    m = gp.Model("master")
    m.Params.LogToConsole = 0  
    P_pv = m.addVar(lb = P_min_pv, ub = P_max_pv, vtype = GRB.CONTINUOUS,
                    name = "CapPV")
    E_es = m.addVar(lb = E_min_es, ub = E_max_es, vtype = GRB.CONTINUOUS,
                    name = "EneES") 
    P_inv = m.addVar(lb = P_min_inv, ub = P_max_inv, vtype = GRB.CONTINUOUS,
                    name = "CapINV")
    if not conti_mp:
        n_pv = m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.INTEGER,
                    name = "no.PV")
        n_inv = m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.INTEGER,
                    name = "no.INV")
        m.addConstr(P_pv == cap_pv_module * n_pv)
        m.addConstr(P_inv == cap_inv_module * n_inv)
     
    LBD = m.addVars(n_S, lb = -4E4, ub = GRB.INFINITY, 
                    vtype = GRB.CONTINUOUS, name = "optimalityCut") 
    # ===================== optimality cuts ==================================
    for s in range(n_S):
        ndarr_cut = dict_cut[f"sce{s}"]
        n_cut = ndarr_cut.shape[0]
        m.addConstrs(LBD[s]>= ndarr_cut[i,0]
                     +(ndarr_cut[i,1]-ndarr_cut[i,2])*(ndarr_cut[i,3]-P_pv)
                     +(ndarr_cut[i,4]-ndarr_cut[i,5])*(ndarr_cut[i,6]-E_es)
                     +(ndarr_cut[i,7]-ndarr_cut[i,8])*(ndarr_cut[i,9]-P_inv)
                     for i in range(n_cut))
    # ======================== objective =====================================
    obj = c_pv * P_pv + c_inv * P_inv + c_es * E_es + quicksum(LBD)
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    inv_cost = c_pv * P_pv.X + c_inv * P_inv.X + c_es * E_es.X
    # print([LBD[i].X for i in range(n_S)])
    return [P_pv.X, E_es.X, P_inv.X, m.objVal, inv_cost]
  
   
def subproblem_s(P_pv_para: float, E_es_para: float, P_inv_para: float, 
                 weight_s: float, norm_pv_gen_s: np.array, p_d_s: np.array, 
                 T_es: float = 1/0.9, n_T: int = 96, eta_ch: float = 0.95, 
                 eta_dis:float = 0.95, epsilon_0: float = 0.5, 
                 epsilon_min: float = 0.1, epsilon_max: float = 0.9, 
                 c_1:float=0.022 * 0.25, c_2:float=0.22 * 0.25, 
                 c_3:float=-0.007 * 0.25, c_4:float=0.07*0.25,
                 p_house_max: float = 10) -> list:
    eta = 0.5 * (eta_ch + 1/eta_dis)
    # ======================= variables ======================================
    m = gp.Model("subproblem_s")   
    m.Params.LogToConsole = 0    
    P_pv = m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS,
                    name = "CapPV")
    E_es = m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS,
                    name = "EneES") 
    P_inv = m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS,
                    name = "CapINV")
    p_pv = m.addVars(n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "PVGen")
    p_ch = m.addVars(n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "EScharge")
    p_dis = m.addVars(n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "ESdischarge")  
    E_es_lb = m.addVars(n_T+1, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "ESenergyLowBound")
    E_es_ub = m.addVars(n_T+1, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "ESenergyUppBound")
    p_buy = m.addVars(n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "powerBought")
    p_sell = m.addVars(n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "powerSold") 
    
    # ======================= constraints ====================================
    # coupling constraint: replicating variables
    cons_pv = m.addConstr(P_pv <= P_pv_para)
    cons_pvn = m.addConstr(-P_pv <= -P_pv_para)
    cons_es = m.addConstr(E_es <= E_es_para)
    cons_esn = m.addConstr(-E_es <= -E_es_para)
    cons_inv = m.addConstr(P_inv <= P_inv_para)
    cons_invn = m.addConstr(-P_inv <= -P_inv_para)
    # operating constraints
    m.addConstrs(p_pv[t] <= P_pv * norm_pv_gen_s[t] for t in range(n_T))
    m.addConstrs(p_ch[t] + p_dis[t] <= 1/T_es * E_es for t in range(n_T))
    m.addConstr(E_es_lb[0] == epsilon_0 * E_es)
    m.addConstr(E_es_ub[0] == epsilon_0 * E_es)
    m.addConstrs(E_es_lb[t+1] == E_es_lb[t] + eta_ch * p_ch[t] 
                 - 1/eta_dis * p_dis[t] for t in range(n_T))
    m.addConstrs(E_es_ub[t+1] == E_es_ub[t] + eta * (p_ch[t]-p_dis[t]) 
                 for t in range(n_T))  
    m.addConstrs(E_es_lb[t] >= epsilon_min * E_es for t in range(n_T+1))
    m.addConstrs(E_es_ub[t] <= epsilon_max * E_es for t in range(n_T+1))
    m.addConstrs(p_pv[t] + p_dis[t] - p_ch[t] <= P_inv for t in range(n_T))
    m.addConstrs(p_pv[t] + p_dis[t] - p_ch[t] >= -P_inv for t in range(n_T))
    m.addConstrs(p_pv[t] + p_dis[t] - p_ch[t] - p_d_s[t] == p_sell[t]
                 -p_buy[t] for t in range(n_T))
    m.addConstrs(p_sell[t] + p_buy[t] <= p_house_max for t in range(n_T)) 
    
    # ======================== objective =====================================
    cost_power =  weight_s * quicksum(c_1*p_buy[t]*p_buy[t] + c_2*p_buy[t] 
            - c_3*p_sell[t]*p_sell[t] - c_4*p_sell[t] for t in range(n_T))
    m.setObjective(cost_power, GRB.MINIMIZE)
    m.optimize()
    
    return [m.objVal,abs(cons_pv.Pi),abs(cons_pvn.Pi),P_pv.X,
                     abs(cons_es.Pi),abs(cons_esn.Pi),E_es.X,
                     abs(cons_inv.Pi),abs(cons_invn.Pi),P_inv.X]
    

def portf_opt(dict_in: dict, epsilon: float = 0.1) -> dict: # 0.1 eur gap
    # ====================== Parameters ======================================
    conti_mp = dict_in["conti_mp"]
    weight = dict_in["weight"] # weights of scenarios
    n_S = dict_in["n_S"] # no. of scenarios
    norm_pv_gen = dict_in["norm_pv_gen"] # numpy array of n_S * n_T
    p_d = dict_in["p_d"] # numpy array of n_S * n_T
    
    # ======================= iterations =====================================
    uppBound = np.Infinity
    lowBound = -np.Infinity
    list_inv = []
    list_gap = []
    dict_cut = {}
    for s in range(n_S):
        dict_cut[f"sce{s}"] = np.ndarray((0, 10)) # para for opt cuts: 3*3+1
    k = 0
    start_time = time.time()
    while uppBound - lowBound > epsilon:
        k += 1
        # ============ solve master problem ==================================
        list_mp = master(dict_cut, conti_mp = conti_mp, n_S = n_S)
        lowBound = list_mp[-2] #  second last element: obj of MP
        list_inv.append(list_mp[0:3])
        # ============ solve subproblems =====================================
        uppBound_cur = list_mp[-1]
        for s in range(n_S):
            # print(s)
            list_sp = subproblem_s(P_pv_para=list_mp[0], E_es_para=list_mp[1],
                 P_inv_para=list_mp[2], weight_s=weight[s], 
                 norm_pv_gen_s = norm_pv_gen[s,:], p_d_s = p_d[s,:])
            dict_cut[f"sce{s}"] = np.append(dict_cut[f"sce{s}"], 
                                [np.array(list_sp)], axis = 0)
            uppBound_cur += list_sp[0]
        uppBound = min(uppBound, uppBound_cur)
        print(f"Lower bound is {lowBound}, upper bound is {uppBound}.")
        print(f"Gap at iteration {k} is {uppBound - lowBound, (uppBound - lowBound)/lowBound*100}%.")
        list_gap.append(uppBound - lowBound)
    end_time = time.time()   
    # =========================== results ====================================
    dict_out = {}
    dict_out["P_pv"] = list_mp[0]
    dict_out["E_es"] = list_mp[1]
    dict_out["P_inv"] = list_mp[2]
    dict_out["list_gap"] = list_gap
    dict_out["objVal"] = uppBound
    dict_out["time"] = end_time - start_time
    dict_out["list_inv"] = list_inv
    return dict_out


if __name__ == '__main__':
    dir_data = "data\\data.xlsx"
    df_pvgen = pd.read_excel(dir_data, sheet_name = "gen_normalized")
    df_load = pd.read_excel(dir_data, sheet_name = "load_normalized")
    nparr_pvgen = df_pvgen.to_numpy()
    nparr_load = df_load.to_numpy()
    nparr_pvgen_reshape = np.reshape(nparr_pvgen, (-1, 96))
    nparr_load_reshape = np.reshape(nparr_load, (-1, 96))
    n_S = 366
    dict_in = {}
    dict_in["conti_mp"] = True
    # Pieter van Eig, End-user economics of PV-coupled residential battery 
    # systems in the Netherlands, master's thesis, Universiteit Utrecht, 2021
    dict_in["n_S"] = n_S 
    dict_in["norm_pv_gen"] = nparr_pvgen_reshape
    dict_in["p_d"] = 4.9 * nparr_load_reshape
    dict_in["weight"] = np.ones(n_S)
    dict_out = portf_opt(dict_in)
    list_gap = dict_out["list_gap"]
    plt.plot(range(1,len(list_gap)), list_gap[1:], label="366")
    plt.yscale('log')
    plt.legend()
    plt.xticks(range(len(list_gap)))
    plt.xlabel("No. of iterations")
    plt.ylabel("Optimality gap")
    plt.show()
