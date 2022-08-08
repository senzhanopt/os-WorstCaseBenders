import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np
import time

def annual_cost(t:float, r:float = 0.02):
    return r/(1-1/(1+r)**t)

def portf_opt(dict_in: dict, c_pv: float = 700*(annual_cost(25) + 0.005), 
              c_es: float =260*(annual_cost(15) + 0.005), 
              c_inv: float = 75*(annual_cost(15) + 0.005), 
              P_max_pv: float = 50, E_max_es: float = 50, P_max_inv: float = 30,
              P_min_pv: float = 1, E_min_es: float = 1, P_min_inv: float = 1,              
                T_es: float = 1/0.9, n_T: int = 96, eta_ch: float = 0.95, 
                eta_dis:float = 0.95, epsilon_0: float = 0.5, 
                epsilon_min: float = 0.1, epsilon_max: float = 0.9, 
                c_1:float=0.022 * 0.25, c_2:float=0.22 * 0.25, 
                c_3:float=-0.007 * 0.25, c_4:float=0.07*0.25,
                p_house_max: float = 10) -> dict:
    '''

    Parameters
    ----------
    dict_in : dict
        Unit in kW, kWh, EUR/kw, EUR/kWh, hour.

    '''
    cap_pv_module = 0.3 
    cap_inv_module = 2
    # ====================== Parameters ======================================
    conti_mp = dict_in["conti_mp"]
    weight = dict_in["weight"] # weights of scenarios
    n_S = dict_in["n_S"] # no. of scenarios
    norm_pv_gen = dict_in["norm_pv_gen"] # numpy array of n_S * n_T
    p_d = dict_in["p_d"] # numpy array of n_S * n_T
    eta = 0.5 * (eta_ch + 1/eta_dis)
    start_time = time.time()
    # ======================= variables ======================================
    m = gp.Model("po")   
    m.Params.LogToConsole = 0     
    P_pv = m.addVar(lb = P_min_pv, ub = P_max_pv, vtype = GRB.CONTINUOUS,
                    name = "CapPV")
    E_es = m.addVar(lb = E_min_es, ub = E_max_es, vtype = GRB.CONTINUOUS,
                    name = "EneES") 
    P_inv = m.addVar(lb = P_min_inv, ub = P_max_inv, vtype = GRB.CONTINUOUS,
                    name = "CapINV")
    p_pv = m.addVars(n_S, n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "PVGen")
    p_ch = m.addVars(n_S, n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "EScharge")
    p_dis = m.addVars(n_S, n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "ESdischarge")  
    E_es_lb = m.addVars(n_S, n_T+1, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "ESenergyLowBound")
    E_es_ub = m.addVars(n_S, n_T+1, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "ESenergyUppBound")
    p_buy = m.addVars(n_S, n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "powerBought")
    p_sell = m.addVars(n_S, n_T, lb = 0.0, ub = GRB.INFINITY, 
                     vtype = GRB.CONTINUOUS, name = "powerSold") 
    if not conti_mp:
        n_pv = m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.INTEGER,
                    name = "no.PV")
        n_inv = m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.INTEGER,
                    name = "no.INV")
        m.addConstr(P_pv == cap_pv_module * n_pv)
        m.addConstr(P_inv == cap_inv_module * n_inv)
    
    # ======================= constraints ====================================
    m.addConstrs(p_pv[s,t] <= P_pv * norm_pv_gen[s,t] for s in range(n_S)
                 for t in range(n_T))
    m.addConstrs(p_ch[s,t] + p_dis[s,t] <= 1/T_es * E_es for s in range(n_S)
                 for t in range(n_T))
    m.addConstrs(E_es_lb[s,0] == epsilon_0 * E_es for s in range(n_S))
    m.addConstrs(E_es_ub[s,0] == epsilon_0 * E_es for s in range(n_S))
    m.addConstrs(E_es_lb[s,t+1] == E_es_lb[s,t] + eta_ch * p_ch[s,t] 
                 - 1/eta_dis * p_dis[s,t] 
                 for s in range(n_S) for t in range(n_T))
    m.addConstrs(E_es_ub[s,t+1] == E_es_ub[s,t] + eta * (p_ch[s,t]-p_dis[s,t]) 
                 for s in range(n_S) for t in range(n_T))  
    m.addConstrs(E_es_lb[s,t] >= epsilon_min * E_es for s in range(n_S)
                 for t in range(n_T+1))
    m.addConstrs(E_es_ub[s,t] <= epsilon_max * E_es for s in range(n_S)
                 for t in range(n_T+1))
    m.addConstrs(p_pv[s,t] + p_dis[s,t] - p_ch[s,t] <= P_inv 
                 for s in range(n_S) for t in range(n_T))
    m.addConstrs(p_pv[s,t] + p_dis[s,t] - p_ch[s,t] >= -P_inv 
                 for s in range(n_S) for t in range(n_T))
    m.addConstrs(p_pv[s,t] + p_dis[s,t] - p_ch[s,t] - p_d[s,t] == p_sell[s,t]
                 -p_buy[s,t] for s in range(n_S) for t in range(n_T))
    m.addConstrs(p_sell[s,t] + p_buy[s,t] <= p_house_max 
                 for s in range(n_S) for t in range(n_T))    
    
    # ======================== objective =====================================
    cost_inv = c_pv * P_pv + c_inv * P_inv + c_es * E_es
    cost_power =  quicksum(weight[s] * quicksum(c_1*p_buy[s,t]*p_buy[s,t] 
            + c_2*p_buy[s,t] - c_3*p_sell[s,t]*p_sell[s,t] - c_4*p_sell[s,t]
                              for t in range(n_T)) for s in range(n_S))
    m.setObjective(cost_inv + cost_power, GRB.MINIMIZE)
    m.optimize()
    end_time = time.time()
    # =========================== results ====================================
    dict_out = {}
    dict_out["P_pv"] = P_pv.X
    dict_out["P_inv"] = P_inv.X
    dict_out["E_es"] = E_es.X
    dict_out["p_sell"] = np.array([[p_sell[s,t].X for t in range(n_T)]
                                   for s in range(n_S)])
    dict_out["p_buy"] = np.array([[p_buy[s,t].X for t in range(n_T)]
                                   for s in range(n_S)]) 
    dict_out["p_ch"] = np.array([[p_ch[s,t].X for t in range(n_T)]
                                   for s in range(n_S)]) 
    dict_out["p_dis"] = np.array([[p_dis[s,t].X for t in range(n_T)]
                                   for s in range(n_S)]) 
    dict_out["objVal"] = m.objval
    dict_out["time"] = end_time - start_time
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
