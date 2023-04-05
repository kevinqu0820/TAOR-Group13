#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 03:02:03 2023

@author: Kevin
"""

import numpy as np
import pandas as pd
import osmnx as ox
import pickle
import time
from VRPForm import loadInstance
from HGA import geneticAlgorithm, evaluate, optimalDecode


G = ox.load_graphml('./3_boroughs_data/network.osm.graphml')

Folders = []
cust_nums = np.arange(10,151,10)
for i in range(10):
    Folders.append('MD_cust' + str(cust_nums[i]))
for i in range(10):
    Folders.append('MDTW_cust' + str(cust_nums[i]))

use_TW = [0] * 10 + [1] * 10


Cost_d = []
Cost_c = []
Util_d = []
Util_c = []

final_routes_c = []
final_routes_d = []

start_time = time.time()

for i in range(len(Folders)):
    if use_TW[i] == 0:
        TW = False
    elif use_TW[i] == 1:
        TW = True
    print('Currently on:',Folders[i])
    vrp = loadInstance(Folders[i], TW = True, diff_companies=True)
    results_d, _, _ = geneticAlgorithm(100, 1000, 0.8, 0.6, 0.8, vrp, 4, G)
   
    vrp = loadInstance(Folders[i], TW = True, diff_companies=False)
    results_c, _, _ = geneticAlgorithm(100, 1000, 0.8, 0.6, 0.8, vrp, 4, G)

    chrom_c = results_c[-1]
    chrom_d = [sublist[-1] for sublist in results_d]

    routes_c = optimalDecode(chrom_c, vrp)
    all_routes_d = [optimalDecode(chrom, vrp) for chrom in chrom_d]
    routes_d = [routes for some_routes in all_routes_d for routes in some_routes]

    final_routes_c.append(routes_c)
    final_routes_d.append(routes_d)

    U_d, _, C_d = evaluate(routes_d, vrp, G)
    U_c, _, C_c = evaluate(routes_c, vrp, G)

    Util_d.append(U_d)
    Util_c.append(U_c)
    Cost_d.append(C_d)
    Cost_c.append(C_c)
    
    with open('routes1.pickle', 'wb') as f:
        pickle.dump({'decentralised': final_routes_d, 'centralised': final_routes_c}, f)
    print('Time Elapsed:', time.time() - start_time)
    
PoA_U = [c/d for c,d in zip(Util_c, Util_d)]
PoA_C = [d/c for c,d in zip(Cost_c, Cost_d)]

data = {
        'Instance' : Folders,
        'C(R_d)' : Cost_d,
        'C(R_c)' : Cost_c,
        'U(R_d)' : Util_d,
        'U(R_c)' : Util_c,
        'PoA_U' : PoA_U,
        'PoA_C' : PoA_C
        }
    
df = pd.DataFrame(data)
df.to_csv('Results_for_instances1.csv', index=False)



    
    


