#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 03:03:03 2023

@author: Kevin
"""

import numpy as np
import pandas as pd
import ast
import os

def loadInstance(inst_folder, TW, diff_companies, main_instances_folder = "3_boroughs_data/instances"):
    inst_folder_path = os.path.join(main_instances_folder, inst_folder)
     # Check if the path is a directory
    if os.path.isdir(inst_folder_path):
        for cust_depot_file in os.listdir(os.path.join(main_instances_folder,inst_folder)):
            if cust_depot_file == "depots.csv":
                #read depo files into DataFrame
                df_depos = pd.read_csv(os.path.join(main_instances_folder, inst_folder,cust_depot_file))
            elif cust_depot_file == 'customers.csv':
                #read customer files into DataFrame
                df_custs = pd.read_csv(os.path.join(main_instances_folder, inst_folder,cust_depot_file))
                
    #variables
    xpos_customer = df_custs.loc[:, "x"]
    ypos_customer = df_custs.loc[:, "y"]
    demand_customer= df_custs.loc[:, "demand"]
    start_customer = df_custs.loc[:, "start"]
    end_customer = df_custs.loc[:, "end"]
    service_dura_customer = df_custs.loc[:, "service_dura"]
    company = df_custs.loc[:, "company"]
    osmid_customer = df_custs.loc[:, "osmid"]
    
    depot_vehicle = [i for i in range(0, len(df_depos)-1)]
    capacity_vehicle = df_depos.loc[:, "veh_cap"]
    max_dura_vehicle = df_depos.loc[:, "rou_lim"]
    osmid_depot = df_depos.loc[:, "osmid"]
    
    xpos_depot = df_depos.loc[:, "x"]
    ypos_depot = df_depos.loc[:, "y"]
    #capacity_depot = df_depos.loc[:, "capacity"]
    vehicle_number = df_depos.loc[:, "veh_num"]
    route_limit = df_depos.loc[:, "rou_lim"]
    companies_depots = df_depos.loc[:, "company"]
    
    vrp = VRPNodes(xpos_customer, ypos_customer, demand_customer, start_customer, end_customer, service_dura_customer,
              depot_vehicle, capacity_vehicle, max_dura_vehicle, xpos_depot, ypos_depot, vehicle_number, route_limit, TW, 
              osmid_customer, osmid_depot, company, companies_depots, diff_companies)
    
    return vrp
    


def getDepo(string):
    n = ''
    for s in string[1:]:
        if s != '_':
            n += s
        else:
            return int(n)
        

class VRPNodes:
    def __init__(self, customers_x, customers_y, demand, start, end, service_dura, depot_vehicle, 
                 capacity_vehicle, max_dura, depot_x, depot_y, vehible_number, route_limit, utw, 
                 osmid_customers, osmid_depots, customer_company, depot_company, diff_companies):
        self.customers_x = customers_x.tolist()
        self.customers_y = customers_y.tolist()
        self.demand = demand.tolist()
        self.start = start.tolist()
        self.end = end.tolist()
        if type(depot_vehicle) == list:
            self.depot_vehicle = depot_vehicle
        else:
            self.depot_vehicle = depot_vehicle.tolist()
        
        self.vehicle_capacity = []
        for cap in capacity_vehicle:
            self.vehicle_capacity.append(ast.literal_eval(cap))
        self.max_dura = max_dura.tolist()
        self.depot_x = depot_x.tolist()
        self.depot_y = depot_y.tolist()
        #self.capacity_depot = capacity_depot.tolist()
        self.vehicle_number = vehible_number.tolist()
        self.company_customer = customer_company.tolist()
        self.company_depot = depot_company.tolist()
        
        self.facilities = list(range(0,len(self.depot_x)))
        self.customers = list(range(len(self.depot_x),len(self.customers_x)+len(self.depot_x)))
        
        #combined nodes
        xpos_temp = depot_x.tolist()
        ypos_temp = depot_y.tolist()
        xpos_temp.extend(customers_x.tolist())
        self.xpos = xpos_temp
        ypos_temp.extend(customers_y.tolist())
        self.ypos = ypos_temp
        tw = [(s, e) for s, e in zip(start, end)]
        dr = service_dura.tolist()
        self.nodes = list(range(0, len(self.xpos)))
        self.time_window = {self.nodes[k+len(depot_x)] : pair for k, pair in enumerate(tw)}
        self.service_duration = {self.nodes[k+len(depot_x)] : s for k, s in enumerate(dr)}
        osd = osmid_depots.tolist()
        osc = osmid_customers.tolist()
        osd.extend(osc)
        self.osmid_nodes = osd
        cd = depot_company.tolist()
        cc = customer_company.tolist()
        
        cd.extend(cc)
        self.company_allocs = cd
        
        #the probability of initialising a population with the nearest neighbour function
        self.rnns = 0.5
        
        #initiate the cost dictionary
        cost = {}
        self.cost = cost
        
        self.vehicle_labels = {s:['S{0}_K{1}'.format(s,k) for k in range(self.vehicle_number[i])] for i,s in enumerate(self.facilities)}
        
        ll = []
        for i, _ in enumerate(self.facilities):
            for e in self.vehicle_labels[i]:
                ll.append(e)
        
        self.label_list = ll
        
        self.label_dict = {getDepo(l):[] for l in ll}
        for l in ll:
            self.label_dict[getDepo(l)].append(l)
        
        self.vehicle_limits = {label: ast.literal_eval(route_limit.tolist()[i])[j] for i,depot in enumerate(self.facilities) for j,label in enumerate(self.vehicle_labels[depot])}
        
        self.vehicle_routes = {label: [] for i,depot in enumerate(self.facilities) for j,label in enumerate(self.vehicle_labels[depot])}
        
        self.eval_coeffs = np.array([-1e6, 0, 16, 16, 16, 5, 20])
        
        self.using_time_window = utw
        
        self.diff_companies = diff_companies
        
        self.length = len(self.nodes)
        
        self.decoded_chromosomes = {}
        self.optimal_decoded_chromosomes = {}
        self.optimise_educate = {}
        self.found_vehicle_routes = {}
        self.found_costs = {}
        self.time_win_start = min([s[1][0] for s in self.time_window.items()]) if self.time_window != [] else []
        self.time_win_end = max([e[1][1] for e in self.time_window.items()]) if self.time_window != [] else []
        self.link_costs = {}