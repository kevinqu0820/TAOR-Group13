#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 03:21:07 2023

@author: Kevin
"""

import numpy as np
import random
import osmnx as ox
from route_cost import route_tt, travel_time
import networkx as nx

G = ox.load_graphml('./3_boroughs_data/network.osm.graphml')


#depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
def depth(l):
    if l == []:
        return 0
    if isinstance(l, list):
        return 1 + max(depth(item) for item in l)
    else:
        return 0
    
def bpr(t0, flow, vp, C, alpha_f = 0.715, beta_f = 2.480, alpha_p = 0.683, beta_p = 2.890):
    ta = t0 * (1 + alpha_f * (flow/C)**beta_f) * (1 + alpha_p * (vp/C) ** beta_p)
    return ta

def route2graph(route):
    """

    Parameters
    ----------
    route : list
        list of integers that denote the route.

    Returns
    -------
    edges : list
        list of tuples, where each tuple is an edge in the route.

    """
    edges = [(route[j], route[j+1]) for j in range(len(route)-1)]
    
    return edges

def chromToString(chrom):
    string = ""
    for s in chrom:
        string += str(s) + "."
    return string

def encodeChrom(routes):
    #routes is a list of routes that the chromosome should represent
    r = routes.copy()
    chromosome = getLen(r)
    return chromosome

def getLen(e1):
    e = e1.copy()
    n = []
    final = -1
    if depth(e) == 1:
        for i in range(len(e)):
            if not e[i] in n:
                n.append(e[i])
        final = e[0]
        e.remove(final)
        e.append(final)
    else:
        for i in e:
            if len(i) > 0:
                n.extend(getLen(i))
    return n

def bookend(x, a):
    h = [a]
    if type(x) == list:
        h.extend(x)
    else:
            h.append(a)
    return h

def decodeChrom(chromosome, vrp):
    x = chromToString(chromosome)
    if x in vrp.decoded_chromosomes:
        return vrp.decoded_chromosomes[x]
    
    routes = []
    route = []
    for e in chromosome:
        if e in vrp.facilities:
            if len(route) == 0:
                route.append(e)
            else:
                route.append(route[0])
                routes.append(route)
                route = [e]
        else:
            route.append(e)
    if len(route) > 0:
        route.append(route[0])
        routes.append(route)
            
    vrp.decoded_chromosomes[x] = routes
    return routes

def createChromosome(vrp):
    x = vrp.facilities.copy()
    x.extend(vrp.customers.copy())
    random.shuffle(x)
    return x

def initialPopulation(pop_size, vrp, G):
    population = []
    for i in range(0, pop_size):
        if random.random() < vrp.rnns:
            chrom, vrp.cost = nearestNeighbor(createChromosome(vrp), vrp, G)
            population.append(chrom)
        else:
            population.append(createChromosome(vrp))
    return population

def initialPopulationsDiffCompanies(pop_size, vrp, G):
    population = []
    companiesList = []
    companiesDict = {}
    populations = []
    for company in vrp.company_allocs:
        if not company in companiesList:
            companiesList.append(company)
    for company in companiesList:
        companiesDict[company] = []
    for node, company in enumerate(vrp.company_allocs):
        companiesDict[company].append(node)
    for (_, l) in companiesDict.items():
        population = []
        for i in range(0, pop_size):
            random.shuffle(l)
            if random.random() < vrp.rnns:
                chrom, vrp.cost = nearestNeighbor(l, vrp, G)
                population.append(chrom)
            else:
                population.append(l)
        populations.append(population)
    return populations

def totalLen(r):
    if depth(r) == 1:
        return len(r)
    else:
        h = 0
        for l in r:
            h += totalLen(l)
        return h
    
def selection(initial_population_size, population,vrp):
    #we assume that the population is sorted into order of fitness
    for pop in population:
        r = optimalDecode(pop, vrp)
        if totalLen(r) > totalLen(shedEmptyRoutes(r)):
            population.remove(pop)
    ret = population[len(population)-initial_population_size:]
    rem = population[:len(population)-initial_population_size]
    for r in rem:
        r = chromToString(r)
        if r in vrp.decoded_chromosomes:
            del vrp.decoded_chromosomes[r]
        if r in vrp.optimal_decoded_chromosomes:
            del vrp.optimal_decoded_chromosomes[r]
        if r in vrp.optimise_educate:
            del vrp.optimise_educate[r]
        if r in vrp.found_vehicle_routes:
            vrp.found_vehicle_routes[r]
    return ret

#nearst neighborhod heuristic
def nearestNeighbor(chromosome, vrp, G):
    """ function to improve the individual(chromosome) by usng nearsest neighbor heuristic [just for initializaiont]
            input: 
                chromosome
            output:
                new_route: new route which will be implement to the chromosome"""
        
    giant_routes = decodeChrom(chromosome, vrp)
    
    N = len(giant_routes)
    
    for i, route in enumerate(giant_routes):
        
        #clear_output(wait=True)
        #print(i, "/", N)
        
        depo = route[0]
        giant_route = route[1:-1].copy()

        #loop to make sure that improvement can only be applied to a route that served more than one customer.
        if len(giant_route) > 1 : 
            start_customer = random.sample(giant_route,1)[0]

            #Find the nearest customer and add it to the route from start
            new_route = [start_customer]
            unsorted = set(giant_route) - set(new_route)
            for _ in range(len(giant_route) - 1):
                next = None
                cost = 1e6
                for customer in unsorted:
                    try: 
                        new_cost = vrp.cost[(vrp.osmid_nodes[new_route[-1]], vrp.osmid_nodes[customer])]
                    except KeyError:
                        new_cost = travel_time(G, vrp.osmid_nodes[new_route[-1]], vrp.osmid_nodes[customer])
                        vrp.cost[(vrp.osmid_nodes[new_route[-1]], vrp.osmid_nodes[customer])] = new_cost

                    if new_cost < cost:
                        next = customer
                        cost = new_cost
                new_route.append(next)
                unsorted = unsorted - set([next])
            q = [depo] + new_route + [depo]
            giant_routes[i] = q
    #convert giant route generated to chromosome, inverse calculation of chromosome
    chromosome_new = encodeChrom(giant_routes)
    return chromosome_new, vrp.cost

def shedEmptyRoutes(routes):
    a = []
    if depth(routes) == 1:
        b = []
        for e in routes:
            if e != []:
                b.append(e)
        a = b
    else:
        for l in routes:
            shed = shedEmptyRoutes(l)
            if shed != []:
                a.append(shed)
    return a

def optimalDecode(chromosome, vrp):
    """decoding chromosome to route plan by using method proposed by prins(2006)
    
    arg:
        input: 
            chromosome
        output:
            routes: optimal and shortest route"""
    x = chromToString(chromosome)
    if x in vrp.optimal_decoded_chromosomes:
        return vrp.optimal_decoded_chromosomes[x]
    
    giant_routes = decodeChrom(chromosome, vrp)
    
    #Optimal split method
    #minimize cost subject to vehicle number and vehicle capacity limit
    
    vrp.vehicle_routes = {label: [] for i,depot in enumerate(vrp.facilities) for j,label in enumerate(vrp.vehicle_labels[depot])}
    total_routes = []
    INF =  1e6

    for giant_route in giant_routes:
        if giant_route[0] not in vrp.facilities:
            continue
        customers = giant_route[1:-1].copy()
        depot = giant_route[0]
        if customers != []:
            #print(depot)
            #print(vrp.vehicle_number)
            vehicle_number = vrp.vehicle_number[depot]
            labels = vrp.vehicle_labels[depot]
            #vehicle_capacity = [vrp.vehicle_limits[label] for label in labels]
            vehicle_capacity = vrp.vehicle_capacity[depot]
            nodes = [depot] + customers

            #set up dict v1 and v2, to store mini distance values
            v1 = {n:INF for n in nodes} #initialize v1 dict, with all nodes in nodes with value infinity
            v1[nodes[0]] = 0 #set first value to zero, as distance for starting to itself is 0
            v2 = v1.copy() #create copy of v1 and assign it to v2
            pre_nodes_dict = {}
            k = 1 #counter
            flag = True  #boolean varialbe to control loop

            #loop to iterate until all vehicles assigned route, or no further improvement can be made
            while (k <= vehicle_number or (not flag)) and k <= vehicle_number:
                flag = True
                #loop iterates over all nodes, except starting node
                for i in range(1, len(nodes)):
                    j = i
                    load = 0 # load to zero
                    #loop to add nodes to route until max vehicle_capacity is reached or all nodes added
                    while j <= len(nodes) -1 and load <= vehicle_capacity[k-1]:
                        load += vrp.demand[getCustIndex(nodes[j], vrp)] #to track the total demand
                        #if we can add node to current route, without exceeding the max cap, route cost calculated
                        if load <= vehicle_capacity[k-1]:
                            if j == i :
                                try:
                                    cost = vrp.cost[(vrp.osmid_nodes[nodes[0]], vrp.osmid_nodes[nodes[i]])] + vrp.cost[(vrp.osmid_nodes[nodes[i]],vrp.osmid_nodes[nodes[0]])]
                                except KeyError:
                                    cost1 = travel_time(G, vrp.osmid_nodes[nodes[0]], vrp.osmid_nodes[nodes[i]])
                                    vrp.cost[(vrp.osmid_nodes[nodes[0]], vrp.osmid_nodes[nodes[i]])] = cost1
                                    cost2 = travel_time(G, vrp.osmid_nodes[nodes[i]], vrp.osmid_nodes[nodes[0]])
                                    vrp.cost[(vrp.osmid_nodes[nodes[i]], vrp.osmid_nodes[nodes[0]])] = cost2
                                    cost = cost1 + cost2
                            else:
                                try:
                                    cost = cost - vrp.cost[(vrp.osmid_nodes[nodes[j-i]],vrp.osmid_nodes[nodes[0]])] + vrp.cost[(vrp.osmid_nodes[nodes[j-i]],vrp.osmid_nodes[nodes[j]])] + vrp.cost[(vrp.osmid_nodes[nodes[j]],vrp.osmid_nodes[nodes[0]])]
                                except KeyError:
                                    cost1 = travel_time(G, vrp.osmid_nodes[nodes[j-i]], vrp.osmid_nodes[nodes[0]])
                                    vrp.cost[(vrp.osmid_nodes[nodes[j-i]], vrp.osmid_nodes[nodes[0]])] = cost1
                                    cost2 = travel_time(G, vrp.osmid_nodes[nodes[j-i]], vrp.osmid_nodes[nodes[j]])
                                    vrp.cost[(vrp.osmid_nodes[nodes[j-i]], vrp.osmid_nodes[nodes[j]])] = cost2
                                    cost3 = travel_time(G, vrp.osmid_nodes[nodes[j]], vrp.osmid_nodes[nodes[0]])
                                    vrp.cost[(vrp.osmid_nodes[nodes[j]], vrp.osmid_nodes[nodes[0]])] = cost3
                                    cost = cost - cost1 + cost2 + cost3
                            #if cost is less than previous cost of final node in route, distance value in v2 updated with new min cost, and final node recorded pre_nodes_dic
                            if v1[nodes[i-1]] + cost < v2[nodes[j]]:
                                v2[nodes[j]] = v1[nodes[i-1]] + cost
                                pre_nodes_dict[nodes[j]] = nodes[i-1]
                                flag = False # algorithm continue to the next iteration if any improvement was made to route
                        j += 1
                    v1 = v2.copy() #v2 copied to v1
                k += 1
            #check if not empty
            if pre_nodes_dict !={}:
                routes = [] #initialize empty list to store routes
                last_index = len(pre_nodes_dict) #to loop up predecessor node for the final node in route
                start_node = pre_nodes_dict[nodes[last_index]] #extract the start for last constructed route
                end_node = nodes[last_index] #extract hte end node
                routes.append([depot] + nodes[nodes.index(start_node) + 1:nodes.index(end_node) + 1] + [depot]) #append nodes for the last constructed route list, use nodes list, index function to extract nodes between start and end nodes
                #loop to extract remaining routes from pre_nodes_dict
                while start_node != nodes[0]:
                    end_node = start_node 
                    start_node = pre_nodes_dict[end_node]
                    routes.append([depot] + nodes[nodes.index(start_node) + 1:nodes.index(end_node) + 1] + [depot])
                total_routes.append(routes)
                
    ########################################################################################################
    
            """ Need a new dictionary with a new method for encoding routes into strings (?) 
                the dictionary will map an optimally decoded route to its chromosome         """
    
    ########################################################################################################
    
    if total_routes != []:
        #if depth(total_routes) == 2:
         #   total_routes = [total_routes]
        vrp.optimal_decoded_chromosomes[x] = total_routes
        #for routes_ind, routes in enumerate(total_routes):
            #if depth(routes) == 1:
             #   total_routes[routes_ind] = [routes]
    return total_routes

def getIndex(node, vrp):
    ind = -1
    for i, j in enumerate(vrp.nodes):
        if j == node:
            ind = i
    return ind

#code for finding if a customer belongs to a depot
def companyMatch(depot, customer, vrp):
    if vrp.company_allocs[depot] == vrp.company_allocs[customer]:
        return True
    return False

#code for code for turning a route into a list of osmid data
def osmidRoute(routes, vrp):
    osroute = []
    for e in routes:
        osroute.append(vrp.osmid_nodes[e])
    return osroute

def getCustIndex(node, vrp):
    return getIndex(node, vrp) - len(vrp.facilities)

def findVehicleRoutes(chromosome, vrp):
    
    #print(depth(chromosome))
    
    return optimalDecode(chromosome, vrp)

def mutate(chromosome, vrp):
    """function to do mutation operation for the GA
    input:
        chromosome
    output:
        chromosome with a mutation operation applied"""
    #muation metho
    if type(chromosome) != list:
        chromosome = list(chromosome)
    new_chromosome = chromosome.copy()
    mutation_way = random.randint(0,2)
    if len(chromosome) > 1:
        location_1 = random.randint(0,len(chromosome)- 1)
        location_2 = random.randint(0,len(chromosome)- 1)
        
        if mutation_way == 0: 
            #swap the route
            new_chromosome[location_1], new_chromosome[location_2] = chromosome[location_2], chromosome[location_1]
        elif mutation_way == 1:
            #insertion route
            new_chromosome.insert(location_2, new_chromosome.pop(location_1))
        else:
            #inversion
            start = min(location_1, location_2)
            end = max (location_1, location_2)
            new_chromosome[start:end] = chromosome[start:end][::-1]

        #routes, total_demand, total_travel_cost = evaluate(chromosome)
    for i, e in enumerate(new_chromosome):
        for j, k in enumerate(new_chromosome):
            if e == k and i != j:
                return chromosome
            if abs(i-j) == 1:
                if e in vrp.facilities and k in vrp.facilities:
                    return chromosome
        
    if len(new_chromosome) < len(chromosome):
        return chromosome
    else:
        return new_chromosome


def crossover(chromosomeA, chromosomeB):
    #this is where the code to do the crossover will go

    child = []
    childA = []
    childB = []
    
    if len(chromosomeA) > 1 and len(chromosomeB) > 1 :

        geneA = random.randint(0,len(chromosomeA)-1)
        geneB = random.randint(0,len(chromosomeA)-1)

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childA.append(chromosomeA[i])

        childB = [item for item in chromosomeB if item not in childA]

        child = childA + childB
        
    # for i, e in enumerate(child):
    #     for j, k in enumerate(child):
    #         if e == k and i != j:
    #             return chromosomeA
    #         if abs(i-j) == 1:
    #             if e in vrp.facilities and k in vrp.facilities:
    #                 return chromosomeA
                
    # if len(child) < len(chromosomeA):
    #     return chromosomeA
                
    return child

#finds the time for service
def findServiceTime(k, route, init_dep_time, vrp):
    depart_time = init_dep_time
    time_delay = 0
    time_idling = 0
    for start, end in zip(route[:-2], route[1:-1]):
        try:
            time = vrp.cost[(vrp.osmid_nodes[start], vrp.osmid_nodes[end])]
        except KeyError:
            time  = travel_time(G, vrp.osmid_nodes[start], vrp.osmid_nodes[end])
        time_delay += max(0, depart_time + time - vrp.time_window[end][1])
        time_idling += vrp.service_duration[end] + max(0, vrp.time_window[end][0]-(depart_time + time))
        depart_time = max(vrp.time_window[end][0] + vrp.service_duration[end], depart_time + time +vrp.service_duration[end])
    try:
        more_time = vrp.cost[(vrp.osmid_nodes[route[-2]], vrp.osmid_nodes[route[-1]])]
    except KeyError:
        more_time = travel_time(G, vrp.osmid_nodes[route[-2]], vrp.osmid_nodes[route[-1]])
    time_excessive = max(0, depart_time + more_time - init_dep_time - vrp.vehicle_limits[k]) 
    #time_excessive = overtime
    #time_delay = sum of all time window violation
    #time_idling = time wasted due to early arrival
    #depart_time = the time when the vehicle leaves the most recently visited customer 
    return [depart_time, time_delay, time_idling, time_excessive]

def educate(chromosome, vrp):
    x = chromToString(chromosome)
    if x in vrp.optimise_educate:
        if vrp.optimise_educate[x] == True:
            return chromosome, True
    
    #this implements the split algorithm to find the optimal route
    giant_routes = findVehicleRoutes(chromosome, vrp)
    
    for giant_route in giant_routes:
        if giant_route[0] not in vrp.facilities:
            return chromosome, False
    
    cost_change = 100
    
    #print(giant_routes)
    
    #figure out how to properly match vehicles with depots
    for k, routes_for_each_depo in enumerate(giant_routes):
        if depth(routes_for_each_depo) == 1:
            routes_for_each_depo = [routes_for_each_depo]
        for (vehicle, route) in zip(vrp.label_dict[routes_for_each_depo[0][0]], routes_for_each_depo):
            improved = True
            n = 0
            while improved:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        n += 1
                        if j-i == 1:
                            continue
                        n1 = route[i-1]
                        n2 = route[i]
                        n3 = route[j-1]
                        n4 = route[j]
                        try:
                            cost13 = vrp.cost[(vrp.osmid_nodes[n1], vrp.osmid_nodes[n3])]
                        except KeyError:
                            cost13 = travel_time(G, vrp.osmid_nodes[n1], vrp.osmid_nodes[n3])
                        try:
                            cost24 = vrp.cost[(vrp.osmid_nodes[n2], vrp.osmid_nodes[n4])]
                        except KeyError:
                            cost24 = travel_time(G, vrp.osmid_nodes[n2], vrp.osmid_nodes[n4])
                        try: 
                            cost12 = vrp.cost[(vrp.osmid_nodes[n1], vrp.osmid_nodes[n2])]
                        except KeyError:
                            cost12 = travel_time(G, vrp.osmid_nodes[n1], vrp.osmid_nodes[n2])
                        try:
                            cost34 = vrp.cost[(vrp.osmid_nodes[n3], vrp.osmid_nodes[n4])]
                        except KeyError:
                            cost34 = travel_time(G, vrp.osmid_nodes[n3], vrp.osmid_nodes[n4])

                        d_travel_cost = cost13 + cost24  - cost12 - cost34

                        new_route = route.copy()
                        new_route[i] = n3
                        new_route[j-1] = n2

                        if vrp.using_time_window:
                            try:
                                cost = vrp.cost[(vrp.osmid_nodes[0], vrp.osmid_nodes[1])] 
                            except KeyError:
                                cost = travel_time(G, vrp.osmid_nodes[0], vrp.osmid_nodes[1] )

                            depart_time = vrp.time_window[route[1]][1]- cost

                            c1 = findServiceTime(vehicle, route, depart_time, vrp)
                            c2 = findServiceTime(vehicle, new_route, depart_time, vrp)
                            dc = np.subtract(c2, c1).tolist()

                            h = np.array([0, 0, d_travel_cost, 0, dc[2], dc[1], dc[3]])

                            cost_change = np.dot(h, vrp.eval_coeffs)
                        else:
                            cost_change = d_travel_cost * vrp.eval_coeffs[2]

                        if cost_change < -10:
                            giant_routes[k] = new_route
                            improved = True   
                        else:
                            improved = False
                        if n > 200 :
                            improved = False
                            
    new_chromosome = encodeChrom(giant_routes)
    
    if len(new_chromosome) < len(chromosome):
        return chromosome, True
    
    for i, e in enumerate(new_chromosome):
        for j, k in enumerate(new_chromosome):
            if e == k and i != j:
                return chromosome, True
            if abs(i-j) == 1:
                if e in vrp.facilities and k in vrp.facilities:
                    return chromosome, True
    
    y = chromToString(chromosome)
    if y == x:
        vrp.optimise_educate[x] = True

    return new_chromosome, True

def evolve(population, mutation_rate, crossover_rate, educate_rate, vrp):
    next_gen = []
    for ind in population:
        if random.random() <= mutation_rate:
            new_ind = mutate(ind, vrp)
            next_gen.append(new_ind)
        if random.random() <= crossover_rate:
            new_ind = crossover(ind, population[random.randint(0, len(population)-1)])
            next_gen.append(new_ind)
        if random.random() <= educate_rate:
            new_ind, valid = educate(ind, vrp)
            if not valid:
                population.remove(new_ind)
            else:
                next_gen.append(new_ind)
    if next_gen != []:
        population.extend(next_gen)
    return population

def routeListToString(route_list):
    x = ""
    for route in route_list:
        x += ":" + chromToString(route)
    return x


def evaluate(list_routes, vrp, G, beta = {'Q': 2,'V': 22, 'TT':10.25, 'IT': 10.25, 'DT':10.25, 'OT':21.25}):
    """
    function to evaluate the quality of solution which has been selected by the algorithm
    
    input:
        individual: list of lists
            list of routes to be evaluated
        beta : dict
            dict with keys 'V','Q', 'TT', 'DT', and 'OT' that contains the coefficients
            of the cost function.
    output:
        profit: scalar
            total profit of the evaluated route
  """
    #extract values and items from individual.routes dict
    #routes_dict_values = individual.routes.values() 
    
    x = routeListToString(list_routes)
    if x in vrp.found_costs:
        return vrp.found_costs[x][0], vrp.cost, vrp.found_costs[x][1]
    
    #Calculate the total demand that satisfied by the solution
    total_demand = 0 #initialize variable
    for routes in list_routes:
        for route in routes: #iterate through each value in route
            for customer in route[1:-1]: #iterate through customer node in the route [all nodes between first & lat nodes on the route]
                #print("cust: ", customer, customer-len(vrp.facilities))
                total_demand += vrp.demand[customer-len(vrp.facilities)] 

    Cost, vrp.cost, _, _, _ = total_cost(list_routes, vrp, beta, G)
    
    profit = beta['Q']*total_demand*10 - Cost
    
    #print(beta['Q']*total_demand*10, Cost, beta['TT'] * tt)
    
    vrp.found_costs[x] = [profit, Cost]
    
    return profit, vrp.cost, Cost


def total_cost(route_plan, vrp, beta, G):
    """
    

    Parameters
    ----------
    route_plan : list of lists
        complete route plan, each list being a single route.
    vrp : class

    beta : dict
        dict with keys 'V', 'TT', 'DT', and 'OT' that contains the coefficients
        of the cost function.
    G : classes.multidigraph.MultiDiGraph
        Road network graph.


    Returns
    -------
    C : scalar
        total monetary cost of route plan.
    travel_time : scalar
        total travel time.
    delay_time : scalar
        total delay time (time window violation).
    overtime : scalar
        total overtime.

    """
    cost_dict = vrp.link_costs
    #print("route_plan depth: ", depth(route_plan))
    giant_routes = route_plan#findVehicleRoutes(route_plan, vrp)
    travel_time_var = 0
    delay_time = 0
    idling_time = 0
    overtime = 0
    V = 0
    traffic_flows = {}
    capacity = {}
    sum_dura = vrp.time_win_end-vrp.time_win_start if vrp.using_time_window else 1
    try:
        cost01 = vrp.cost[(vrp.osmid_nodes[0], vrp.osmid_nodes[1])]
    except KeyError:
        cost01 = travel_time(G, vrp.osmid_nodes[0], vrp.osmid_nodes[1])
        
    #moved this into the loop to go where we start the vehicle's route
    #depart_time = vrp.time_window[route[1]][1] - cost01
    if depth(giant_routes) == 2:
        giant_loop = [giant_routes.copy()]
    else:
        giant_loop = giant_routes.copy()

    for _, routes_for_each_depo in enumerate(giant_loop):
        if depth(routes_for_each_depo) == 1:
            idk = [routes_for_each_depo.copy()]
        else:
            idk = routes_for_each_depo.copy()
        for (vehicle, route) in zip(vrp.label_dict[idk[0][0]], idk):
            # Function to pull osmid
            osmid = osmidRoute(route, vrp)
            
            edges = route2graph(osmid)
            for i in range(len(edges)):
                a,b = edges[i]
                shortest_path = nx.shortest_path(G, a, b, weight='travel_time')
                for j in range(len(shortest_path) - 1):
                    link = (shortest_path[j], shortest_path[j+1])
                    edge_data = G[shortest_path[j]][shortest_path[j+1]][0]
                    try:
                        t0 = cost_dict[link]
                    except KeyError:
                        try:
                            speed = edge_data['maxspeed']
                        except KeyError:
                            # If no maxspeed is specified, use recommended 38 kph
                            speed = 38 
                        if isinstance(speed, str):
                            speed = float(speed.split()[0])
                        elif isinstance(speed, list):
                            speed = float(speed[0].split()[0])
                        speed = speed * 1000
                        t0 = edge_data["length"]/speed
                        cost_dict[link] = t0
                    traffic_flows[link] = 1/sum_dura if traffic_flows.get(link,None) is None else traffic_flows[link]+1/sum_dura
                    capacity[link] = edge_data["capacity"]

    tt_base_var = 0
    t0_base = 0
    for _, routes_for_each_depo in enumerate(giant_loop):
        if depth(routes_for_each_depo) == 1:
            idk = [routes_for_each_depo.copy()]
        else:
            idk = routes_for_each_depo.copy()
        for (vehicle, route) in zip(vrp.label_dict[idk[0][0]], idk):
            depart_time = vrp.time_window[route[1]][1] - cost01
            times = findServiceTime(vehicle, route, depart_time, vrp)
            osmid = osmidRoute(route, vrp)
            edges = route2graph(osmid)
            tt_base,_ = route_tt(G, osmid)
            tt_base_var += tt_base
            for i in range(len(edges)):
                
                a,b = edges[i]
                links = nx.shortest_path(G, a, b, weight='travel_time')
                for j in range(len(links) - 1):
                    link = (links[j],links[j+1])
                    t0 = cost_dict[link]
                    t0_base += t0
                    flow = traffic_flows[link]
                    cap = float(capacity[link])
                    travel_time_var += bpr(t0, flow, 0.4*cap, cap)
                    #print(t0, flow, cap)
                    
            delay_time += times[1]
            idling_time += times[2]
            overtime += times[3]
            V += 1
           
    C = beta['V'] * V + beta['TT'] * travel_time_var + beta['DT'] * delay_time + beta['IT'] * idling_time + beta['OT']*overtime 
    #print(travel_time_var, t0_base, tt_base_var)   
    return C, cost_dict, travel_time_var, delay_time, overtime


def chromosomeFitness(chromosome, vrp, G):
    routes = findVehicleRoutes(chromosome,vrp)
    profit, vrp.cost, Cost = evaluate(routes, vrp, G)
    return profit, vrp.cost, Cost

def rankChromosomes(population, vrp, G):
    fitness_results = []
    for i in range(0, len(population)):
        fitness,vrp.cost, _ = chromosomeFitness(population[i], vrp,G)
        fitness_results.append(fitness)
    population = [x for _, x in sorted(zip(fitness_results, population))]
    #print("New ranks")
    #print([x for x, _ in sorted(zip(fitness_results, population))])
    return population #, vrp.cost

def getAvg(l):
    avg = 0
    for e in l:
        avg += e/len(l)
    return avg

def split(l, m):
    r = []
    if len(l)/m - int(len(l)/m) == 0.0 :
        for i in range(0, m):
            a = []
            for j in range(0, int(len(l)/m)):
                a.append(l[i*m+j])
            r.append(a)
        return r
    print("length of list isn't divisible by the number of sections")
    return r

def geneticAlgorithm(initial_population_size, n_gens, mutation_rate, crossover_rate, educate_rate, vrp, m , G):
    # initial_population_size number of routes to put in the initial population
    # n_gens number of generations to run the algorithm for
    # nodes a VRPNodes object containing 2 lists, one for customers and one for facilities
    # m number of subsets to cut the population into when making the next generation
    # G: road network graph
    if vrp.diff_companies == False:
        population = initialPopulation(initial_population_size, vrp, G)
        y = []
        costi = []
        for i in range(n_gens):
            #first split the population randomly into m even parts
            random.shuffle(population)
            populations = split(population, m)
            new_population = []
            for pop in populations:
                new_pop = evolve(pop, mutation_rate, crossover_rate, educate_rate, vrp)
                new_population.extend(new_pop)
            population.extend(new_population)
            population = selection(initial_population_size, rankChromosomes(population, vrp, G), vrp)

            f,vrp.cost,c = chromosomeFitness(population[-1], vrp, G)
            y.append(f)
            costi.append(c)
        #plt.plot(y)
        return population, y, costi
    else:
        y = []
        Cost = []
        populations = initialPopulationsDiffCompanies(initial_population_size, vrp, G)
        final_populations = []
        for n, population in enumerate(populations):
            yi = []
            costi = []
            for i in range(n_gens):
                #first split the population randomly into m even parts
                #clear_output(wait=True)
                #print(i)
                random.shuffle(population)
                ps = split(population, m)
                new_population = []
                for pop in ps:
                    new_pop = evolve(pop, mutation_rate, crossover_rate, educate_rate, vrp)
                    new_population.extend(new_pop)
                population.extend(new_population)
                population = selection(initial_population_size, rankChromosomes(population, vrp, G), vrp)

                f, vrp.cost,c = chromosomeFitness(population[-1], vrp, G)
                yi.append(f)
                costi.append(c)
            final_populations.append(population)
            y.append(yi)
            Cost.append(costi)
        g = []
        for i in range(len(y[0])):
            h = 0
            for k in y:
                h += k[i]
            g.append(h)
        #plt.plot(g)
        return final_populations, y, Cost
    
    

