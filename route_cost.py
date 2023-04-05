#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:43:08 2023

@author: Kevin
"""

import networkx as nx

"""
Code to read road network graph:
    import osmnx as ox
    G = ox.load_graphml('./3_boroughs_data/network.osm.graphml')
"""

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

def travel_time(G, node1, node2, speed_conversion='kph'):
    """

    Parameters
    ----------
    G : classes.multidigraph.MultiDiGraph
        Road network graph.
    node1 : int
        index of node 1 being evaluated.
    node2 : int
        index of node 2 being evaluate.
    speed_conversion : string, optional
        speed units in graph. The default is 'kph'.

    Returns
    -------
    travel_time : scalar
        travel time in hours amongst shortest path between nodes 1 and 2.

    """
    
    
    # Calculate the shortest path between the nodes
    try:
        shortest_path = nx.shortest_path(G, node1, node2, weight='travel_time')
    except nx.NetworkXNoPath:
        return None
    
    edges_data = []
    # Calculate the travel time of the shortest path
    travel_time = 0
    for i in range(len(shortest_path) - 1):
        edge_data = G[shortest_path[i]][shortest_path[i + 1]][0]
        edges_data.append(edge_data)
        try:
            speed = edge_data['maxspeed']
        except KeyError:
            # If no maxspeed is specified, use recommended 38 kph
            speed = 38 
        
        if isinstance(speed, str):
            speed = float(speed.split()[0])
        elif isinstance(speed, list):
            speed = float(speed[0].split()[0])
            
        if speed_conversion == 'mph':
            speed = speed * 1.60934 * 1000  # Convert mph to meters/h
        else:
            speed = speed * 1000 # Convert kph to meters/h

        travel_time += edge_data['length'] / speed
        #print(edge_data['length'] / speed)


    return travel_time

def route_tt(G, route, cost_dict = {}):
    """
    

    Parameters
    ----------
    G : classes.multidigraph.MultiDiGraph
        Road network graph.
    route : list
        Sequency of osmid of a single route.
    cost : dictionary, optional
        Dictionary of known costs. The default is empty.

    Returns
    -------
    cost : scalar
        Total travel time of route.
    cost_dict : TYPE
        Updated cost dictionary.

    """
    
    edges = route2graph(route)
    total_cost = 0
    for i in range(len(edges)):
        a,b = edges[i]
        try:
            cost = cost_dict[(a,b)]
        except KeyError:
            cost = travel_time(G,a,b)
            cost_dict[(a,b)] = cost
        total_cost += cost
    return total_cost, cost_dict


    