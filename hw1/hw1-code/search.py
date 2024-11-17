# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import heapq
import sys
import copy
import numpy as np

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def manhattan_distance(start, end):
    return sum(abs(a - b) for a, b in zip(start, end))

def actual_distance(maze):
    # only run bfs on objectives (because the distance will only be used when constructing MST)
    objectives = maze.getObjectives()
    table = dict()

    for point in objectives:
        q = []
        visited = set()
        start = Node(point, 0, 0, 0)
        start.parent = start
        q.append(start)
        distance = 0

        while q:
            distance += 1
            tmp = q.copy()
            q.clear()
            while tmp:
                current = tmp.pop(0)
                neighbors = maze.getNeighbors(*current.position)
                visited.add(current.position)
                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    table[frozenset([point, neighbor])] = distance
                    fringe = Node(neighbor, 0, 0, 0, current)
                    q.append(fringe)
    return table

# def heuristic_cost(points, table):  # heuristic: sum(closest_point_distance(point))
#     if frozenset(points) in table:
#         return table[frozenset(points)]
#     cost = []
#     for i in points:
#         min_distance = sys.maxsize
#         for j in points:
#             if i == j:
#                 continue
#             min_distance = min(min_distance, manhattan_distance(i, j))
#         cost.append(min_distance)
#     total_cost = np.sum(cost) - np.max(cost)
#     table[frozenset(points)] = total_cost
#     return total_cost

def heuristic_cost(points, table):  # heuristic: Minimum Spanning Tree
    if frozenset(points) in table:
        return table[frozenset(points)]
    roots = list(range(len(points)))
    graph = []
    cost = 0
    edge = 0

    def find(a):
        while a != roots[a]:
            a = roots[a]
        return a

    def union(a, b):
        roots[find(a)] = find(b)
        
    for i in range(len(points)):
        for j in range(i + 1,len(points)):
            distance = manhattan_distance(points[i], points[j])
            graph.append((i, j, distance))
    
    graph.sort(key=lambda x:x[2])
    for a, b ,distance in graph:        
        if find(a) == find(b):  # a and b are already connected
            continue
        union(a, b)
        cost += distance
        edge += 1
        if edge == len(points) - 1:  # all connected
            break

    table[frozenset(points)] = cost
    return cost

def heuristic_actual_cost(points, table):  # heuristic: Minimum Spanning Tree
    roots = list(range(len(points)))
    graph = []
    cost = 0
    edge = 0

    def find(a):
        while a != roots[a]:
            a = roots[a]
        return a

    def union(a, b):
        roots[find(a)] = find(b)
        
    for i in range(len(points)):
        for j in range(i + 1,len(points)):
            distance = table[frozenset([points[i], points[j]])]
            graph.append((i, j, distance))
    
    graph.sort(key=lambda x:x[2])
    for a, b ,distance in graph:        
        if find(a) == find(b):  # a and b are already connected
            continue
        union(a, b)
        cost += distance
        edge += 1
        if edge == len(points) - 1:  # all connected
            break
    return cost

class Node:
    def __init__(self, position, past_cost, future_cost, objectives, parent=None):
        self.position = position
        self.past_cost = past_cost
        self.future_cost = future_cost
        self.total_cost = past_cost + future_cost
        self.parent = parent  # another node
        self.objectives = objectives

    def __lt__(self, other):
        return self.total_cost < other.total_cost

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item):
        heapq.heappush(self.elements, item)

    def get(self):
        return heapq.heappop(self.elements)

def backtrace(node):  # print the path
    path = []
    while node.parent != node:
        path.append(node.position)
        node = node.parent
    path.append(node.position)
    path.reverse()
    return path

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    q = []
    visited = set()

    start = Node(maze.getStart(), 0, 0, maze.getObjectives())
    start.parent = start
    q.append(start)

    while q:
        current = q.pop(0)
        if current.position == maze.getObjectives()[0]:  # found all dots -> return path
            return backtrace(current)
        neighbors = maze.getNeighbors(*current.position)
        if (current.position, *current.objectives) in visited:
            continue
        visited.add((current.position, *current.objectives))
        for neighbor in neighbors:
            if (neighbor, *current.objectives) in visited:
                continue
            fringe = Node(neighbor, 0, 0, maze.getObjectives(), current)
            q.append(fringe)

    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
 
    pq = PriorityQueue()
    visited = set()
    table = dict()

    start = Node(maze.getStart(), 0, heuristic_cost([maze.getStart(), *maze.getObjectives()], table), maze.getObjectives())
    start.parent = start
    pq.put(start)


    while not pq.empty():
        current = pq.get()
        if current.position == maze.getObjectives()[0]:  # found all dots -> return path
            return backtrace(current)
        if (current.position, *current.objectives) in visited:
            continue
        neighbors = maze.getNeighbors(*current.position)
        visited.add((current.position, *current.objectives))
        for neighbor in neighbors:
            if (neighbor, *current.objectives) in visited:
                continue
            fringe = Node(neighbor, current.past_cost + 1, heuristic_cost([neighbor, *maze.getObjectives()], table), maze.getObjectives(), current)
            pq.put(fringe)

    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    pq = PriorityQueue()
    visited = set()
    table = dict()

    start = Node(maze.getStart(), 0, heuristic_cost([maze.getStart(), *maze.getObjectives()], table), maze.getObjectives())
    start.parent = start
    pq.put(start)

    while not pq.empty():
        current = pq.get()
        if len(current.objectives) == 0:  # found all dots -> return path
            return backtrace(current)
        if (current.position, *current.objectives) in visited:
            continue
        neighbors = maze.getNeighbors(*current.position)
        visited.add((current.position, *current.objectives))
        for neighbor in neighbors:
            if (neighbor, *current.objectives) in visited:
                continue
            objectives = copy.deepcopy(current.objectives)
            if neighbor in objectives:
                objectives.remove(neighbor)
            fringe = Node(neighbor, current.past_cost + 1, heuristic_cost([neighbor, *objectives], table), objectives, current)
            pq.put(fringe)

    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    pq = PriorityQueue()
    visited = set()
    table = actual_distance(maze)

    start = Node(maze.getStart(), 0, heuristic_actual_cost([maze.getStart(), *maze.getObjectives()], table), maze.getObjectives())
    start.parent = start
    pq.put(start)

    while not pq.empty():
        current = pq.get()
        if len(current.objectives) == 0:  # found all dots -> return path
            return backtrace(current)
        if (current.position, *current.objectives) in visited:
            continue
        neighbors = maze.getNeighbors(*current.position)
        visited.add((current.position, *current.objectives))
        for neighbor in neighbors:
            if (neighbor, *current.objectives) in visited:
                continue
            # new states
            objectives = copy.deepcopy(current.objectives)
            if neighbor in objectives:
                objectives.remove(neighbor)
            fringe = Node(neighbor, current.past_cost + 1, heuristic_actual_cost([neighbor, *objectives], table), objectives, current)
            pq.put(fringe)

    return []

def find_closest(point, objectives):
    distance = sys.maxsize
    for objective in objectives:
        if manhattan_distance(point, objective) < distance:
            distance = manhattan_distance(point, objective)
            closest = objective
    return closest

def fast_astar(maze, start, objective, objectives):
    pq = PriorityQueue()
    visited = set()
    table = dict()

    start = Node(start, 0, heuristic_cost([start, objective], table), objective)
    start.parent = start
    pq.put(start)


    while not pq.empty():
        current = pq.get()
        if current.position == objective:  # found all dots -> return path
            path = backtrace(current)
            for point in path:
                if point in objectives:
                    objectives.remove(point)
            return objectives, path

        if (current.position, *current.objectives) in visited:
            continue
        neighbors = maze.getNeighbors(*current.position)
        visited.add((current.position, *current.objectives))
        for neighbor in neighbors:
            fringe = Node(neighbor, current.past_cost + 1, heuristic_cost([neighbor, objective], table), objective, current)
            pq.put(fringe)

    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    # find closest point -> change the problem into single dot -> Time Complexity: O(single dot) * n
    objectives = maze.getObjectives()
    start = maze.getStart()
    path = []

    while objectives:
        objective = find_closest(start, objectives)
        objectives, new_path = fast_astar(maze, start, objective, objectives)
        if start != maze.getStart():
            new_path.pop(0)  # remove duplicates
        path.extend(new_path)
        start = objective

    return path