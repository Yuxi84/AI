# Author: Yuxi Zhang
# Purpose: Using Kruskal's algorithm to calculate the total length of MST given a set of point
# citation: reference code provided in
# http://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst/
import math

class KruskalTree(object):
    def __init__(self,points):
        '''
        :param points: list of n terminal points plus potential (n-2) Steiner points
        '''
        self.V = len(points) # Number of vertices

        # any possible out of this set of points could be used to construct MST
        # get a list of (u,v,w): point1, point2, edge weight
        self.graph = self.completeGraph(points)

    def find(self, parent, i):
        # TODO: better documentation
        # helper function to determine whether an edge will close a cycle
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        # TODO: better documentation
        # helper function
        xroot = self.find(parent,x)
        yroot = self.find(parent,y)

        if rank[xroot]<rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def KruskalMST(self):
        MST_set = set()

        i = 0
        e = 0

        # sort the complete graph in increasing order with respect to w in(u,v,w)
        self.graph = sorted(self.graph, key=lambda item:item[2])

        parent =[]
        rank = []

        for vIndex in range(self.V):
            parent.append(vIndex)
            rank.append(0)

        # from graph theory, we know |MST|=self.V-1
        while e < self.V-1:
            # find the smallest edge and increment i
            #TODO: make sure
            (u,v,w) = self.graph[i]
            i += 1

            x = self.find(parent,u)
            y = self.find(parent,v)

            # check if this edge close any cycle, if not, include it in set MST
            if x != y:
                e += 1
                MST_set.add((u,v,w))
                self.union(parent,rank,x,y)
        return MST(MST_set)
    #
    # def treeLen(self, mst):
    #     '''
    #     :param mst: a minimal spanning tree
    #     :return: length of the tree
    #     '''
    #     length = 0
    #     for edge in mst:
    #         length += edge[2]
    #     return length

    def edgeWeight(self,p1,p2):
        '''
        :param p1: tuple of the first point
        :param p2: tuple of the second point
        :return: the euclidean distance between two points
        '''
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def completeGraph(self, points):
        '''
        :param points: list
        :return: a list of 3-tuple (index of vertex1, index of vertex2, weight of the edge)
        '''
        graph = []
        for u in range(self.V-1):
            for v in range(u,self.V):
                w = self.edgeWeight(points[u], points[v])
                graph.append((u,v,w))
        return graph

class MST(object):

    def __init__(self, set):
        self.edges = set

    def length(self):
        '''
            :param mst: a minimal spanning tree
            :return: length of the tree
        '''
        len = 0
        for edge in self.edges:
            len += edge[2]
        return len
