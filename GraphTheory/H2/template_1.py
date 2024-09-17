"""
    Student template for the third homework of LINMA1691 "Théorie des graphes".

    Authors : Devillez Henri
"""

import math
import heapq


    
def prim_mst(N, roads):
    """
    INPUT :
        - N, the number of crossroads
        - roads, list of tuple (u, v, s) giving a road between u and v with satisfaction s
    OUTPUT :
        - return the maximal satisfaction that can be achieved

        See homework statement for more details
    """

    satisfaction = 0
    isinT = [True ] + [False] * (N - 1)
    heap = []
    lost = []
    heapq.heapify(heap)
    sizeT = 0

    for edge in roads:
        heapq.heappush(heap, Edge(edge))
        satisfaction += edge[2]

    while sizeT < (N - 1):
        minweight_edge = heapq.heappop(heap)
        begin = minweight_edge.begin
        end = minweight_edge.end

        if (isinT[begin] and not isinT[end]) or (isinT[end] and not isinT[begin]):
            isinT[begin] = True
            isinT[end] = True
            satisfaction -= minweight_edge.weight
            sizeT+=1

            for lostedge in lost:
                if not (isinT[lostedge.begin] and isinT[lostedge.end]):
                    heapq.heappush(heap, lostedge)
            lost = []
        else:
            lost.append(minweight_edge)

    return satisfaction


class Edge:
    def __init__(self, edge):
        self.begin = edge[0]
        self.end = edge[1]
        self.weight = edge[2]

    def __lt__(self, other):
        return self.weight < other.weight











    
if __name__ == "__main__":

    # Read Input for the first exercice
    
    with open('in1.txt', 'r') as fd:
        l = fd.readline()
        l = l.rstrip().split(' ')
        
        n, m = int(l[0]), int(l[1])
        
        roads = []
        for road in range(m):
        
            l = fd.readline().rstrip().split()
            roads.append(tuple([int(x) for x in l]))
            
    # Compute answer for the first exercice
     
    ans1 = prim_mst(n, roads)
     
    # Check results for the first exercice

    with open('out1.txt', 'r') as fd:
        l_output = fd.readline()
        expected_output = int(l_output)
        
        if expected_output == ans1:
            print("Exercice 1 : Correct")
        else:
            print("Exercice 1 : Wrong answer")
            print("Your output : %d ; Correct answer : %d" % (ans1, expected_output)) 
        
#exemple de l'enonce:
print(prim_mst(4, [(0, 1, 3), (0, 2, 2), (1, 2, 1), (1, 3, 4), (2, 3, 5)]))



"""

    heap = []
    heapq.heapify(heap)

    # liste d'adjacence, somme des aretes:
    adj, satisfaction = AdjacencyList(N, roads)

    # index des noeuds atteints par l'arbre
    T = [True] + [False] * (N - 1)
    # nombre d'arete rajoutéee
    sizeT = 0

    while sizeT < N - 1:
        for node in T:
            for edge in adj[node]:
                # arete deja ajoutee ? arete adj a plus d'un noeud ?
                if not T[edge[1]]:
                    heapq.heappush(heap, edge)

        minweight_edge = heapq.heappop(heap)

        if not T[minweight_edge[1]]:
            T[minweight_edge[1]] = True
            sizeT += 1
            satisfaction -= minweight_edge[0]

    return satisfaction


"""

