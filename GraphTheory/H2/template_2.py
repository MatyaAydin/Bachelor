"""
    Student template for the third homework of LINMA1691 "Théorie des graphes".

    Authors : Devillez Henri
"""


import math
import random

class Union_Find():
    """
    Disjoint sets data structure for Kruskal's or Karger's algorithm.
  
    It is useful to keep track of connected components (find(a) == find(b) iff they are connected).
    same root <=> in the same subtree
  
    """
    
    def __init__(this, N):
        """
        Corresponds to MakeSet for all the nodes
        INPUT :
            - N : the initial number of disjoints sets
        """
        this.N = N
        this.p = list(range(N))
        this.size = [1]*N
        
    def union(this, a, b):
        """
        INPUT :
            - a and b : two elements such that 0 <= a, b <= N-1
        OUTPUT:
            - return nothing
            
        After the operation, the two given sets are merged
        """

        a = this.find(a)
        b = this.find(b)

        # deja connecte: fusion change rien
        if a == b:
            return
       
        # Swap variables if necessary
        if this.size[a] > this.size[b]:
            a,b = b,a
        
        this.size[b] += this.size[a]
        this.p[a] = b
        
    def find(this, a):
        """
        INPUT :
            - a : one element such that 0 <= a <= N-1
        OUTPUT:
            - return the root of the element a
        """
        if a != this.p[a]: this.p[a] = this.find(this.p[a])
        return this.p[a]
    

def min_cut(N, edges):
    """ 
    INPUT : 
        - N the number of nodes
        - edges, list of tuples (u, v) giving an edge between u and v

    OUTPUT :
        - return an estimation of the min cut of the graph
        
    This method has to return the correct answer with probability bigger than 0.999
    See project homework for more details
    """
    def karger(N, edges):

        """ 
        INPUT : 
            - N the number of nodes
            - edges, list of tuples (u, v) giving an edge between u and v

        OUTPUT :
            - return an estimation of the min cut of the graph
              
        See project homework for more details
        """

        size = len(edges)
        available_edges = [True] * size
        this_min_cut = 0
        set = Union_Find(N)
        contracted_nodes = [False] * N

        #print("edges = ", edges)


        #randomly adds edges until 2 forests are created
        while set.N > 2:
            #print("size = ", set.size)
            #print(" p = ", set.p)
            rand_idx = random.randint(0, size - 1)
            rand_edge = edges[rand_idx]
            #print("rand edge = ", rand_edge)
            if not available_edges[rand_idx]:
                pass

            begin = rand_edge[0]
            end = rand_edge[1]

            if(set.find(begin) != set.find(end)):
                set.union(begin, end)
                set.N -= 1


            available_edges[rand_idx] = False
            #print("--------------------------------------")

        #print("end of while state:")
        #print("size = ", set.size)
        #print(" p = ", set.p)


        #count edges separating the 2 trees:
        for i in range(size):
            begin = edges[i][0]
            end = edges[i][1]
            if  (set.find(begin) != set.find(end)): #available_edges[i]
                this_min_cut +=1

        
        # TO COMPLETE

        return this_min_cut


    #pour estimer k:
    p = (2 * math.factorial(N - 2))/(math.factorial(N))
    print(" p =", p)

    size = len(edges)
    result = [0 for _ in range(size)]

    n_trials = 1000
    for i in range(n_trials):
        pred = karger(N, edges)
        result[pred] += 1/n_trials

    candidats = [i for i in range(size) if result[i] >= p]
    #print("candidats = ", candidats)
    
    return candidats[0]
    

if __name__ == "__main__":

    # Read Input for the second exercice
    
    with open('in2.txt', 'r') as fd:
        l = fd.readline()
        l = l.rstrip().split(' ')
        
        n, m = int(l[0]), int(l[1])
        
        edges = []
        for edge in range(m):
        
            l = fd.readline().rstrip().split()
            edges.append(tuple([int(x) for x in l]))
            
    # Compute answer for the second exercice
     
    ans = min_cut(n, edges)
     
    # Check results for the second exercice

    with open('out2.txt', 'r') as fd:
        l_output = fd.readline()
        expected_output = int(l_output)
        
        if expected_output == ans:
            print("Exercice 2 : Correct")
        else:
            print("Exercice 2 : Wrong answer")
            print("Your output : %d ; Correct answer : %d" % (ans, expected_output)) 

