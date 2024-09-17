from collections import deque


class Edge:
    def __init__(self, u, v, capa, weight, residual=None):
        self.u = u
        self.v = v
        self.capa = capa  # capacity that there is left to the edge
        self.weight = weight  # weight of the edge
        self.residual = residual  # corresponding edge in the residual graph

    def __str__(self):
        return f'({self.u})--->({self.v}) with capa {self.capa} and weight {self.weight}'


def create_graph(capacities, costs, green_sources: dict, gas_centrals: dict, consumers: dict):
    #TODO

    nb_node = len(capacities)

    #unused numbers for source and target:
    s = nb_node
    t = s + 1
    graph = [[] for _ in range(t)]

    #computes intermediate edges:
    for i in range(nb_node):
        for j in range(nb_node):
            cout = costs[i][j]
            capa = capacities[i][j]
            if capa != 0:
                #res = Edge(j, i, 0, -cout)
                e = Edge(i, j, capa, cout)
                graph[i].append(e)



    #transforms into SISO network:
    create_source(s, green_sources, gas_centrals, graph)
    create_target(t, consumers, graph)


    return s, t, graph


def create_source(s, green_sources, gas_centrals, graph):
    """
    Makes the network SI
    deals with capacity and weights assigned to nodes by creating edges btw s and (green) centrals
    """

    # Green source:
    for source in green_sources.keys():
        #res = Edge(source, s, 0, 0)
        e_green = Edge(s, source, green_sources[source], 0)
        graph[s].append(e_green)


    # Gas:
    for source in gas_centrals.keys():
        for i in range(1, len(gas_centrals[source])):
            pente = (gas_centrals[source][i][1] - gas_centrals[source][i-1][1])/(gas_centrals[source][i][0] - gas_centrals[source][i-1][0])
            capa = gas_centrals[source][i][0] - gas_centrals[source][i-1][0]
            #res = Edge(source, s, 0, -pente)
            e_gas = Edge(s, source, capa, pente)
            graph[s].append(e_gas)


    return


def create_target(t, consumers, graph):
    """
    Makes the network SO
    deals with capacity on consumer nodes by creating edges of given capacity btw consumers and t
    """
    for target in consumers.keys():
        #res = Edge(t, target, 0, 0)
        e_target = Edge(target, t, consumers[target], 0)
        graph[target].append(e_target)

    return



def get_residual(graph):
    # TODO
    graph_residual = [[] for _ in range(len(graph) + 2)]

    for i in range(len(graph)):
        for j in range(len(graph[i])):
            e = graph[i][j]
            res = Edge(e.v, e.u, 0, -e.weight)
            e.residual = res
            graph_residual[e.u].append(e)
            graph_residual[e.v].append(res) #ca change legit r dans les tests de le mettre ou pas


    return graph_residual






def min_cost_max_flow(s, t, graph_residual):
    # TODO

    nb_node = len(graph_residual)
    parent = [-1] * (nb_node)  # to get our shortest path

    def BellmanFord():

        distance = [float('Inf')] * (nb_node)
        distance[s] = 0
        Q = deque([s])
        in_Q = [False] * (nb_node)
        in_Q[s] = True
        while Q:
            u = Q.popleft()
            in_Q[u] = False
            for edge in graph_residual[u]:
                if edge is not None and edge.capa !=0:
                    if edge.u == u:
                        v = edge.v
                        if edge.weight + distance[u] < distance[v]:
                            distance[v] = edge.weight + distance[u]
                            parent[v] = edge
                            if not in_Q[v]:
                                Q.append(v)
                                in_Q[v] = True




        return distance[t] != float("Inf") #boolean qui indique s'il existe encore des chemins augmentant, cad si t a ete visited


    maximum_flow = 0
    minimum_cost = 0

    while BellmanFord():

        source = t
        path_flow = float("Inf")
        while source != s: #remonte le chemin augmentant de target a source

            #On fait passer le minimum des capa des e du chemin
            path_flow = min(path_flow, parent[source].capa)
            source = parent[source].u


        #max flow update
        maximum_flow += path_flow



        #residual graph update (changer les capa des aretes visited via path_flow):
        v = t
        while v != s:

            parent[v].capa -= path_flow
            #cout marginal donc on multiplie par le flot qui est passé:
            minimum_cost += path_flow*parent[v].weight
            if parent[v].residual is not None:
                parent[v].residual.capa += path_flow

            v = parent[v].u


    return maximum_flow, minimum_cost



















