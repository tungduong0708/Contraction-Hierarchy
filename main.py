from Path import *
from Stop import *
from RouteVar import *
from Graph import *

import time
import openai
import json
import networkx as nx
from pyproj import Transformer
from rtree import *

if __name__ == '__main__':
    graph = Graph()
    time1 = time.time()
    graph.make_graph_demo()
    time2 = time.time()
    graph.get_shortest_path_dijkstra('B', 'G')
    time3 = time.time()

    print("Time making graph: ", time2 - time1)
    print('Executing time: ', time3 - time2)