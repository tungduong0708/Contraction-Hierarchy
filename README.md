# Contraction Hierarchies Implementation

This project implements Contraction Hierarchies, an efficient algorithm for solving the shortest path problem in large road networks. The implementation is based on the guide from [Contraction Hierarchies Guide](https://jlazarsfeld.github.io/ch.html).

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Algorithm Overview](#algorithm-overview)
7. [Performance](#performance)
8. [Contributing](#contributing)
   
## Introduction

Contraction Hierarchies is a speed-up technique for computing shortest paths in road networks. It consists of two main processes: preprocessing and querying. The preprocessing phase creates a hierarchical structure that allows for faster query times compared to traditional shortest path algorithms.

## Features

- Efficient preprocessing of road network graphs
- Fast shortest path queries using bidirectional Dijkstra's algorithm
- Support for large-scale road networks
- Customizable node ordering strategies

## Requirements

- Python 3.x
- NetworkX library for graph operations

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/tungduong0708/Contraction-Hierarchy.git
   cd Contraction-Hierarchy
   ```

2. Install the required NetworkX library:
   ```
   pip install networkx
   ```

## Usage

To use the Contraction Hierarchies implementation:

1. Ensure you're in the project directory.
2. Run the main script:
   ```
   python main.py
   ```

This will execute the Contraction Hierarchies algorithm on the provided graph.

## Algorithm Overview

The Contraction Hierarchies algorithm consists of two main processes:

1. **Preprocessing**: 
   - Nodes are contracted in a specific order
   - Shortcut edges are added to preserve shortest paths
   - A hierarchical structure is created
   
   During this phase, the algorithm:
   - Selects nodes to contract based on a chosen strategy
   - For each contracted node, adds necessary shortcuts between its neighbors
   - Updates the graph structure to reflect the contraction

2. **Query**: 
   - Bidirectional Dijkstra's algorithm is used for querying
   - The search is performed on the preprocessed graph
   - The search only needs to go "upwards" in the hierarchy

   The query process works as follows:
   - Starts a forward search from the source node
   - Starts a backward search from the target node
   - Both searches only consider edges leading to higher-ranked nodes
   - The search terminates when the two search spaces meet
   - The shortest path is reconstructed from the meeting point

Complexity:
- Preprocessing time: O(|E + V| log |V|), where |E| is the number of edges and |V| is the number of vertices in the graph.
- Query time: O(|E + V| log |V|) in the worst case, but typically much faster in practice due to the hierarchical structure.

By using bidirectional Dijkstra for querying, we can significantly reduce the search space and improve query times compared to standard Dijkstra's algorithm. The hierarchical structure created during preprocessing allows for efficient "upward" searches, which dramatically reduces the number of nodes explored during a query.

In conclusion, Contraction Hierarchies offer a powerful trade-off between preprocessing time and query efficiency, making them particularly well-suited for large-scale road networks where multiple queries need to be performed.

## Performance

Our implementation of Contraction Hierarchies shows significant improvements in query time and efficiency compared to traditional pathfinding algorithms. Here's a comparison of performance metrics:

| Algorithm | Preprocess Time (s) | Query Time (s) | Nodes checked |
|-----------|---------------------|----------------|---------------|
| A* | 0.0 | 0.0184 | 2,799 |
| Dijkstra | 0.0 | 0.0156 | 4,397 |
| Contraction Hierarchies | 17.4383 | 0.0032 | 226 |

These results were obtained using a sample graph with roughly 5000 nodes and 10000 edges.

Key observations:

1. **Preprocessing Time**: Contraction Hierarchies requires a significant preprocessing time (17.4383 seconds) to build the hierarchical structure. This is a one-time cost that enables faster subsequent queries.

2. **Query Time**: 
   - Contraction Hierarchies: 0.0032 seconds (3.2 milliseconds)
   - A*: 0.0184 seconds
   - Dijkstra: 0.0156 seconds

   Contraction Hierarchies achieves a query time that is nearly 6 times faster than A* and nearly 5 times faster than Dijkstra's algorithm.

3. **Nodes Checked**: 
   - Contraction Hierarchies: 226 nodes
   - A*: 2,799 nodes
   - Dijkstra: 4,397 nodes

   The number of nodes that need to be checked during a query is significantly reduced with Contraction Hierarchies:
   - Nearly 20 times fewer nodes compared to Dijkstra's algorithm
   - Over 14 times fewer nodes compared to A*

In conclusion, by investing in a modest preprocessing time, Contraction Hierarchies dramatically reduces the number of nodes that need to be checked during queries. This results in significantly faster query times, making it an excellent choice for applications requiring frequent shortest path computations on large road networks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to the [Contraction-Hierarchy repository](https://github.com/tungduong0708/Contraction-Hierarchy).
