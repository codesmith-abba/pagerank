import networkx as nx

# Example corpus
corpus = {
    "1.html": {"2.html"},
    "2.html": {"3.html"},
    "3.html": {"1.html", "2.html"},
    "4.html": {"2.html", "3.html"}
}

# Create a directed graph
G = nx.DiGraph(corpus)

# Calculate PageRank
pagerank = nx.pagerank(G, alpha=0.85)  # Damping factor = 0.85
print(pagerank)
