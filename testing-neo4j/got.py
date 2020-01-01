from py2neo import authenticate, Graph
from igraph import Graph as IGraph

authenticate("localhost:7474", "neo4j", "Paparasta1+")
graph = Graph("http://localhost:7474/db/data/")


query = '''
MATCH (c1:Character)-[r:INTERACTS]->(c2:Character)
RETURN c1.name, c2.name, r.weight AS weight
'''

ig = IGraph.TupleList(graph.run(query), weights=True)

"""
pg = ig.pagerank()
pgvs = []
for p in zip(ig.vs, pg):
    print(p)
    pgvs.append({"name": p[0]["name"], "pg": p[1]})
pgvs

write_clusters_query = '''
UNWIND {nodes} AS n
MATCH (c:Character) WHERE c.name = n.name
SET c.pagerank = n.pg
'''

graph.run(write_clusters_query, nodes=pgvs)
"""


clusters = IGraph.community_walktrap(ig, weights="weight").as_clustering()

nodes = [{"name": node["name"]} for node in ig.vs]
for node in nodes:
    idx = ig.vs.find(name=node["name"]).index
    node["community"] = clusters.membership[idx]

write_clusters_query = '''
UNWIND {nodes} AS n
MATCH (c:Character) WHERE c.name = n.name
SET c.community = toInt(n.community)
'''

graph.run(write_clusters_query, nodes=nodes)