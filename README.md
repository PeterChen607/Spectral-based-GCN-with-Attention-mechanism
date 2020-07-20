# Spectral-based-GCN-with-Attention-mechanism
This is an experiment on attention mechanism in spectral based GCN.

The code is based on normal spectral based GCN and GAT.

Original source code & data: 
  GCN: https://github.com/tkipf/gcn
  GAT: https://github.com/PetarV-/GAT
  
  
Dependencies:
- `Python==3.6.2`
- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`

Modification:
  Here I import the attention layer in GAT and put it into the GCN layer in layers.py
  If you want to change the dataset, don't forget to change the 'nb_nodes' variable in layers.py
  Values for 'nb_nodes':
    cora: 2708
    pubmed: 19717
    citeseer: 3328
  
  
