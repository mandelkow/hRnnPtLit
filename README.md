# hRnnPtLit : A minimalis definition of RNNs (LSTM,GRU) using PyTorch and Lightning.

## `hRnn( MdlPar:list )`
A sequential stack of RNN layers is defined by a list of strings and numbers:
 - 1st int = input dim
 - ['lstm','gru','linear','bilinear'] : set type for subsequent layers
 - ['tanh','relu','lelu','ident','noact'] : set activation function following layer
 - ['bidir','bidir0'] : set layer parameter bidirectional = True / False
 - int(16) : add layer of currently set type and output dim=16
 - '16' : same as int(16)
 - '8r2' : layer of size 8 with parameter num_layers = 2, i.e. repeat layer x2
 - '8p1' : layer of size 8 with parameter proj_size = 1, i.e. add linear with output dim=1


#### EXAMPLES:
```python
Mdl = hRnn([2,'lstm','relu',32,16,8,'linear','ident',1])
# = InputDim:2, 3 layers LSTM+ReLu of dims:[32,16,8], Linear(+no activation) output dim: 1

Mdl = hRnn([2,'lstm','relu','32r2','ident','16p1'])
# = InputDim:2, 2 layers LSTM+ReLu of dims:[32,32], 1 layer LSTM+projection(linear)(+no activation) output dim: 1
```