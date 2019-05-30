========================================================================
    Role2vec
========================================================================

role2vec is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. 

The code works under Windows with Visual Studio or Cygwin with GCC,
Mac OS X, Linux and other Unix variants with GCC. Make sure that a
C++ compiler is installed on the system. Visual Studio project files
and makefiles are provided. For makefiles, compile the code with
"make all".

/////////////////////////////////////////////////////////////////////////////

Parameters:
Input graph path (-i:)
Output graph path (-o:)
Number of dimensions. Default is 128 (-d:)
Length of walk per source. Default is 80 (-l:)
Number of walks per source. Default is 10 (-r:)
Context size for optimization. Default is 10 (-k:)
Number of epochs in SGD. Default is 1 (-e:)
Return hyperparameter. Default is 1 (-p:)
Inout hyperparameter. Default is 1 (-q:)
Number of node roles in bipartite graph (including role 'out'). Default is 1. (-rc:)
Probability of staying in the same node role. For a negative value, node2vec probabilities are used. Default is -1. (-sp:)
Weight of each role (role 0 'out' is excluded) to be chosen by random walker, nth weight is for nodes with role 'n' having id = tc * i + n. Default is '1,1,..'. (-rw:)
Probability of teleporting from an 'in' node to its corresponding 'out' node. Default is 0. (-tp:)
Role based negative sampling. (-rn)
Verbose output. (-v)
Graph is directed. (-dr)
Graph is weighted. (-w)
Output random walks instead of embeddings. (-ow)

/////////////////////////////////////////////////////////////////////////////

Usage:
./role2vec -i:graph/small.edgelist -o:emb/small.emb -l:3 -d:24 -rc:3 -rw:1,1 -sp:0.74 -dr -v
