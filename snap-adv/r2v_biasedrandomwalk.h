#ifndef N_RAND_WALK_H
#define N_RAND_WALK_H

typedef TNodeEDatNet<TIntIntVFltVPrH, TFlt> TWNet;
typedef TPt<TWNet> PWNet;

// Print the transition matrix per 'out' node for debugging purposes
//void printTransitionProbabilities(PWNet& InNet, const int& typeCount);

///Preprocesses transition probabilities for random walks. Has to be called once before SimulateWalk calls
void PreprocessTransitionProbs(PWNet& InNet, const int& typeCount, TFltV& typeWeight,
	const double& stayP, const bool& verbose);

///Simulates one walk and writes it into Walk vector
void SimulateWalk(PWNet& InNet, int64 StartNId, const int& WalkLen, TRnd& Rnd, TIntV& Walk,
	const int& typeCount, const double& teleportP);

//Predicts approximate memory required for preprocessing the graph
int64 PredictMemoryRequirements(PWNet& InNet);

#endif //N_RAND_WALK_H
