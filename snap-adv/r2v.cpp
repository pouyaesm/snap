#include "stdafx.h"
#include "r2v.h"

void role2vec(PWNet& InNet, const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  const bool& OutputWalks, TVVec<TInt, int64>& WalksVV,
	TIntFltVH& EmbeddingsHV, const double& ParamP, const double& ParamQ,
	const int& roleCount, TFltV& roleWeight,
	const double& stayP, const double& teleportP, const bool& roleNegativeSampling) {
  //Preprocess transition probabilities
  PreprocessTransitionProbs(InNet, ParamP, ParamQ, roleCount, roleWeight, stayP, Verbose);
  TIntV NIdsV;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    NIdsV.Add(NI.GetId());
  }

  //Generate random walks
  int64 AllWalks = (int64)NumWalks * NIdsV.Len();
  WalksVV = TVVec<TInt, int64>(AllWalks,WalkLen);
  TRnd Rnd(time(NULL));
  int64 WalksDone = 0;
  for (int64 i = 0; i < NumWalks; i++) {
    NIdsV.Shuffle(Rnd);
 #pragma omp parallel for schedule(dynamic)
    for (int64 j = 0; j < NIdsV.Len(); j++) {
      if ( Verbose && WalksDone%10000 == 0 ) {
        printf("\rWalking Progress: %.2lf%%",(double)WalksDone*100/(double)AllWalks);fflush(stdout);
      }
      TIntV WalkV;
	    SimulateWalk(InNet, NIdsV[j], WalkLen, Rnd, WalkV, roleCount, teleportP);
      for (int64 k = 0; k < WalkV.Len(); k++) {
        WalksVV.PutXY(i*NIdsV.Len()+j, k, WalkV[k]);
      }
      WalksDone++;
    }
  }
  if (Verbose) {
    printf("\n");
    fflush(stdout);
  }
  //Learning embeddings
  if (!OutputWalks) {
    LearnEmbeddings(WalksVV, Dimensions, WinSize, roleCount, roleNegativeSampling,
		  Iter, Verbose, EmbeddingsHV);
  }
}

