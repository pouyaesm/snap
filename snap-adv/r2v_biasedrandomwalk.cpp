#include "stdafx.h"
#include "Snap.h"
#include "r2v_biasedrandomwalk.h"


/*
 * Preprocess alias sampling method
 * In computing, the alias method is a family of efficient algorithms
 * for sampling from a discrete probability distribution
 */
void GetNodeAlias(TFltV& PTblV, TIntVFltVPr& NTTable) {
  int64 N = PTblV.Len();
  TIntV& KTbl = NTTable.Val1;
  TFltV& UTbl = NTTable.Val2;
	// initialize (Key, Value) pairs
  for (int64 i = 0; i < N; i++) {
    KTbl[i]=0;
    UTbl[i]=0;
  }
  TIntV UnderV;
  TIntV OverV;
  for (int64 i = 0; i < N; i++) {
    UTbl[i] = PTblV[i]*N;
    if (UTbl[i] < 1) {
      UnderV.Add(i);
    } else {
      OverV.Add(i);
    }
  }
  while (UnderV.Len() > 0 && OverV.Len() > 0) {
    int64 Small = UnderV.Last();
    int64 Large = OverV.Last();
    UnderV.DelLast();
    OverV.DelLast();
    KTbl[Small] = Large;
    UTbl[Large] = UTbl[Large] + UTbl[Small] - 1;
    if (UTbl[Large] < 1) {
      UnderV.Add(Large);
    } else {
      OverV.Add(Large);
    }
  }
  while(UnderV.Len() > 0){
    int64 curr = UnderV.Last();
    UnderV.DelLast();
    UTbl[curr]=1;
  }
  while(OverV.Len() > 0){
    int64 curr = OverV.Last();
    OverV.DelLast();
    UTbl[curr]=1;
  }

}

int64 AliasDrawInt(TIntVFltVPr& NTTable, TRnd& Rnd) {
  int64 N = NTTable.GetVal1().Len();
  TInt X = static_cast<int64>(Rnd.GetUniDev()*N);
  double Y = Rnd.GetUniDev();
//	printf("Val1: ");
//	for (int i = 0 ; i < N; i++){
//		if(i == X)
//				printf("[%.3f]\t", NTTable.GetVal1()[i]);
//		else
//				printf("%.3f\t", NTTable.GetVal1()[i]);
//	}
//	printf("\nVal2: ");
//	for (int i = 0 ; i < N; i++){
//		if(i == X)
//				printf("[%.3f]\t", NTTable.GetVal2()[i]);
//		else
//			printf("%.3f\t", NTTable.GetVal2()[i]);
//	}
//	printf("\nY: %.3f\n", Y);
  return Y < NTTable.GetVal2()[X] ? X : NTTable.GetVal1()[X];
}

TFltV* calculateProbabilityTableNode2Vec(PWNet& InNet, const double& ParamP, const double& ParamQ,
 const int& preId, const int& curId, THash <TInt, TBool>& preNeighbors, const bool& Verbose) {
	TWNet::TNodeI curI = InNet->GetNI(curId);
  double Psum = 0;
  TFltV* PTable = new TFltV(); //Probability distribution table
	//for each node x in t (pre) -> v (cur) -> x (next)
  for (int64 j = 0; j < curI.GetOutDeg(); j++) {
    int64 nextId = curI.GetNbrNId(j);
    TFlt Weight;
    if (!(InNet->GetEDat(curI.GetId(), nextId, Weight))){ continue; }
    if (nextId==preId) { // d(t, x) == 0 => 1/p
      PTable->Add(Weight / ParamP);
      Psum += Weight / ParamP;
    } else if (preNeighbors.IsKey(nextId)) { // d(t, x) == 1 => 1
      PTable->Add(Weight);
      Psum += Weight;
    } else { // d(t, x) == 2 => 1/q
      PTable->Add(Weight / ParamQ);
      Psum += Weight / ParamQ;
    }
  }
  //Normalizing table
  for (int64 j = 0; j < curI.GetOutDeg(); j++) {
    PTable->SetVal(j, PTable->GetVal(j) / Psum);
  }
	// Print for debugging
//	printf("Psum: %.2f\n", Psum);
//  for (int64 n = 0; n < curI.GetOutDeg(); n++) {  //for each 'next' node
//    int64 nextId = curI.GetNbrNId(n);
//    TFlt edgeWeight;
//    InNet->GetEDat(curId, nextId, edgeWeight);
//	  printf("Pr(n = %d | c = %d, p = %d): %.3f,\tw(%d, %d):  %.2f\n",
//		  nextId, curId, preId, PTable->GetVal(n), curId, nextId, edgeWeight);
//  }
//	fflush(stdout);
	return PTable;
}

TFltV* calculateProbabilityTableRoleOut(PWNet& InNet, const int& roleCount, TFltV& roleWeight, const double& stayP,
 const int& preId, const int& curId, const bool& Verbose){
	TWNet::TNodeI curI = InNet->GetNI(curId);
	int preRole = preId % roleCount - 1; // role 'out' is excluded
	// probability denominator for same-role normalization
	double Psame = 0;
	// probability denominator for different-role normalization
	double Pdiff = 0;
	//Probability distribution table (pre, next) in "pre -> cur -> next" path
  TFltV* PTable = new TFltV();
  for (int64 n = 0; n < curI.GetOutDeg(); n++) {  //for each 'next' node
    int64 nextId = curI.GetNbrNId(n);
		int nextRole = nextId % roleCount - 1; // role 'out' is excluded
    TFlt edgeWeight;
    InNet->GetEDat(curId, nextId, edgeWeight);
    if (preRole == nextRole) {
      PTable->Add(stayP * edgeWeight);
      Psame += edgeWeight;
    } else {
	    double totalWeight = edgeWeight * roleWeight[nextRole];
      PTable->Add((1.0 - stayP) * totalWeight);
      Pdiff += totalWeight;
    }
  }
  //Normalizing probabilities
  for (int64 n = 0; n < curI.GetOutDeg(); n++) {
	  int nextRole = curI.GetNbrNId(n) % roleCount - 1;
	  if(preRole == nextRole){
			PTable->SetVal(n, PTable->GetVal(n) / Psame);
	  }else{
		  PTable->SetVal(n, PTable->GetVal(n) / Pdiff);
	  }
  }
	// Print for debugging
//	printf("Psame: %.2f, Pdiff: %.2f\n", Psame, Pdiff);
//  for (int64 n = 0; n < curI.GetOutDeg(); n++) {  //for each 'next' node
//    int64 nextId = curI.GetNbrNId(n);
//	  int nextRole = nextId % roleCount - 1; // role 'out' is excluded
//    TFlt edgeWeight;
//    InNet->GetEDat(curId, nextId, edgeWeight);
//	  printf("Pr(n = %d[%d] | c = %d, p = %d[%d]): %.3f,\tw(%d, %d):  %.2f\n",
//		  nextId, nextRole, curId, preId, preRole, PTable->GetVal(n), curId, nextId, edgeWeight);
//  }
//	fflush(stdout);
	return PTable;
 }

// Probability of going from a node of role 'in' to a node of role 'out'
// is independent of the pre node before 'in'
TFltV* calculateProbabilityTableRoleIn(PWNet& InNet, const int& preId, const int& curId,
	const bool& Verbose){
	TWNet::TNodeI curI = InNet->GetNI(curId);
	double Psum = 0; // probability denominator for normalization
	//Probability distribution table (pre, next) in "pre -> cur -> next" path
  TFltV* PTable = new TFltV();
  for (int64 n = 0; n < curI.GetOutDeg(); n++) {  //for each 'next' node
    int64 nextId = curI.GetNbrNId(n);
    TFlt edgeWeight;
    InNet->GetEDat(curId, nextId, edgeWeight);
    PTable->Add(edgeWeight);
    Psum += edgeWeight;
  }
  //Normalizing probabilities
  for (int64 n = 0; n < curI.GetOutDeg(); n++) {
	  PTable->SetVal(n,  PTable->GetVal(n) / Psum);
  }
	// Print for debugging
//	printf("Psum: %.2f\n", Psum);
//  for (int64 n = 0; n < curI.GetOutDeg(); n++) {  //for each 'next' node
//    int64 nextId = curI.GetNbrNId(n);
//    TFlt edgeWeight;
//    InNet->GetEDat(curId, nextId, edgeWeight);
//	  printf("Pr(n = %d | c = %d, p = %d): %.3f,\tw(%d, %d):  %.2f\n",
//		  nextId, curId, preId, PTable->GetVal(n), curId, nextId, edgeWeight);
//  }
//	fflush(stdout);
	return PTable;
 }

void PreprocessNode (PWNet& InNet, const double& ParamP, const double& ParamQ,
	const int& roleCount, TFltV& roleWeight, const double& stayP,
 TWNet::TNodeI preI, int64& NCnt, const bool& Verbose) {
  if (Verbose && NCnt%100 == 0) {
    printf("\rPreprocessing progress: %.2lf%% ",(double)NCnt*100/(double)(InNet->GetNodes()));fflush(stdout);
  }
	int preId = preI.GetId();
//	printf("\ncurId: %d\n", curId);
	//Neighbors of pre (used in node2vec transition probablity calculation)
  THash <TInt, TBool> preNeighbors;
  for (int64 i = 0; i < preI.GetOutDeg(); i++) {
    preNeighbors.AddKey(preI.GetNbrNId(i));
  }
	for(int c = 0 ; c < preI.GetOutDeg() ; c++){
		int curId = preI.GetNbrNId(c);
		TFltV* PTable;
		if(stayP < 0){ // use node2vec transition probabilities
			PTable = calculateProbabilityTableNode2Vec(InNet, ParamP, ParamQ,
				preId, curId, preNeighbors, Verbose);
		} else if (curId % roleCount == 0) { // specialized transition from out to in[type]
			PTable = calculateProbabilityTableRoleOut(InNet, roleCount, roleWeight, stayP,
				preId, curId, Verbose);
		} else { // uniform transition from in[type] to out
			PTable = calculateProbabilityTableRoleIn(InNet, preId, curId, Verbose);
		}
		GetNodeAlias(*PTable, InNet->GetNI(curId).GetDat().GetDat(preId));
		delete PTable;
	}
  NCnt++;
}

/*
 * Preprocess transition probabilities for each path "pre -> cur -> next"
 **/
void PreprocessTransitionProbs(PWNet& InNet, const double& ParamP, const double& ParamQ,
	const int& roleCount, TFltV& roleWeight, const double& stayP, const bool& Verbose) {
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    InNet->SetNDat(NI.GetId(),TIntIntVFltVPrH());
  }
	/*
	 * Each cur node has a (pre, next) matrix
	 * For 'out' cur node: (pre, next) matrix holds Pr(next | pre, cur)
	 * For 'in' cur node: (pre, next) matrix holds P(next | cur)
	 * repeated for all pres (next would definitely be an 'out' node)
	 **/
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
	  int64 preId = NI.GetId();
	  for (int64 i = 0; i < NI.GetOutDeg(); i++) {  //allocating space in advance to avoid issues with multithreading
      int64 curId = NI.GetNbrNId(i);
	    TWNet::TNodeI curI = InNet->GetNI(curId);
      curI.GetDat().AddDat(preId,
			  TPair<TIntV,TFltV>(TIntV(curI.GetOutDeg()),TFltV(curI.GetOutDeg())));
		  // printf("data created for (pre, cur): (%d, %d)\n", preId, curId);
    }
  }
  int64 NCnt = 0;
  TIntV NIds;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    NIds.Add(NI.GetId());
  }
 // #pragma omp parallel for schedule(dynamic)
  for (int64 i = 0; i < NIds.Len(); i++) {
	  TWNet::TNodeI preI = InNet->GetNI(NIds[i]);
    PreprocessNode(InNet, ParamP, ParamQ, roleCount, roleWeight, stayP, preI, NCnt, Verbose);
  }
//  printTransitionProbabilities(InNet, roleCount);
  if(Verbose){ printf("\n"); }
}

int64 PredictMemoryRequirements(PWNet& InNet) {
  int64 MemNeeded = 0;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    for (int64 i = 0; i < NI.GetOutDeg(); i++) {
      TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
      MemNeeded += CurrI.GetOutDeg()*(sizeof(TInt) + sizeof(TFlt));
    }
  }
  return MemNeeded;
}

// Simulates a random walk and returns the ordered list of visited nodes
void SimulateWalk(PWNet& InNet, int64 StartNId, const int& WalkLen, TRnd& Rnd, TIntV& WalkV,
	const int& roleCount, const double& teleportP) {
  WalkV.Add(StartNId);
  if (WalkLen == 1) { return; }
  if (InNet->GetNI(StartNId).GetOutDeg() == 0) { return; }
  WalkV.Add(InNet->GetNI(StartNId).GetNbrNId(Rnd.GetUniDevInt(InNet->GetNI(StartNId).GetOutDeg())));
  while (WalkV.Len() < WalkLen) {
    int64 Cur = WalkV.Last();
    int64 Pre = WalkV.LastLast();
    if (InNet->GetNI(Cur).GetOutDeg() == 0) { return; }
	  // Jump from in[role] to out node (which is divisible by roleCount) with probablity teleportP
	  // and select a random next node to be set as the new Cur
	  if(Cur % roleCount > 0 &&  Rnd.GetUniDev() < teleportP) {
		  int SrcTemp = Cur - Cur % roleCount; // go from in[role] to out of the same original node
		  if (InNet->IsNode(SrcTemp)) {
			  Pre = SrcTemp;
			  Cur = InNet->GetNI(Pre).GetNbrNId(Rnd.GetUniDevInt(InNet->GetNI(Pre).GetOutDeg()));
			  WalkV.Add(Pre);
			  WalkV.Add(Cur);
		  }
	  }
	  int64 Next = AliasDrawInt(InNet->GetNDat(Cur).GetDat(Pre), Rnd);
	  int64 NextNId = InNet->GetNI(Cur).GetNbrNId(Next);
//	  printf("(%d, %d) -> %d[%d]\n", Pre, Cur, NextNId, Next);
    WalkV.Add(NextNId);
  }
	// Remove extra walks. In one pass of the while loop more than one node might be added
	while(WalkV.Len() > WalkLen) {
			WalkV.DelLast();
	}
//	printf("walk %d (%d, %.2f): ", WalkV.Len(), roleCount, teleportP);
//  for (int i = 0; i < WalkV.Len(); i++) {
//	  printf("%d ", WalkV[i]);
//  }
//  printf("\n");
//  fflush(stdout);
}