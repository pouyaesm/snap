#include "stdafx.h"
#include "Snap.h"
#include "r2v_word2vec.h"

//Code from https://github.com/nicholas-leonard/word2vec/blob/master/word2vec.c
//Customized for SNAP and node2vec

// Number of times each word Vocab[i] is repeated in the collection of random walks WalksVV
void LearnVocab(TVVec<TInt, int64>& WalksVV, TIntV& Vocab) {
  for( int64 i = 0; i < Vocab.Len(); i++) { Vocab[i] = 0; }
  for( int64 i = 0; i < WalksVV.GetXDim(); i++) {
    for( int j = 0; j < WalksVV.GetYDim(); j++) {
      Vocab[WalksVV(i,j)]++;
    }
  }
}

//Precompute unigram table using alias sampling method
void InitUnigramTable(TIntV& Vocab, TIntV& KTable, TFltV& UTable) {
  double TrainWordsPow = 0;
  double Pwr = 0.75;
  TFltV ProbV(Vocab.Len());
  for (int64 i = 0; i < Vocab.Len(); i++) {
    ProbV[i]=TMath::Power(Vocab[i],Pwr);
    TrainWordsPow += ProbV[i];
    KTable[i]=0;
    UTable[i]=0;
  }
  for (int64 i = 0; i < ProbV.Len(); i++) {
    ProbV[i] /= TrainWordsPow;
  }
  TIntV UnderV;
  TIntV OverV;
  for (int64 i = 0; i < ProbV.Len(); i++) {
    UTable[i] = ProbV[i] * ProbV.Len();
    if ( UTable[i] < 1 ) {
      UnderV.Add(i);
    } else {
      OverV.Add(i);
    }
  }
  while(UnderV.Len() > 0 && OverV.Len() > 0) {
    int64 Small = UnderV.Last();
    int64 Large = OverV.Last();
    UnderV.DelLast();
    OverV.DelLast();
    KTable[Small] = Large;
    UTable[Large] = (UTable[Large] + UTable[Small]) - 1;
    if (UTable[Large] < 1) {
      UnderV.Add(Large);
    } else {
      OverV.Add(Large);
    }
  }
  while(UnderV.Len() > 0){
    int64 curr = UnderV.Last();
    UnderV.DelLast();
    UTable[curr]=1;
  }
  while(OverV.Len() > 0){
    int64 curr = OverV.Last();
    OverV.DelLast();
    UTable[curr]=1;
  }
}

int64 RndUnigramInt(TIntV& KTable, TFltV& UTable, TRnd& Rnd) {
  TInt X = static_cast<int64>(Rnd.GetUniDev()*KTable.Len());
  double Y = Rnd.GetUniDev();
  return Y < UTable[X] ? X : KTable[X];
}

//Initialize negative embeddings
void InitNegEmb(TIntV& Vocab, const int& Dimensions, TVVec<TFlt, int64>& SynNeg) {
  SynNeg = TVVec<TFlt, int64>(Vocab.Len(),Dimensions);
  for (int64 i = 0; i < SynNeg.GetXDim(); i++) {
    for (int j = 0; j < SynNeg.GetYDim(); j++) {
      SynNeg(i,j) = 0;
    }
  }
}

//Initialize positive embeddings
void InitPosEmb(TIntV& Vocab, const int& Dimensions, TRnd& Rnd, TVVec<TFlt, int64>& SynPos) {
  SynPos = TVVec<TFlt, int64>(Vocab.Len(),Dimensions);
  for (int64 i = 0; i < SynPos.GetXDim(); i++) {
    for (int j = 0; j < SynPos.GetYDim(); j++) {
      SynPos(i,j) =(Rnd.GetUniDev()-0.5)/Dimensions;
    }
  }
}

void TrainModel(TVVec<TInt, int64>& WalksVV, TIntIntH& RnmH, TIntIntH& RnmBackH, const int& Dimensions,
    const int& WinSize, const int& roleCount, const bool& roleNegativeSampling,
		const int& Iter, const bool& Verbose,
    TIntV& KTable, TFltV& UTable, int64& WordCntAll, TFltV& ExpTable,
    double& Alpha, int64 CurrWalk, TRnd& Rnd,
    TVVec<TFlt, int64>& SynNeg, TVVec<TFlt, int64>& SynPos)  {
  TFltV Neu1eV(Dimensions);
  int64 AllWords = WalksVV.GetXDim()*WalksVV.GetYDim();
  TIntV WalkV(WalksVV.GetYDim());
  for (int j = 0; j < WalksVV.GetYDim(); j++) { WalkV[j] = WalksVV(CurrWalk,j); }
  for (int64 WordI=0; WordI<WalkV.Len(); WordI++) {
    if ( WordCntAll%10000 == 0 ) {
      if ( Verbose ) {
        printf("\rLearning Progress: %.2lf%% ",(double)WordCntAll*100/(double)(Iter*AllWords));
        fflush(stdout);
      }
      Alpha = StartAlpha * (1 - WordCntAll / static_cast<double>(Iter * AllWords + 1));
      if ( Alpha < StartAlpha * 0.0001 ) { Alpha = StartAlpha * 0.0001; }
    }
	  // This algorithm first fixes an output word then iterates on input words
	  // opposite to paper's approach where input word is central
	  // and output words are chosen from its neighborhood
    int64 OutWord = WalkV[WordI]; // output word
	  int64 OutWordOriginal = RnmBackH.GetDat(OutWord); // word id before normalization
	  int OutWordRole = OutWordOriginal % roleCount;
	  int64 baseWord = OutWordOriginal / roleCount;
    for (int i = 0; i < Dimensions; i++) {
      Neu1eV[i] = 0;
    }
    int Offset = Rnd.GetUniDevInt() % WinSize;
    for (int a = Offset; a < WinSize * 2 + 1 - Offset; a++) {
      if (a == WinSize) { continue; } // output word itself at index "WordI - WinSize + WinSize"
      int64 InWordI = WordI - WinSize + a;
      if (InWordI < 0){ continue; }
      if (InWordI >= WalkV.Len()){ continue; }
      int64 InWord = WalkV[InWordI]; // input word
	    int64 InWordOriginal = RnmBackH.GetDat(InWord); // word id before normalization
	    int InWordRole = InWordOriginal % roleCount;
      for (int i = 0; i < Dimensions; i++) { Neu1eV[i] = 0; }
      // negative sampling
//	    printf("\nSamples for relation (%d, %d): ", RnmBackH.GetDat(InWord), InWordOriginal);
      for (int n = 0; n < NegSamN + roleCount; n++) {
        int64 Target = 0, Label;
	      if (n == 0) {
	        // The output word as the only positive sample (done inside the negative sampling loop)
          Target = OutWord;
          Label = 1;
//		      printf("%d (positive), ", RnmBackH.GetDat(Target));
				} else if (n < roleCount){
					// If input word is of type 'out', 
					// sample other "in" roles of the original output word (Word)
					// For example, if pair is (2out, 4in-), 2out should be far from 4in+ 
					// For role count = 4, words with ids  3 x w + [1, 2, 3]
			    // are "in" roles of base word w. 
					if(roleNegativeSampling && InWordRole == 0 && 
						OutWordRole != 0 && OutWordRole != n){
						int64 otherRoleId = baseWord * roleCount + n;
						if(!RnmH.IsKey(otherRoleId)){
//							printf("Sample [%d]: other role %d not found for %d[%d]\n", n + 1,
//							otherRoleId,  InWordOriginal, CurrWord);
							continue;
						}
						Target = RnmH.GetDat(otherRoleId);
//						printf("%d (other roles), ", RnmBackH.GetDat(Target));
						Label = 0;
					}else{
						continue;
					}
			  } else {
	        // Sample a negative (output) word w proportional to frequency(w) ^ (3/4)
          Target = RndUnigramInt(KTable, UTable, Rnd);
				  // output word is a positive sample, also do not select the input node itself
          if (Target == OutWord || Target == InWord) { continue; }
          Label = 0;
//					printf("%d (random), ", RnmBackH.GetDat(Target));
        }
	      // Inner product of input (cur) word with nth negative (positive) output sample
	      // input vector is named SynPos, and output vector is named SynNeg
        double Product = 0;
        for (int i = 0; i < Dimensions; i++) {
          Product += SynPos(InWord,i) * SynNeg(Target,i);
        }
        double Grad; //Gradient multiplied by learning rate
        if (Product > MaxExp) { Grad = (Label - 1) * Alpha; }
        else if (Product < -MaxExp) { Grad = Label * Alpha; }
        else {
          double Exp = ExpTable[static_cast<int>(Product*ExpTablePrecision)+TableSize/2];
          Grad = (Label - 1 + 1 / (1 + Exp)) * Alpha;
        }
        for (int i = 0; i < Dimensions; i++) {
          Neu1eV[i] += Grad * SynNeg(Target,i);
          SynNeg(Target,i) += Grad * SynPos(InWord,i);
        }
      }
//	    fflush(stdout);
	    // Add gradient of positive sample and negative samples
	    // to the positive embedding of input word
      for (int i = 0; i < Dimensions; i++) {
        SynPos(InWord,i) += Neu1eV[i];
      }
    }
    WordCntAll++;
  }
}


void LearnEmbeddings(TVVec<TInt, int64>& WalksVV, const int& Dimensions,
  const int& WinSize, const int& roleCount, const bool& roleNegativeSampling,
	const int& Iter, const bool& Verbose,
  TIntFltVH& EmbeddingsHV) {
  TIntIntH RnmH;
  TIntIntH RnmBackH;
  int64 NNodes = 0;
  //renaming nodes into consecutive numbers
  for (int i = 0; i < WalksVV.GetXDim(); i++) {
    for (int64 j = 0; j < WalksVV.GetYDim(); j++) {
      if ( RnmH.IsKey(WalksVV(i, j)) ) {
        WalksVV(i, j) = RnmH.GetDat(WalksVV(i, j));
      } else {
        RnmH.AddDat(WalksVV(i,j),NNodes);
        RnmBackH.AddDat(NNodes,WalksVV(i, j));
        WalksVV(i, j) = NNodes++;
      }
    }
  }
  TIntV Vocab(NNodes);
  LearnVocab(WalksVV, Vocab);
//	for (int v = 0 ; v < Vocab.Len() ; v++){
//		printf("word: %d, count: %d\n", RnmBackH.GetDat(v), Vocab[v]);
//	}
//	fflush(stdout);
  TIntV KTable(NNodes);
  TFltV UTable(NNodes);
  TVVec<TFlt, int64> SynNeg;
  TVVec<TFlt, int64> SynPos;
  TRnd Rnd(time(NULL));
  InitPosEmb(Vocab, Dimensions, Rnd, SynPos);
  InitNegEmb(Vocab, Dimensions, SynNeg);
  InitUnigramTable(Vocab, KTable, UTable);
  TFltV ExpTable(TableSize);
  double Alpha = StartAlpha;                              //learning rate
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < TableSize; i++ ) {
    double Value = -MaxExp + static_cast<double>(i) / static_cast<double>(ExpTablePrecision);
    ExpTable[i] = TMath::Power(TMath::E, Value);
  }
  int64 WordCntAll = 0;
// op RS 2016/09/26, collapse does not compile on Mac OS X
//#pragma omp parallel for schedule(dynamic) collapse(2)
  for (int j = 0; j < Iter; j++) {
 #pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < WalksVV.GetXDim(); i++) {
      TrainModel(WalksVV, RnmH, RnmBackH, Dimensions, WinSize, roleCount,
		   roleNegativeSampling, Iter, Verbose, KTable, UTable,
       WordCntAll, ExpTable, Alpha, i, Rnd, SynNeg, SynPos);
    }
  }
  if (Verbose) { printf("\n"); fflush(stdout); }
  for (int64 i = 0; i < SynPos.GetXDim(); i++) {
    TFltV CurrV(SynPos.GetYDim());
    for (int j = 0; j < SynPos.GetYDim(); j++) { CurrV[j] = SynPos(i, j); }
    EmbeddingsHV.AddDat(RnmBackH.GetDat(i), CurrV);
  }
}
