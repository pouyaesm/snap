#ifndef R2V_H
#define R2V_H

#include "stdafx.h"

#include "Snap.h"
#include "r2v_biasedrandomwalk.h"
#include "r2v_word2vec.h"

/// Calculates role2vec feature representation for nodes and writes them into EmbeddinsHV
void role2vec(PWNet& InNet, const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  const bool& OutputWalks, TVVec<TInt, int64>& WalksVV,
	TIntFltVH& EmbeddingsHV, const double& ParamP, const double& ParamQ,
	const int& roleCount, TFltV& roleWeight,
	const double& stayP, const double& teleportP, const bool& roleNegativeSampling);

#endif //R2V_H
