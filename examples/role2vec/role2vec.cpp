#include "stdafx.h"
#include <sstream>
#include "r2v.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Repeat a given string n times (used for array args initial value)
std::string repeat_string(char* string, int n) {
  std::ostringstream os;
  for(int i = 0; i < n; i++)
      os << string;
  return os.str();
}

// Convert string "1.2,3.3" to float vector {1.2, 3.3}
TFltV getFltV(TStr line){
	// Vector of string to save tokens
  TFltV values;

	// Returns first token
  char *token = strtok(line.GetCStr(), ",");

	// Keep printing tokens while one of the
	// delimiters present in str[].
	while(token != NULL)
	{
		values.Add(std::stof(token));
		token = strtok(NULL, ",");
	}
	return values;
}

void ParseArgs(int& argc, char* argv[], TStr& InFile, TStr& OutFile,
 int& Dimensions, int& WalkLen, int& NumWalks, int& WinSize, int& Iter,
 bool& Verbose, bool& Weighted,
	bool& OutputWalks, int& roleCount, TFltV& roleWeight, double& stayP, double& teleportP) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nAn algorithmic framework for representational learning on graphs."));
  InFile = Env.GetIfArgPrefixStr("-i:", "graph/karate.edgelist",
   "Input graph path");
  OutFile = Env.GetIfArgPrefixStr("-o:", "emb/karate.emb",
   "Output graph path");
  Dimensions = Env.GetIfArgPrefixInt("-d:", 128,
   "Number of dimensions. Default is 128");
  WalkLen = Env.GetIfArgPrefixInt("-l:", 80,
   "Length of walk per source. Default is 80");
  NumWalks = Env.GetIfArgPrefixInt("-r:", 10,
   "Number of walks per source. Default is 10");
  WinSize = Env.GetIfArgPrefixInt("-k:", 10,
   "Context size for optimization. Default is 10");
  Iter = Env.GetIfArgPrefixInt("-e:", 1,
   "Number of epochs in SGD. Default is 1");
	roleCount = Env.GetIfArgPrefixInt("-rc:", 1,
	 "Number of node roles in n-partite graph, role 'out' is assumed to have id = tc * i. Default is 1.");
	TStr initRoleWeight(repeat_string("1,", roleCount - 1).c_str());
	roleWeight = getFltV(Env.GetIfArgPrefixStr("-rw:", initRoleWeight,
	 "Weight of each role (role 'out' is excluded) to be chosen by random walker, "
	 "nth weight is for nodes with role 'n' having id = tc * i + n. Default is '1,1,..'."));
	stayP = Env.GetIfArgPrefixFlt("-sp:", 0.75,
	  "Probability of staying in the same node role. Default is 0.75.");
  teleportP = Env.GetIfArgPrefixFlt("-tp:", 0,
	  "Probability of teleporting from an 'in' node to its corresponding 'out' node. Default is 0.");
  Verbose = Env.IsArgStr("-v", "Verbose output.");
  // Directed = Env.IsArgStr("-dr", "Graph is directed.");
  Weighted = Env.IsArgStr("-w", "Graph is weighted.");
  OutputWalks = Env.IsArgStr("-ow", "Output random walks instead of embeddings.");
}

void ReadGraph(TStr& InFile, bool& Weighted, bool& Verbose, PWNet& InNet) {
  TFIn FIn(InFile);
  int64 LineCnt = 0;
  try {
    while (!FIn.Eof()) {
      TStr Ln;
      FIn.GetNextLn(Ln);
      TStr Line, Comment;
      Ln.SplitOnCh(Line,'#',Comment);
      TStrV Tokens;
      Line.SplitOnWs(Tokens);
      if(Tokens.Len()<2){ continue; }
      int64 SrcNId = Tokens[0].GetInt();
      int64 DstNId = Tokens[1].GetInt();
      double Weight = 1.0;
      if (Weighted) { Weight = Tokens[2].GetFlt(); }
      if (!InNet->IsNode(SrcNId)){ InNet->AddNode(SrcNId); }
      if (!InNet->IsNode(DstNId)){ InNet->AddNode(DstNId); }
      InNet->AddEdge(SrcNId,DstNId,Weight);
      InNet->AddEdge(DstNId,SrcNId,Weight);  // input graph is supposed to be not directed
      LineCnt++;
    }
    if (Verbose) { printf("Read %lld lines from %s\n", (long long)LineCnt, InFile.CStr()); }
  } catch (PExcept Except) {
    if (Verbose) {
      printf("Read %lld lines from %s, then %s\n", (long long)LineCnt, InFile.CStr(),
       Except->GetStr().CStr());
    }
  }
}

void WriteOutput(TStr& OutFile, TIntFltVH& EmbeddingsHV, TVVec<TInt, int64>& WalksVV,
 bool& OutputWalks) {
  TFOut FOut(OutFile);
  if (OutputWalks) {
    for (int64 i = 0; i < WalksVV.GetXDim(); i++) {
      for (int64 j = 0; j < WalksVV.GetYDim(); j++) {
        FOut.PutInt(WalksVV(i,j));
	if(j+1==WalksVV.GetYDim()) {
          FOut.PutLn();
	} else {
          FOut.PutCh(' ');
	}
      }
    }
    return;
  }
  bool First = 1;
  for (int i = EmbeddingsHV.FFirstKeyId(); EmbeddingsHV.FNextKeyId(i);) {
    if (First) {
      FOut.PutInt(EmbeddingsHV.Len());
      FOut.PutCh(' ');
      FOut.PutInt(EmbeddingsHV[i].Len());
      FOut.PutLn();
      First = 0;
    }
    FOut.PutInt(EmbeddingsHV.GetKey(i));
    for (int64 j = 0; j < EmbeddingsHV[i].Len(); j++) {
      FOut.PutCh(' ');
      FOut.PutFlt(EmbeddingsHV[i][j]);
    }
    FOut.PutLn();
  }
}

int main(int argc, char* argv[]) {
  TStr InFile,OutFile;
	int Dimensions, WalkLen, NumWalks, WinSize, Iter, roleCount;
	TFltV roleWeight;
	double stayP;
	double teleportP;
  bool Weighted, Verbose, OutputWalks;
  ParseArgs(argc, argv, InFile, OutFile, Dimensions, WalkLen, NumWalks, WinSize,
		Iter, Verbose, Weighted, OutputWalks, roleCount, roleWeight, stayP, teleportP);
  PWNet InNet = PWNet::New();
  TIntFltVH EmbeddingsHV;
  TVVec <TInt, int64> WalksVV;
  ReadGraph(InFile, Weighted, Verbose, InNet);
  role2vec(InNet, Dimensions, WalkLen, NumWalks, WinSize, Iter,
		Verbose, OutputWalks, WalksVV, EmbeddingsHV, roleCount, roleWeight, stayP, teleportP);
  WriteOutput(OutFile, EmbeddingsHV, WalksVV, OutputWalks);
  return 0;
}
