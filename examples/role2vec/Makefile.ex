MAIN = role2vec
DEPH = $(EXSNAPADV)/r2v.h $(EXSNAPADV)/r2v_word2vec.h $(EXSNAPADV)/r2v_biasedrandomwalk.h
DEPCPP = $(EXSNAPADV)/r2v.cpp $(EXSNAPADV)/r2v_word2vec.cpp $(EXSNAPADV)/r2v_biasedrandomwalk.cpp
CXXFLAGS += $(CXXOPENMP)
CXXFLAGS += -ggdb
