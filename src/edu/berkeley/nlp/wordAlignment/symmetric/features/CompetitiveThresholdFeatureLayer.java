package edu.berkeley.nlp.wordAlignment.symmetric.features;

import edu.berkeley.nlp.fig.basic.NumUtils;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.FeatureLayer;

import java.util.*;

public class CompetitiveThresholdFeatureLayer extends FeatureLayer {

   @Opt(required = true)
   public String posteriorKeyPrefix;

  public CompetitiveThresholdFeatureLayer() {
	GlobalOptionParser.fillOptions(this);
  }

  // EF posteriors must be transposed to get [E,F]
  public double[][] getEFPosteriors(SentencePair sp) {
    double[][] posts = sp.getKeyedAlignment(posteriorKeyPrefix+".ef").getEnglishByForeignPosteriors();
    return NumUtils.transpose(posts);
  }

  // FE posteriors are in correct order [F,E]
  public double[][] getFEPosteriors(SentencePair sp) {
    double[][] posts = sp.getKeyedAlignment(posteriorKeyPrefix+".fe").getForeignByEnglishPosteriors();
    return posts;
  }

  public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
	  double[][] fePosteriors = getFEPosteriors(sp);
	  double[][] efPosteriors = getEFPosteriors(sp);
	
	  int m = sp.getForeignWords().size();
	  int n = sp.getEnglishWords().size();
	 
	  boolean[][] bestJForI = new boolean[m][n];
	  for (int i = 0; i < m; ++i) {
	  	int bestJ = -1;
	  	double bestVal = 0.0;
	  	for (int j = 0; j < n; ++j) {
	  	  double softUnion = 0.5*(fePosteriors[i][j] + efPosteriors[j][i]);
	  	  if (softUnion > bestVal) {
	      	  bestVal = softUnion;
	      	  bestJ = j;
	  	  }
	  	  if (bestVal > 0.5) {
	  		  bestJForI[i][bestJ] = true;
	  	  }
	    }
	 }
	 
	 boolean[][] bestIForJ = new boolean[m][n];
	 for (int j = 0; j < n; ++j) {
		   int bestI = -1;
	     double bestVal = 0.0;
	  	for (int i = 0; i < m; ++i) {
	  	  double softUnion = 0.5*(fePosteriors[i][j] + efPosteriors[j][i]);
	  	  if (softUnion > bestVal) {
	      	  bestVal = softUnion;
	      	  bestI = i;
	  	  }
	  	  if (bestVal > 0.5) {
	  		  bestIForJ[bestI][j] = true;
	  	  }
	    }
	 }
	 
	 for (int i = 0; i < m; ++i) {
	   for (int j = 0; j < n; ++j) {
		   if (bestIForJ[i][j] && bestJForI[i][j]) {
		    	alignFeats[i][j].add(new FeatureValuePair("CT_on", 1.0));
		   } else {
			   alignFeats[i][j].add(new FeatureValuePair("CT_off", 1.0));
		   }
	   }
	 }
  }

  @Override
  public void addFrNullFeatures(SentencePair sp, List<FeatureValuePair>[] frNullFeats) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    
    double[][] efPosteriors = getEFPosteriors(sp);
	double[][] fePosteriors = getFEPosteriors(sp);

   	for (int i = 0; i < m; ++i) {
   		boolean existSoftUnion = false;
    	for (int j = 0; j < n; ++j) {
    		double softUnion = 0.5*(efPosteriors[j][i] + fePosteriors[i][j]);
    		existSoftUnion = existSoftUnion || (softUnion > 0.5);
    	}
    		
//   		frNullFeats[i].add(new FeatureValuePair("no_soft_union_FR", 1.0));
    }
  }

  @Override
  public void addEnNullFeatures(SentencePair sp, List<FeatureValuePair>[] enNullFeats) {
	  int m = sp.getForeignWords().size();
	  int n = sp.getEnglishWords().size();
	    
	  double[][] efPosteriors = getEFPosteriors(sp);
	  double[][] fePosteriors = getFEPosteriors(sp);
	  
	   	for (int j = 0; j < n; ++j) {
	   		boolean existSoftUnion = false;
	    	for (int i = 0; i < m; ++i) {
	    		double softUnion = 0.5*(efPosteriors[j][i] + fePosteriors[i][j]);
	    		existSoftUnion = existSoftUnion || (softUnion > 0.5);
	    	}
//	   		enNullFeats[j].add(new FeatureValuePair("no_soft_union_EN", 1.0));
	    }
  }
}
