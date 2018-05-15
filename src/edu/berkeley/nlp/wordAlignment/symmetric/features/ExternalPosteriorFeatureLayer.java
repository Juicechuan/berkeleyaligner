package edu.berkeley.nlp.wordAlignment.symmetric.features;

import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.FeatureLayer;

import java.util.List;

public class ExternalPosteriorFeatureLayer extends FeatureLayer {

   @Opt(required = true)
   public int itgBlockSize = 0;

   @Opt(required = true)
   public String posteriorKeyPrefix;
  
  public ExternalPosteriorFeatureLayer(String efKey, String feKey) {
    GlobalOptionParser.fillOptions(this);
  }


  // EF posteriors must be transposed to get [E,F]
  public double[][] getEFPosteriors(SentencePair sp) {
    return sp.getKeyedAlignment(posteriorKeyPrefix+".ef").getEnglishByForeignPosteriors();
  }

  // FE posteriors are in correct order [F,E]
  public double[][] getFEPosteriors(SentencePair sp) {
    return sp.getKeyedAlignment(posteriorKeyPrefix+".fe").getForeignByEnglishPosteriors();
  }

  public double minProd(SentencePair sp, int i, int j, int size, boolean frBlock) {
    double[][] efPosteriors = getEFPosteriors(sp);
	double[][] fePosteriors = getFEPosteriors(sp);
	  
	int frExt = frBlock ? 1 : 0;
	int enExt = !frBlock ? 1 : 0;
	double min = Double.POSITIVE_INFINITY;
	for (int k = 0; k < size; k++) {
	   int fIndex = i + frExt * k;
	   int eIndex = j + enExt * k;
 	   
	   double combined = efPosteriors[eIndex][fIndex]*fePosteriors[fIndex][eIndex];
	   min = Math.min(combined, min);
	}
	return min;
  }	   

  public double minEGivenF(SentencePair sp, int i, int j, int size, boolean frBlock) {
      double[][] efPosteriors = getEFPosteriors(sp);

      int frExt = frBlock ? 1 : 0;
	  int enExt = !frBlock ? 1 : 0;
	  double min = Double.POSITIVE_INFINITY;
	  for (int k = 0; k < size; k++) {
	      int fIndex = i + frExt * k;
	      int eIndex = j + enExt * k;

	      min = Math.min(efPosteriors[eIndex][fIndex], min);
	  }
	  return min;
  }	   

	  
  public double minFGivenE(SentencePair sp, int i, int j, int size, boolean frBlock) {
	double[][] fePosteriors = getFEPosteriors(sp);
	
	int frExt = frBlock ? 1 : 0;
    int enExt = !frBlock ? 1 : 0;
    double min = Double.POSITIVE_INFINITY;
    for (int k = 0; k < size; k++) {
      int fIndex = i + frExt * k;
      int eIndex = j + enExt * k;
    
      min = Math.min(fePosteriors[fIndex][eIndex], min);
    }
    return min;
  }

  public double avgProd(SentencePair sp, int i, int j, int size, boolean frBlock) {
    double[][] efPosteriors = getEFPosteriors(sp);
	double[][] fePosteriors = getFEPosteriors(sp);
	  
	int frExt = frBlock ? 1 : 0;
	int enExt = !frBlock ? 1 : 0;
	double sum = 0.0;
	for (int k = 0; k < size; k++) {
	   int fIndex = i + frExt * k;
	   int eIndex = j + enExt * k;
 	   
	   double combined = efPosteriors[eIndex][fIndex]*fePosteriors[fIndex][eIndex];
	   sum += combined;
	}
	return sum / ((double)size);
  }	   

  public double avgEGivenF(SentencePair sp, int i, int j, int size, boolean frBlock) {
      double[][] efPosteriors = getEFPosteriors(sp);

      int frExt = frBlock ? 1 : 0;
	  int enExt = !frBlock ? 1 : 0;
	  double sum = 0.0;
	  for (int k = 0; k < size; k++) {
	      int fIndex = i + frExt * k;
	      int eIndex = j + enExt * k;

	      sum += efPosteriors[eIndex][fIndex];
	  }
	  return sum / ((double)size);
  }	   

	  
  public double avgFGivenE(SentencePair sp, int i, int j, int size, boolean frBlock) {
	double[][] fePosteriors = getFEPosteriors(sp);
	
	int frExt = frBlock ? 1 : 0;
    int enExt = !frBlock ? 1 : 0;
    double sum = 0.0;
    for (int k = 0; k < size; k++) {
      int fIndex = i + frExt * k;
      int eIndex = j + enExt * k;
    
      sum += fePosteriors[fIndex][eIndex];
    }
    return sum / ((double)size);
  }
  
  @Override
  public void addEnBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] enBlockFeats,
                                 List<FeatureValuePair>[][] alignmentFeats,
                                 List<FeatureValuePair>[] frNullFeats,
                                 List<FeatureValuePair>[] enNullFeats) {

    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    //Utils.AlignmentDecomposition ad = Utils.decomposeAlignment(sp.getAlignment(),itgBlockSize);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 2; k <= itgBlockSize && j + k <= n; k++) {
            enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-bias-" + k, 1.0));

 			enBlockFeats[i][j][k].add(new FeatureValuePair("efPosterior", avgEGivenF(sp, i, j, k, false)*k));
 			enBlockFeats[i][j][k].add(new FeatureValuePair("fePosterior", avgFGivenE(sp, i, j, k, false)*k));
            enBlockFeats[i][j][k].add(new FeatureValuePair("combinedPosterior", avgProd(sp, i, j, k, false)*k));
        }
      }
    }
    
  }

  @Override
  public void addFrBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] frBlockFeats,
                                 List<FeatureValuePair>[][] alignmentFeats,
                                 List<FeatureValuePair>[] frNullFeats,
                                 List<FeatureValuePair>[] enNullFeats) {
	
	int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    //Utils.AlignmentDecomposition ad = Utils.decomposeAlignment(sp.getAlignment(),itgBlockSize);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 2; k <= itgBlockSize && i + k <= m; k++) {
            frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-bias-" + k, 1.0));
          
 			frBlockFeats[i][j][k].add(new FeatureValuePair("efPosterior", avgEGivenF(sp, i, j, k, true)*k));
 			frBlockFeats[i][j][k].add(new FeatureValuePair("fePosterior", avgFGivenE(sp, i, j, k, true)*k));
            frBlockFeats[i][j][k].add(new FeatureValuePair("combinedPosterior", avgProd(sp, i, j, k, true)*k));
        }
      }
    }
  }

  @Override
  public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
	try {
	  double[][] efPosteriors = getEFPosteriors(sp);
	  double[][] fePosteriors = getFEPosteriors(sp);
	
	
	int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
    	  double combined = efPosteriors[j][i]*fePosteriors[i][j]; 

 			alignFeats[i][j].add(new FeatureValuePair("efPosterior", efPosteriors[j][i]));
 			alignFeats[i][j].add(new FeatureValuePair("fePosterior", fePosteriors[i][j]));
   			alignFeats[i][j].add(new FeatureValuePair("combinedPosterior", combined));
      }
    }
	} catch (Exception e) {
		e.printStackTrace();
		System.exit(0);
	}
  }

//  @Override
//  public void addFrNullFeatures(SentencePair sp, List<FeatureValuePair>[] frNullFeats) {
//    int m = sp.getForeignWords().size();
//    int n = sp.getEnglishWords().size();
//
//    double[][] efPosteriors = getEFPosteriors(sp);
//	double[][] fePosteriors = getFEPosteriors(sp);
//
//    if (externalPosteriorBinsFeatureLayer == null) {
//    	for (int i = 0; i < m; ++i) {
//    		double prod = 1.0;
//    		for (int j = 0; j < n; ++j) {
//    			double softUnion = 0.5*(efPosteriors[j][i] + fePosteriors[i][j]);
//    			prod *= (1.0 - softUnion);
//    		}
//
////    		frNullFeats[i].add(new FeatureValuePair("external_posterior_frnull", prod));
//    	}
//    }
//  }

//  @Override
//  public void addEnNullFeatures(SentencePair sp, List<FeatureValuePair>[] enNullFeats) {
//	  int m = sp.getForeignWords().size();
//	  int n = sp.getEnglishWords().size();
//
//	  double[][] efPosteriors = getEFPosteriors(sp);
//	  double[][] fePosteriors = getFEPosteriors(sp);
//
//	  if (externalPosteriorBinsFeatureLayer == null) {
//		  for (int j = 0; j < n; ++j) {
//	    	double prod = 1.0;
//	    	for (int i = 0; i < m; ++i) {
//	    		double softUnion = 0.5*(efPosteriors[j][i] + fePosteriors[i][j]);
//	    		prod *= (1.0 - softUnion);
//	    	}
//
////	    	enNullFeats[j].add(new FeatureValuePair("external_posterior_ennull", prod));
//	     }
//	  }
//  }
  
}
