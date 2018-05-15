package edu.berkeley.nlp.wordAlignment.symmetric.features;

import edu.berkeley.nlp.fig.basic.NumUtils;
import edu.berkeley.nlp.util.Logger;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.FeatureLayer;

import java.util.HashMap;
import java.util.List;

public class CompetitivePosteriorFeatureLayer extends FeatureLayer {

   @Opt(required = true)
   public double threshold;
   
   @Opt(required = true)
   public double blockThreshold;
   
   @Opt(required = true)
   public int itgBlockSize = 0;
   
   @Opt(required = true)
   public String posteriorKeyPrefix;

   public static enum ExternalInferenceType {
	   SOFT_UNION,
	   HARD_UNION,
	   SOFT_INTERSECTION
   }
	 
	@Opt (required = true)
	public ExternalInferenceType externalInferenceType;
   
  public CompetitivePosteriorFeatureLayer() {
	GlobalOptionParser.fillOptions(this);
  }

  // EF posteriors must be transposed to get [E,F]
  public double[][] getEFPosteriors(SentencePair sp) {
    double[][] posts = sp.getKeyedAlignment(posteriorKeyPrefix+".ef").getEnglishByForeignPosteriors();
    return posts;
  }

  // FE posteriors are in correct order [F,E]
  public double[][] getFEPosteriors(SentencePair sp) {
    double[][] posts = sp.getKeyedAlignment(posteriorKeyPrefix+".fe").getForeignByEnglishPosteriors();
    return posts;
  }

    private boolean decode(double ef, double fe) {
		switch (externalInferenceType) {
		case SOFT_UNION:
			return (0.5 * (ef + fe) > threshold);
		case SOFT_INTERSECTION:
			return (ef*fe > threshold);
		case HARD_UNION:
			return ((ef > threshold) || (fe > threshold));
		default:
			Logger.logs("Invalid external inference type");
			return false;
		}
	}
	
	private double decodeVal(double ef, double fe) {
		switch (externalInferenceType) {
		case SOFT_UNION:
			return (0.5 * (ef + fe));
		case SOFT_INTERSECTION:
			return (ef*fe);
		case HARD_UNION:
			return ((ef > fe) ? ef : fe);
		default:
			Logger.logs("Invalid external inference type");
			return ((ef > fe) ? ef : fe);
		}
	}

  @Override
  public void addEnBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] enBlockFeats,
                                 List<FeatureValuePair>[][] alignmentFeats,
                                 List<FeatureValuePair>[] frNullFeats,
                                 List<FeatureValuePair>[] enNullFeats) {
	
	double[][] efPosteriors = getEFPosteriors(sp);
	double[][] fePosteriors = getFEPosteriors(sp);

    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();

    // Compute best foreign items for each English item
    int[] maxIVals = new int[n];
    for (int j = 0; j < n; ++j) {
    	  double maxDV = 0.0;
    	  int maxI = -1;
    	  for (int i = 0; i < m; ++i) {
    		  double dv = decodeVal(efPosteriors[j][i], fePosteriors[i][j]); 
    		  if (dv > maxDV) {
    			  maxI = i;
    			  maxDV = dv;
    		  }
    	  }
    	  maxIVals[j] = (maxDV > blockThreshold) ? maxI : -1;
    }

    // Find the percent of foreign block items which are a best match to the English
    for (int i = 0; i < m; ++i) {
    	for (int j = 0; j < n; ++j) {
    		for (int k = 2; k <= itgBlockSize && j + k <= n; k++) {
    			double sum = 0.0;
    			for (int ell = j; ell < j+k; ++ell) {
    				if (maxIVals[ell] == i) {
    					sum++;
    				}
    			}
//    	        enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-maxI%", sum/((double)k)));
    	        enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-maxI", sum));
    		}
    	}
    }   	  
  
  }

  @Override
  public void addFrBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] frBlockFeats,
                                 List<FeatureValuePair>[][] alignmentFeats,
                                 List<FeatureValuePair>[] frNullFeats,
                                 List<FeatureValuePair>[] enNullFeats) {
	    
	    double[][] efPosteriors = getEFPosteriors(sp);
		double[][] fePosteriors = getFEPosteriors(sp);
		
		int m = sp.getForeignWords().size();
	    int n = sp.getEnglishWords().size();
	    
	    // Compute best English items for each foreign item
	    int[] maxJVals = new int[m];
	    for (int i = 0; i < m; ++i) {
	    	  double maxDV = 0.0;
	    	  int maxJ = -1;
	    	  for (int j = 0; j < n; ++j) {
	    		  double dv = decodeVal(efPosteriors[j][i], fePosteriors[i][j]); 
	    		  if (dv > maxDV) {
	    			  maxJ = j;
	    			  maxDV = dv;
	    		  }
	    	  }
	    	  maxJVals[i] = (maxDV > blockThreshold) ? maxJ : -1;
	    }

	    // Find the percent of foreign block items which are a best match to the English
	    for (int i = 0; i < m; ++i) {
	    	for (int j = 0; j < n; ++j) {
	    		for (int k = 2; k <= itgBlockSize && i + k <= m; k++) {
	    			double sum = 0.0;
	    			for (int ell = i; ell < i+k; ++ell) {
	    				if (maxJVals[ell] == j) {
	    					sum++;
	    				}
	    			}
//	    	        frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-maxJ%", sum/((double)k)));
	    	        frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-maxJ", sum));
	    		}
	    	}
	    }   	  
   }
   
  
  @Override
  public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
	double[][] efPosteriors = getEFPosteriors(sp);
	double[][] fePosteriors = getFEPosteriors(sp);
	
	int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    for (int i = 0; i < m; ++i) {
      double maxDV = 0.0;
      int maxJ = -1;
      for (int j = 0; j < n; ++j) {
    	  double dv = decodeVal(efPosteriors[j][i], fePosteriors[i][j]); 
    	  if (dv > maxDV) {
    		  maxJ = j;
    		  maxDV = dv;
    	  }
      }
   	  if (maxDV > threshold)
   		  alignFeats[i][maxJ].add(new FeatureValuePair("max_J_for_I", 1.0));
    }   	  
   	
    for (int j = 0; j < n; ++j) {
	    double maxDV = 0.0;
	    int maxI = -1;
	    for (int i = 0; i < m; ++i) {
	  	  double dv = decodeVal(efPosteriors[j][i],fePosteriors[i][j]); 
	  	  if (dv > maxDV) {
	  		  maxI = i;
	  		  maxDV = dv;
	  	  }
	    }
	    if (maxDV > threshold)
	    	alignFeats[maxI][j].add(new FeatureValuePair("max_I_for_J", 1.0));
    }
  }

  @Override
  public void addFrNullFeatures(SentencePair sp, List<FeatureValuePair>[] frNullFeats) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    
    double[][] efPosteriors = getEFPosteriors(sp);
	double[][] fePosteriors = getFEPosteriors(sp);

   	for (int i = 0; i < m; ++i) {
   		boolean existOn = false;
    	for (int j = 0; j < n; ++j) {
    		existOn = existOn || decode(efPosteriors[j][i],fePosteriors[i][j]);
    	}
    	if (!existOn)
    		frNullFeats[i].add(new FeatureValuePair("no_soft_union_FR", 1.0));
    }
  }

  @Override
  public void addEnNullFeatures(SentencePair sp, List<FeatureValuePair>[] enNullFeats) {
	  int m = sp.getForeignWords().size();
	  int n = sp.getEnglishWords().size();
	    
	  double[][] efPosteriors = getEFPosteriors(sp);
	  double[][] fePosteriors = getFEPosteriors(sp);
	  
	   	for (int j = 0; j < n; ++j) {
	   		boolean existOn = false;
	    	for (int i = 0; i < m; ++i) {
	    		existOn = existOn || decode(efPosteriors[j][i],fePosteriors[i][j]);
	    	}
	    	if (!existOn)
	    		enNullFeats[j].add(new FeatureValuePair("no_soft_union_EN", 1.0));
	    }
  }
}
