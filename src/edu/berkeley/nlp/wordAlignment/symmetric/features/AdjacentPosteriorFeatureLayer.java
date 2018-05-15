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

public class AdjacentPosteriorFeatureLayer extends FeatureLayer {

   @Opt(required = true)
   public double threshold;

   @Opt(required = true)
   public int itgBlockSize = 0;

   public static enum ExternalInferenceType {
	   SOFT_UNION,
	   SOFT_INTERSECTION,
	   HARD_UNION,
   }
	
   @Opt(required = true)
   public String posteriorKeyPrefix;

   @Opt (required = true)
   public ExternalInferenceType externalInferenceType;
   
  public AdjacentPosteriorFeatureLayer() {
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
    
    // Find the percent of foreign block items which are a best match to the English
    for (int i = 0; i < m; ++i) {
    	for (int j = 0; j < n; ++j) {
    		for (int k = 2; k <= itgBlockSize && j + k <= n; k++) {
    			if ( (i > 0) && (j > 0) ) {  
    				double dv = decodeVal(efPosteriors[j-1][i-1],fePosteriors[i-1][j-1]); 
    				if (dv > threshold) {
    	        	  enBlockFeats[i][j][k].add(new FeatureValuePair("LL_ABOVE", 1.0));
    				}
    			}
    		}
    	}
    	
    	for (int j = 0; j < n-1; ++j) {
    		for (int k = 2; k <= itgBlockSize && j + k <= n; k++) {
    			if ( (i > 0) && (j+k < n) ) {  
    				double dv = decodeVal(efPosteriors[j+k][i-1],fePosteriors[i-1][j+k]); 
    				if (dv > threshold) {
    	        	  enBlockFeats[i][j][k].add(new FeatureValuePair("LR_ABOVE", 1.0));
    				}
    			}
    		}
        }
    	
    	for (int j = 1; j < n; ++j) {
    		for (int k = 2; k <= itgBlockSize && j + k <= n; k++) {
    			if ( (i < m-1) && (j > 0) ) {  
    				double dv = decodeVal(efPosteriors[j-1][i+1],fePosteriors[i+1][j-1]); 
    				if (dv > threshold) {
    	        	  enBlockFeats[i][j][k].add(new FeatureValuePair("RL_ABOVE", 1.0));
    				}
    			}
    		}
    	}
    	
    	for (int j = 0; j < n-1; ++j) {
    		for (int k = 2; k <= itgBlockSize && j + k <= n; k++) {
    			if ( (i < m-1) && (j+k < n) ) {  
    				double dv = decodeVal(efPosteriors[j+k][i+1],fePosteriors[i+1][j+k]); 
    				if (dv > threshold) {
    					enBlockFeats[i][j][k].add(new FeatureValuePair("RR_ABOVE", 1.0));
    				}
    			}
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
    
	    // Find the percent of foreign block items which are a best match to the English
	    for (int i = 0; i < m; ++i) {
	    	for (int j = 0; j < n; ++j) {
	    		for (int k = 2; k <= itgBlockSize && i + k <= m; k++) {
	    			if ( (i > 0) && (j > 0) ) {  
	    				double dv = decodeVal(efPosteriors[j-1][i-1],fePosteriors[i-1][j-1]); 
	    				if (dv > threshold) {
	      	        	  frBlockFeats[i][j][k].add(new FeatureValuePair("LL_ABOVE", 1.0));
	    				}
	    			}
	    		}
	    	}
	    	
	    	for (int j = 0; j < n-1; ++j) {
	    		for (int k = 2; k <= itgBlockSize && i + k <= m; k++) {
	    			if ( (i > 0) && (j < n-1) ) {  
	    				double dv = decodeVal(efPosteriors[j+1][i-1],fePosteriors[i-1][j+1]); 
	    				if (dv > threshold) {
	    	        	  frBlockFeats[i][j][k].add(new FeatureValuePair("LR_ABOVE", 1.0));
	    				}
	    			}
	    		}
	        }
	    	
	    	for (int j = 1; j < n; ++j) {
	    		for (int k = 2; k <= itgBlockSize && i + k <= m; k++) {
	    			if ( (i+k < m) && (j > 0) ) {  
	    				double dv = decodeVal(efPosteriors[j-1][i+k],fePosteriors[i+k][j-1]); 
	    				if (dv > threshold) {
	    	        	  frBlockFeats[i][j][k].add(new FeatureValuePair("RL_ABOVE", 1.0));
	    				}
	    			}
	    		}
	    	}
	    	
	    	for (int j = 0; j < n-1; ++j) {
	    		for (int k = 2; k <= itgBlockSize && i + k <= m; k++) {
	    			if ( (i+k < m) && (j < n-1) ) {  
	    				double dv = decodeVal(efPosteriors[j+1][i+k],fePosteriors[i+k][j+1]); 
	    				if (dv > threshold) {
	    					frBlockFeats[i][j][k].add(new FeatureValuePair("RR_ABOVE", 1.0));
	    				}
	    			}
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
    for (int i = 1; i < m; ++i) {
      for (int j = 1; j < n; ++j) {
    	  double dv = decodeVal(efPosteriors[j-1][i-1],fePosteriors[i-1][j-1]); 
          if (dv > threshold) {
        	  alignFeats[i][j].add(new FeatureValuePair("LL_ABOVE", 1.0));
          }
      }
      for (int j = 0; j < n-1; ++j) {
    	  double dv = decodeVal(efPosteriors[j+1][i-1],fePosteriors[i-1][j+1]); 
          if (dv > threshold) {
        	  alignFeats[i][j].add(new FeatureValuePair("LR_ABOVE", 1.0));
          }
      }
    }
    for (int i = 0; i < m-1; ++i) {
    	for (int j = 1; j < n; ++j) {
    		double dv = decodeVal(efPosteriors[j-1][i+1],fePosteriors[i+1][j-1]); 
    		if (dv > threshold) {
    			alignFeats[i][j].add(new FeatureValuePair("RL_ABOVE", 1.0));
    		}
    	}
    	for (int j = 0; j < n-1; ++j) {
    		double dv = decodeVal(efPosteriors[j+1][i+1],fePosteriors[i+1][j+1]); 
    		if (dv > threshold) {
    			alignFeats[i][j].add(new FeatureValuePair("RR_ABOVE", 1.0));
    		}
    	}
    }
  }
}
