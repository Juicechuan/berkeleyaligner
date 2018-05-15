package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.WordAligner;
import edu.berkeley.nlp.wordAlignment.symmetric.BipartiteMatchingExtractor;
import edu.berkeley.nlp.wordAlignment.symmetric.MatchingExtractor;
import edu.berkeley.nlp.wordAlignment.symmetric.MatchingWordAligner;
import edu.berkeley.nlp.wordAlignment.symmetric.features.ExternalPosteriorFeatureLayer;

import java.util.Arrays;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 15, 2008
 */
public class ExternalPosteriorsWordAligner extends WordAligner {
	
	ExternalPosteriorFeatureLayer epfl;

	public int multiWayAlignments = 0;
	
	MatchingExtractor matchingExtractor = new BipartiteMatchingExtractor();

	@Opt (required = true)
	public double threshold = 0.0;
	
	public static enum ExternalInferenceType {
	    PRODUCT,
	    SOFT_UNION,
	    HARD_UNION,
	    MATCHING_SOFT_UNION,
	    MATCHING_HARD_UNION,
	    COMPETITIVE_THRESHOLD_SOFT_UNION,
	    COMPETITIVE_THRESHOLD_HARD_UNION,
	}
	 
	@Opt (required = true)
	public ExternalInferenceType externalInferenceType;
	
	public ExternalPosteriorsWordAligner(ExternalPosteriorFeatureLayer epfl) {
		GlobalOptionParser.fillOptions(this);

		this.epfl = epfl;
	}

	private boolean decode(double ef, double fe) {
		switch (externalInferenceType) {
		case SOFT_UNION:
			return (0.5 * (ef + fe) > threshold);
		case PRODUCT:
			return ((ef*fe) > threshold);
		case MATCHING_SOFT_UNION:
			return (0.5 * (ef + fe) > threshold);
		case COMPETITIVE_THRESHOLD_SOFT_UNION:
			return (0.5 * (ef + fe) > threshold);
		case HARD_UNION:
			return ((ef > threshold) || (fe > threshold));
		case MATCHING_HARD_UNION:
			return ((ef > threshold) || (fe > threshold));
		default:
			return ((ef > threshold) || (fe > threshold));
		}
	}
	
	private double decodeVal(double ef, double fe) {
		switch (externalInferenceType) {
		case SOFT_UNION:
			return (0.5 * (ef + fe));
		case PRODUCT:
			return (ef*fe);
		case MATCHING_SOFT_UNION:
			return (0.5 * (ef + fe));
		case COMPETITIVE_THRESHOLD_SOFT_UNION:
			return (0.5 * (ef + fe));
		case HARD_UNION:
			return ((ef > fe) ? ef : fe);
		case MATCHING_HARD_UNION:
			return ((ef > fe) ? ef : fe);
		default:
			return ((ef > fe) ? ef : fe);
		}
	}

	@Override
	public Alignment alignSentencePair(SentencePair sp) {
		switch (externalInferenceType) {
			case SOFT_UNION:
				return alignSentencePairNoCT(sp);
			case PRODUCT:
				return alignSentencePairNoCT(sp);
			case HARD_UNION:
				return alignSentencePairNoCT(sp);
			case MATCHING_SOFT_UNION:
				return alignSentencePairMatchingSoftUnion(sp);
			case MATCHING_HARD_UNION:
				return alignSentencePairMatchingHardUnion(sp);
			case COMPETITIVE_THRESHOLD_SOFT_UNION:
				return alignSentencePairCT(sp);
			default:
				return alignSentencePairCT(sp);
		}
	}

		
	public Alignment alignSentencePairNoCT(SentencePair sp) {
		double[][] fePosteriors = epfl.getFEPosteriors(sp);
		double[][] efPosteriors = epfl.getEFPosteriors(sp);
	
	    Alignment align = new Alignment(sp, false);
	    int m = sp.getForeignWords().size();
	    int n = sp.getEnglishWords().size();
	
	    int[] totalFEAlignmentsPerWord = new int[m];
	    int[] totalEFAlignmentsPerWord = new int[n];
	    
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (decode(fePosteriors[i][j], efPosteriors[j][i])) {
					 align.addAlignment(j, i, true);
			    	 totalFEAlignmentsPerWord[i]++;
			    	 totalEFAlignmentsPerWord[j]++;
				}
			}
		}
		
		for (int i = 0; i < m; ++i) {
			if (totalFEAlignmentsPerWord[i] > 1) multiWayAlignments++;
		} 
		for (int j = 0; j < n; ++j) {
			if (totalEFAlignmentsPerWord[j] > 1) multiWayAlignments++;
		}
		
		return align;
	}

  public double[][]  getPosteriors(SentencePair sp) {

    double[][] fePosteriors = epfl.getFEPosteriors(sp);
    double[][] efPosteriors = epfl.getEFPosteriors(sp);

    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    double[][] posts = new double[m][n];

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        posts[i][j] =  Math.max(fePosteriors[i][j],efPosteriors[j][i]);
      }
    }
    return posts;
  }

  public Alignment alignSentencePairCT(SentencePair sp) {
	    Alignment align = new Alignment(sp, false);
	    int m = sp.getForeignWords().size();
	    int n = sp.getEnglishWords().size();

	    double[][] fePosteriors = epfl.getFEPosteriors(sp);
	    double[][] efPosteriors = epfl.getEFPosteriors(sp);
	    
	    double[][] post = new double[m][n];
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (decode(fePosteriors[i][j], efPosteriors[j][i])) {
					 align.addAlignment(j, i, true);
					 post[i][j] = 0.5*(fePosteriors[i][j] + efPosteriors[j][i]);
				}
			}
		}
				
		// I then J
		for (int i = 0; i < m; i++) {
			int maxIndex = -1;
			double maxValue = -1;

			// Find the maximum
			for (int j = 0; j < n; j++) {
				if (post[i][j] > maxValue) {
					maxValue = post[i][j];
					maxIndex = j;
				}
			}

			if (maxValue >= threshold) {
				// Fill above
				boolean contiguous = true;
				for (int j = maxIndex; j < n; j++) {
					if (contiguous) {
						if (post[i][j] < threshold) {
							contiguous = false;
						}
						if (j > maxIndex) multiWayAlignments++;
					} else {
						align.removeAlignment(j, i);
					}
				}

				// Fill below
				contiguous = true;
				for (int j = maxIndex; j >= 0; j--) {
					if (contiguous) {
						if (post[i][j] < threshold) {
							contiguous = false;
						}
						if (j < maxIndex) multiWayAlignments++;
					} else {
						align.removeAlignment(j, i);
					}
				}
			}
		}

		// J then I
		for (int j = 0; j < n; j++) {
			int maxIndex = -1;
			double maxValue = -1;

			// Find the maximum
			for (int i = 0; i < m; i++) {
				if (post[i][j] > maxValue) {
					maxValue = post[i][j];
					maxIndex = i;
				}
			}

			if (maxValue >= threshold) {
				// Fill below
				boolean contiguous = true;
				for (int i = maxIndex; i < m; i++) {
					if (contiguous) {
						if (post[i][j] < threshold) {
							contiguous = false;
						}
						if (i > maxIndex) multiWayAlignments++;
					} else {
						align.removeAlignment(j, i);
					}
				}

				// Fill above
				contiguous = true;
				for (int i = maxIndex; i >= 0; i--) {
					if (contiguous) {
						if (post[i][j] < threshold) {
							contiguous = false;
						}
						if (i < maxIndex) multiWayAlignments++;
					} else {
						align.removeAlignment(j, i);
					}
				}
			}
		}

		return align;
	}
    
  public Alignment alignSentencePairMatchingSoftUnion(SentencePair sp) {
	    double[][] fePosteriors = epfl.getFEPosteriors(sp);
		double[][] efPosteriors = epfl.getEFPosteriors(sp);
	
	    int m = sp.getForeignWords().size();
	    int n = sp.getEnglishWords().size();
	   
	    
	    double[][] potentials = new double[m + n][m + n];
	    for (double[] row : potentials) {
	      Arrays.fill(row, 0.0);
	    }
	   
	    for (int i = 0; i < m; ++i) {
	      for (int j = 0; j < n; ++j) {
	    	  double softUnion = 0.5*(fePosteriors[i][j] + efPosteriors[j][i]);
	          if (softUnion > threshold) {
	        	  potentials[i][j] = softUnion;
	          } else {
	        	  potentials[i][j] = softUnion-1.0;
	          }   
	      }
	   }
	    
       int[] matching = matchingExtractor.extractMatching(potentials);
	   return MatchingWordAligner.makeAlignment(sp, matching);
  }

  public Alignment alignSentencePairMatchingHardUnion(SentencePair sp) {
	    double[][] fePosteriors = epfl.getFEPosteriors(sp);
		double[][] efPosteriors = epfl.getEFPosteriors(sp);
	
	    int m = sp.getForeignWords().size();
	    int n = sp.getEnglishWords().size();
	   
	    
	    double[][] potentials = new double[m + n][m + n];
	    for (double[] row : potentials) {
	      Arrays.fill(row, 0.0);
	    }
	   
	    for (int i = 0; i < m; ++i) {
	      for (int j = 0; j < n; ++j) {
	    	  double hardUnion = (fePosteriors[i][j] > efPosteriors[j][i]) ? fePosteriors[i][j] : efPosteriors[j][i];
	          if (hardUnion > threshold) {
	        	  potentials[i][j] = hardUnion;
	          } else {
	        	  potentials[i][j] = hardUnion-1.0;
	          }   
	      }
	   }
	    
     int[] matching = matchingExtractor.extractMatching(potentials);
	   return MatchingWordAligner.makeAlignment(sp, matching);
  }

  @Override
  public String getName() {
	return "ExternalPosteriors";
  }

}
