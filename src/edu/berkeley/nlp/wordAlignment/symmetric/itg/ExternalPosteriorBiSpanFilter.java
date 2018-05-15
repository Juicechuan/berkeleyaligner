package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.SentencePair;

import java.util.ArrayList;
import java.util.List;

/**
 * User: aria42
 * Date: Feb 17, 2009
 */
public class ExternalPosteriorBiSpanFilter implements BiSpanFilter {

  ExternalPosteriorsWordAligner wordAligner;
  double thresh = 0.9;

  double singleCellRecallThreshold = 1.0e-4;
    
  double prunedPrecisionCells = 0;
  double totalPrecisionCells = 0;

  @Opt(required = true)
  public int allowedPrecisionConflicts = -1;

  public ExternalPosteriorBiSpanFilter(ExternalPosteriorsWordAligner wordAligner) {
    GlobalOptionParser.fillOptions(this);
	this.wordAligner = wordAligner;
  }

  public static boolean[][][][] createFilter(SentencePair sp) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    boolean[][][][] filter = new boolean[m+1][][][];
    for (int i = 0; i <= m; i++) {
      filter[i] = new boolean[m + 1][][];
      for (int j = i; j <= m; j++) {
        filter[i][j] = new boolean[n + 1][];
        for (int s = 0; s <= n; s++) {
          filter[i][j][s] = new boolean[n + 1];
        }
      }
    }
    return filter;
  }

  @Override
public boolean[][][][] getBiSpanFilter(SentencePair sp) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    boolean[][][][] filter = createFilter(sp);
    double[][]  posts = wordAligner.getPosteriors(sp);
    List<Pair<Integer,Integer>> sureAligns = new ArrayList<Pair<Integer,Integer>>();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (posts[i][j] > thresh) sureAligns.add(Pair.newPair(i,j));
      }
    }
//    Logger.logss("Min sentence Length: %d",Math.min(m,n));

    double prunedSingleCells = 0;
    double totalSingleCells = 0;
       
    for (int i = 0; i < m; i++) {
    	for (int s = 0; s < n; s++) {
    		if (posts[i][s] < singleCellRecallThreshold) {
        		filter[i][i+1][s][s+1] = true;
        		prunedSingleCells++;
        	}
        	totalSingleCells++;
    	}
    }
    //Logger.logss("Pruned %2.1f / %2.1f single cells: %.3f",prunedSingleCells,totalSingleCells,prunedSingleCells/totalSingleCells);
    
    for (int i = 0; i <= m; i++) {
      for (int j = i; j <= m; j++) {
        for (int s = 0; s <= n; s++) {
          for (int t = s; t <= n; t++) {
            int minLen = Math.min(j-i,t-s);
            int maxLen = Math.max(j-i,t-s);
            if (minLen == 1 && maxLen > 5) {
              filter[i][j][s][t] = true;
              continue;
            }
            if (j-i < 2 || t-s < 2) continue;
            int numConflics = 0;
            for (Pair<Integer, Integer> sureAlign : sureAligns) {
              int x = sureAlign.getFirst();
              int y = sureAlign.getSecond();
              // Above, Left, Right, Bottom
              boolean conflict =  (x >= i && x < j && y < s) ||
                                  (x < i && y >= s && y < t) ||
                                  (x >= j && y >=s && y < t) ||
                                  (x >= i && x < j && y >= t);
              if (conflict && ++numConflics >= allowedPrecisionConflicts) {
                break;
              }
            }
            filter[i][j][s][t] = filter[i][j][s][t] || (numConflics >= allowedPrecisionConflicts);
            
            if (filter[i][j][s][t]) prunedPrecisionCells++;
            totalPrecisionCells++;
          }
        }
      }
    }
//    Logger.logss("Pruned %2.1f / %2.1f big bispans: %.3f",prunedPrecisionCells,totalPrecisionCells, prunedPrecisionCells/totalPrecisionCells);
    
    return filter;
  }

  @Override
public String report() {
    double fracPruned = (prunedPrecisionCells) / (totalPrecisionCells);
    String s = String.format("Pruned %.5f (%d,%d)",fracPruned,prunedPrecisionCells,totalPrecisionCells);
    return s;
  }

}
