package edu.berkeley.nlp.wordAlignment.symmetric;

import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wa.mt.SentencePair;
//import edu.berkeley.nlp.wordalign.symmetric.itg.ITGViterbiParser;
import edu.berkeley.nlp.wordAlignment.WordAligner;

// import edu.berkeley.nlp.wordalign.symmetrical.olditg.FastITGViterbiParser;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Oct 20, 2008
 * Time: 4:09:53 PM
 */
public abstract class MatchingWordAligner extends WordAligner {

  public static Alignment makeAlignment(SentencePair sp, boolean[][] alignMatrix) {
    Alignment align = new Alignment(sp,false);
    for (int i = 0; i < sp.getForeignWords().size(); i++) {
      for (int j = 0; j < sp.getEnglishWords().size(); j++) {
        if (alignMatrix[i][j]) {
          align.addAlignment(j,i);
        }
      }
    }
    return align;
  }

  public enum MatchingType { ITG, BIPARTITE }

  @Opt
  public MatchingType matchingType = MatchingType.ITG;

  @Opt
  public double potentialThresh = 0.0;//Double.NEGATIVE_INFINITY;

  public MatchingWordAligner() {
    GlobalOptionParser.fillOptions(this);
  }

  public abstract double[][] getMatchingPotentials(SentencePair sp);

  public int[] getMatching(SentencePair sp) {
//    BipartiteMatchings bm = new BipartiteMatchings();
    double[][] matchingPotentials = getMatchingPotentials(sp);
    MatchingExtractor matcher = null;
    switch (matchingType) {
      case ITG:
////        matcher = new ITGViterbiParser(new NormalFormWithNullGrammarBuilder().buildGrammar()) ;
//        matcher = new FastITGViterbiParser();
//        matcher = new ITGParser();
        break;
      case BIPARTITE:
        matcher = new BipartiteMatchingExtractor() ;
        break;
//      case ITG_SLOW:
//        matcher = new ITGViterbiParser(new DanNormalFormWithNullGrammarBuilder().buildGrammar());
    }
    int[] matching = matcher.extractMatching(matchingPotentials);
    for (int i=0; i < matching.length; ++i) {
    	int j = matching[i];
    	if (j >= 0 && matchingPotentials[i][j] < potentialThresh) matching[i] = -1;
    }
    int m = sp.getForeignWords().size(); 
    int n = sp.getEnglishWords().size();
    for (int i = 0; i < m; ++i) {
      int j = matching[i];
      if (j < 0 || j >= n) matching[i] = -1;
    }
    return matching;
  }

//  public double[][] getMatchingPosteriors(SentencePair sp) {
//    int m = sp.getForeignWords().size();
//    int n = sp.getEnglishWords().size();
//    double[][] matchingPotentials = getMatchingPotentials(sp);
//    BipartiteMatchings bm = new BipartiteMatchings();
//    double[][] allMatchScores = bm.getAllMaxMatchingCosts(matchingPotentials);
//    negateInPlace(allMatchScores);
//    double max = Double.NEGATIVE_INFINITY;
//    for (double[] row : allMatchScores) max = Math.max(max, DoubleArrays.max(row));
//    double[][] posteriors = new double[m + 1][n + 1];
//    for (int i = 0; i <= m; ++i) {
//      for (int j = 0; j <= n; ++j) {
//        double diff = (allMatchScores[i][j] - max);
//        posteriors[i][j] = Math.exp(diff);
//      }
//    }
//    return posteriors;
//  }
//
//  protected void negateInPlace(double[][] potentials) {
//    for (double[] row : potentials) {
//      DoubleArrays.scale(row, -1.0);
//    }
//  }

  public Alignment alignSentencePair(SentencePair sentencePair) {
    int[] matching = getMatching(sentencePair);
    return makeAlignment(sentencePair, matching);
  }
  
  public static Alignment makeAlignment(SentencePair sentencePair, int[] matching) {
    Alignment align = new Alignment(sentencePair, false);
    int m = sentencePair.getForeignWords().size();
    int n = sentencePair.getEnglishWords().size();
    for (int f = 0; f < m; ++f) {
      int e = matching[f];
      if (e >= 0 && e < n) {
    	  align.addAlignment(e, f, true);
      }
    }
    return align;
  }
}
