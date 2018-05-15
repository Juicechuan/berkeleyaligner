package edu.berkeley.nlp.wordAlignment.symmetric.itg;
//package edu.berkeley.nlp.wordalign.symmetric.itg;
//
//import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
//import edu.berkeley.nlp.util.optionparser.Opt;
//import edu.berkeley.nlp.wa.mt.SentencePair;
//import edu.berkeley.nlp.wordalign.symmetric.MatchingWordAligner;
//import edu.berkeley.nlp.wordalign.symmetric.WordAligner;
//import fig.basic.LogInfo;
//
///**
// * Created by IntelliJ IDEA.
// * User: aria42
// * Date: Nov 7, 2008
// */
//public class WordAlignerBiSpanFilter implements BiSpanFilter {
//  private WordAligner wordAligner;
//
//  @Opt
//  public double filterFrac = 0.5;
//
//  private int numPruned = 0;
//  private int total = 0;
//
//  public boolean[][][][] getBiSpanFilter(double[][] alignmentPotentials,
//                                         double[] frNullPotentials,
//                                         double[] enNullPotentials,
//                                         float[][][][][] rulePotentials) {
//    return new boolean[0][][][];  //To change body of implemented methods use File | Settings | File Templates.
//  }
//
//  public void setWordAligner(MatchingWordAligner wordAligner) {
//    this.wordAligner = wordAligner;
//  }
//
//  public WordAlignerBiSpanFilter(WordAligner wordAligner) {
//    this.wordAligner = wordAligner;
//    GlobalOptionParser.fillOptions(this);
//  }
//
//  public boolean[][][][] getBiSpanFilter(SentencePair sp) {
//    int[] matching = wordAligner.getMatching(sp);
//    int m = sp.getForeignWords().size();
//    int n = sp.getEnglishWords().size();
//    boolean[][][][] filter = new boolean[m+1][m+1][n+1][n+1];
//    for (int sumLen=0; sumLen <= (m+n); ++sumLen) {
//      for (int frLen=0; frLen <= sumLen; ++frLen) {
//        int enLen = sumLen - frLen;
//        for (int frStart = 0; frStart+frLen <= m; ++frStart) {
//          int frStop = frStart+frLen;
//          for (int enStart = 0; enStart+enLen <= n; ++enStart) {
//            int enStop = enStart+enLen;
//            // [frStart,frStop], [enStart,enStop]
//            double matchCount = 0.0;
//            for (int i=frStart; i < frStop; ++i) {
//              int j = matching[i];
//              if (j < 0 || j >= enStart && j <= enStop) {
//                matchCount += 1.0;
//              }
//            }
//            double areaScore = matchCount / Math.max(frLen,enLen);
//            if (areaScore < filterFrac) {
//              numPruned ++;
//              filter[frStart][frStop][enStart][enStop] = true;
//            }
//            total++;
//          }
//        }
//      }
//    }
//    if (m > 30 && n > 30) LogInfo.logs(report());
//
//    return filter;
//  }
//
//  public String report() {
//    double fracPruned = ((double) numPruned) /((double) total);
//    return "fracPruned: " + fracPruned;
//  }
//}
