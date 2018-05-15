package edu.berkeley.nlp.wordAlignment.symmetric.features;

import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.Logger;
import edu.berkeley.nlp.util.functional.Function;
import edu.berkeley.nlp.util.functional.FunctionalUtils;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wa.mt.SentencePairReader.PairDepot;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.FeatureLayer;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class LexicalFeatureLayer extends FeatureLayer {

  HashSet<String> validPairs = new HashSet<String>();
  Set<String> frWords = new HashSet();
  Set<String> enWords = new HashSet();

  @Opt(required = true)
  public int lexicalPairsCount = 0;

  @Opt
  public int itgBlockSize = 0;

  @Opt
  public boolean addBlockLexFeats = true;

  public LexicalFeatureLayer(PairDepot trainingPairs, PairDepot testPairs) {
    GlobalOptionParser.fillOptions(this);

    Counter<Pair<String,String>> allPairs = new Counter();

    for (SentencePair sp : trainingPairs) {
      for (String frw : sp.getForeignWords()) {
        for (String enw : sp.getEnglishWords()) {
          allPairs.incrementCount(Pair.newPair(enw,frw),1.0);
        }
      }
    }

    List<Pair<String,String>> topPairs = allPairs.getSortedKeys().subList(0,lexicalPairsCount);
    for (Pair<String, String> topPair : topPairs) {
      frWords.add(topPair.getSecond());
      enWords.add(topPair.getFirst());
      String p = topPair.getFirst() + "_" + topPair.getSecond();
      validPairs.add(p);
    }
    Logger.logss("TOP LEXICAL PAIRS: " + topPairs.toString() + "\n");
  }

  @Override
  public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();

    for (int i = 0; i < m; ++i) {
      String frWord = sp.getForeignWords().get(i);
      for (int j = 0; j < n; ++j) {
        String enWord = sp.getEnglishWords().get(j);

        if (validPairs.contains(enWord + "_" + frWord)) {
          alignFeats[i][j].add(new FeatureValuePair("lexical_" + enWord + "_" + frWord, 5.0));
        }
      }
    }
  }

  

  private String project(String word, boolean fr) {
    if (fr) return frWords.contains(word) ? word : "X";
    return enWords.contains(word) ? word : "X";
  }

  @Override
  public void addFrBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] frBlockFeats,
                                 List<FeatureValuePair>[][] alignmentFeats,
                                 List<FeatureValuePair>[] frNullFeats,
                                 List<FeatureValuePair>[] enNullFeats) {
    if (!addBlockLexFeats) return;
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    //Utils.AlignmentDecomposition ad = Utils.decomposeAlignment(sp.getAlignment(),itgBlockSize);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        // Fr Block
        String enWord = project(sp.getEnglishWords().get(j),false);
        String frWord = project(sp.getForeignWords().get(i),true);
        for (int k = 2; k <= Math.min(3,itgBlockSize) && i + k <= m; k++) {
          List<String> frPhrase = FunctionalUtils.map(sp.getForeignWords().subList(i,i+k),
              new Function<String, String>() { public String apply(String input) { return project(input,true); }});
          
          frBlockFeats[i][j][k].add(new FeatureValuePair("BlockLexicalFr:" + frPhrase + enWord,1.0));
        }
      }
    }
  }

  @Override
  public void addEnBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] enBlockFeats,
                                 List<FeatureValuePair>[][] alignmentFeats,
                                 List<FeatureValuePair>[] frNullFeats,
                                 List<FeatureValuePair>[] enNullFeats) {
    if (!addBlockLexFeats) return;
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    //Utils.AlignmentDecomposition ad = Utils.decomposeAlignment(sp.getAlignment(),itgBlockSize);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        // Fr Block
        String frWord = project(sp.getForeignWords().get(i),true);
        for (int k = 2; k <= Math.min(3,itgBlockSize) && j + k <= n; k++) {
          List<String> enPhrase = FunctionalUtils.map(sp.getEnglishWords().subList(j,j+k),
              new Function<String, String>() { public String apply(String input) { return project(input,false); }});
          enBlockFeats[i][j][k].add(new FeatureValuePair("BlockLexicalEn:" + enPhrase + frWord,1.0));
        }
      }
    }
  }

}
