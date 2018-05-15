package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;

import java.util.List;

/**
 * Created by IntelliJ IDEA.
* User: aria42
* Date: Nov 21, 2008
*/
public abstract class FeatureLayer {
  public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) { }
  public void addFrNullFeatures(SentencePair sp, List<FeatureValuePair>[] frNullFeats) { }
  public void addEnNullFeatures(SentencePair sp, List<FeatureValuePair>[] enNullFeats) { }
  public void addRuleFeatures(SentencePair sp, List<FeatureValuePair>[][][][][] ruleFeats) { }
  public void addBiSpanFeatures(SentencePair sp, List<FeatureValuePair>[][][][] bispanFeats) {}
  public void addFrBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] frBlockFeats,
                                                  List<FeatureValuePair>[][] alignFeats,
                                                  List<FeatureValuePair>[] frNullFeats,
                                                  List<FeatureValuePair>[] enNullFeats) {}
  public void addEnBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] enBlockFeats,
                                                  List<FeatureValuePair>[][] alignFeats,
                                                  List<FeatureValuePair>[] frNullFeats,
                                                  List<FeatureValuePair>[] enNullFeats) {}
  public boolean hasRuleFeatures() { return false; }
  public boolean hasBiSpanFeatures() { return false; }
}
