package edu.berkeley.nlp.wordAlignment.symmetric;

import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wa.mt.SentencePair;

import java.util.HashSet;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 18, 2008
 */
public class SentencePairFeatures {
  public final List<FeatureValuePair>[][] alignmentFeats;
  public final List<FeatureValuePair>[] frNullFeats;
  public final List<FeatureValuePair>[] enNullFeats;
  public final List<FeatureValuePair>[][][][][]  ruleFeats;
  public final List<FeatureValuePair>[][][][]    bispanFeats;

  // fr,en,block-size => (fr,fr+block-size) <=> en
  public final List<FeatureValuePair>[][][] frBlockFeats;
  // fr,en,block-size => (en,en+block-size) <=> fr
  public final List<FeatureValuePair>[][][] enBlockFeats;

  public boolean[][][][] filter;
  public final int m, n;
  public final SentencePair sp;
  public Object data;

  // Maps alignment --> loss
  public HashSet<Alignment> guessedAlignments;
  public Alignment goldAlignment;
  public double[] guessFeatVector;
  public double loss;
  public int cachedPenalty = -1;
  public boolean cachedSkip = false;

  public SentencePairFeatures(SentencePair sp,
                             List<FeatureValuePair>[][] alignmentFeats,
                             List<FeatureValuePair>[] frNullFeats,
                             List<FeatureValuePair>[] enNullFeats,
                             List<FeatureValuePair>[][][][] bispanFeats,
                             List<FeatureValuePair>[][][][][] ruleFeats,
                             List<FeatureValuePair>[][][] frBlockFeats,
                             List<FeatureValuePair>[][][] enBlockFeats,
                             boolean[][][][] filter) {
    this.sp = sp;
    this.alignmentFeats = alignmentFeats;
    this.frNullFeats = frNullFeats;
    this.enNullFeats = enNullFeats;
    this.bispanFeats = bispanFeats;
    this.ruleFeats = ruleFeats;
    this.frBlockFeats = frBlockFeats;
    this.enBlockFeats = enBlockFeats;
    this.m = alignmentFeats.length;
    this.n = alignmentFeats[0].length;
    this.filter = filter;
    
    resetGuessed();
  }

  public void resetGuessed() {
	  guessedAlignments = new HashSet<Alignment>();
  }

}
