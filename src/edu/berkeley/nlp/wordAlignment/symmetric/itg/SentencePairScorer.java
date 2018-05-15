package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.wa.mt.SentencePair;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 */
public interface SentencePairScorer {

  public double[][] getScore(SentencePair sp);

}
