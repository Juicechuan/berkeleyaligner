package edu.berkeley.nlp.wordAlignment.symmetric;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Oct 15, 2008
 * Time: 1:44:14 AM
 */
public interface MatchingExtractor {
  public int[] extractMatching(double[][] matchingPotentials);
}
