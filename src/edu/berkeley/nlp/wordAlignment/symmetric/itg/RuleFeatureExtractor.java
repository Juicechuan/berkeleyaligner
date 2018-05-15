package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;

import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 16, 2008
 */
public interface RuleFeatureExtractor {
  public List<FeatureValuePair> extractFeatureValuePairs(Rule rule);
}
