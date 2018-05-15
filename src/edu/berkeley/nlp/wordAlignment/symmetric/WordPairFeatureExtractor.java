package edu.berkeley.nlp.wordAlignment.symmetric;

import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 16, 2008
 */
public interface WordPairFeatureExtractor {
  public List<FeatureValuePair> extractFeatures(String frWord, String enWord);
  public List<FeatureValuePair> extractFrNullFeatures(String frWord);
  public List<FeatureValuePair> extractEnNullFeatures(String enWord); 
}
