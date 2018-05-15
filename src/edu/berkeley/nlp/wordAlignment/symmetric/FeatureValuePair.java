package edu.berkeley.nlp.wordAlignment.symmetric;

import edu.berkeley.nlp.fig.basic.Pair;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 16, 2008
 */
public class FeatureValuePair {
  public Feature feat;
  public double value;

  public FeatureValuePair(Object f, double val) {
    if (Double.isInfinite(val) || Double.isNaN(val)) {
      throw new RuntimeException("Bad feature val: " + val);
    }
    this.feat = new Feature(f);
    this.value = val;
  }

  public FeatureValuePair(Feature feat, double val) {
    this.feat = feat;
    this.value = val;
  }

  public String toString() {
    return String.format("(%s,%.3f)",feat.feature,value);
  }
}
