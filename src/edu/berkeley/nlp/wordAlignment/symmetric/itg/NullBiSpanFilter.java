package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.wa.mt.SentencePair;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 16, 2008
 */
public class NullBiSpanFilter implements BiSpanFilter {

  public NullBiSpanFilter() {}

  public boolean[][][][] getBiSpanFilter(double[][] alignmentPotentials,
                                         double[] frNullPotentials,
                                         double[] enNullPotentials,
                                         float[][][][][] rulePotentials) {
    return null;
  }

  @Override
public String report() {
    return "";
  }

  @Override
public boolean[][][][] getBiSpanFilter(SentencePair sp) {
	return null;
  }
	
}
