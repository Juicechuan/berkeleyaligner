package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.util.optionparser.Opt;

/**
 * Created by IntelliJ IDEA.
* User: aria42
* Date: Nov 15, 2008
*/ // Options
public class ParserOptions {
  @Opt
  public double maxEccentricityRatio = Double.MAX_VALUE;

  @Opt
  public int maxEccentrictyAbs = Integer.MAX_VALUE;

  @Opt
  public int inverseMax = Integer.MAX_VALUE;
}
