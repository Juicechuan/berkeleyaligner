package edu.berkeley.nlp.wordAlignment.symmetric.itg;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 16, 2008
 */
public class Rule {
  public final State parent;
  public Object data;
  public double potential = 0.0;
  //public float potential;
  public int index;

  public boolean isNormal() { return parent.isNormal; }
  
  public Rule(State parent) {
    this.parent = parent;
  }
}
