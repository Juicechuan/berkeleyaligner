package edu.berkeley.nlp.wordAlignment.symmetric.itg;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 */
public class UnaryRule extends Rule {
  public final State child;

  public UnaryRule(State parent, State child) {
    super(parent);
    this.child = child;
  }

  @Override
  public String toString() {
    return String.format("%s %s %s",parent,"-->",child);
  }
}
