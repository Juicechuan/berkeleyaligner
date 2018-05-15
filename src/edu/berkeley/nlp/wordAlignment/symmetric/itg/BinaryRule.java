package edu.berkeley.nlp.wordAlignment.symmetric.itg;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 */
public class BinaryRule extends Rule {
  public final State lchild, rchild;
 
  public BinaryRule(State parent, State lchild, State rchild) {
    super(parent);
    this.lchild = lchild;
    this.rchild = rchild;
  }

  @Override
  public String toString() {
    String arrow = isNormal()  ? "-->" : "-|>" ;
    return String.format("%s %s %s %s",parent,arrow,lchild,rchild);
  }
}
