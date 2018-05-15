package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 */
public class Grammar {
  public final UnaryGrammar ug;
  public final BinaryGrammar bg;
  public final State alignTerm;
  public final State frNullTerm;
  public final State enNullTerm;
  public final State root;
  public final List<State> states;
  public final List<BinaryRule> binarys;
  public final List<UnaryRule> unarys;
  public final List<Rule> rules;
  public final int minSumLen;
  
  public Grammar(List<UnaryRule> unaryRules,
                 List<BinaryRule> binaryRules,
                 State root,
                 State alignTerm,
                 State frNullTerm,
                 State enNullTerm,
                 int sumLen)
  {
    if (root.isTerminal || !alignTerm.isTerminal ||
        !frNullTerm.isTerminal || !enNullTerm.isTerminal)
    {
      throw new RuntimeException("Error in grammar");  
    }

    this.root = root;
    this.alignTerm = alignTerm;
    this.frNullTerm = frNullTerm;
    this.enNullTerm = enNullTerm;
    this.unarys = Collections.unmodifiableList(unaryRules);
    this.binarys = Collections.unmodifiableList(binaryRules);
    this.minSumLen = sumLen;
    
    Set<State> stateSet = new HashSet<State>();
    for (UnaryRule rule : unaryRules) {
      stateSet.add(rule.parent);
      stateSet.add(rule.child);
    }
    for (BinaryRule binaryRule : binaryRules) {
      stateSet.add(binaryRule.parent);
      stateSet.add(binaryRule.lchild);
      stateSet.add(binaryRule.rchild);
    }
    this.states = new ArrayList<State>(stateSet);
    for (int i = 0; i < states.size(); ++i) {
      State state = states.get(i);
      state.setIndex(i);
    }
    ug = new UnaryGrammar(states,unaryRules);
    ug.createOrderings();
    bg = new BinaryGrammar(states,binaryRules);
    this.rules = new ArrayList<Rule>();
    this.rules.addAll(unaryRules);
    this.rules.addAll(binaryRules);
    for (int i = 0; i < rules.size(); i++) {
      Rule rule = rules.get(i);
      rule.index = i;
    }
  }
}
