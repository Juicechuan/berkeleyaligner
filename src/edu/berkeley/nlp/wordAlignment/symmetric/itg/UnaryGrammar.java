package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.util.Counter;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 */
public class UnaryGrammar {
  private List<UnaryRule>[] unariesByParentList;
  private List<UnaryRule>[] unariesByChildList;
  private List<State> states;

  public UnaryRule[][] unariesByParent;
  public UnaryRule[][] unariesByChild;

  public State[] topDownOrdering ;
  public State[] bottomUpOrdering ;

  void createOrderings() {
    Counter<State> stateIndices = new Counter<State>();
    int currIdx = 0;

    for (State state : states) {
      for (UnaryRule r : unariesByChild[state.index]) {
        currIdx = addStateAndAbove(r.parent, stateIndices, currIdx);
      }
      if (!stateIndices.containsKey(state)) {
        stateIndices.setCount(state, currIdx);
        currIdx++;
      }
    }

    assert(currIdx == topDownOrdering.length);

    for (State state : states) {
      int idx = (int)(stateIndices.getCount(state));
      topDownOrdering[idx] = state;
    }

    for (int idx = 0; idx < topDownOrdering.length; ++idx) {
      bottomUpOrdering[currIdx-1-idx] = topDownOrdering[idx];
    }
  }

  private int addStateAndAbove(State state, Counter<State> stateIndices, int currIdx) {
      for (UnaryRule r : unariesByChild[state.index]) {
      currIdx = addStateAndAbove(r.parent, stateIndices, currIdx);
      }
      if (!stateIndices.containsKey(state)) {
        stateIndices.setCount(state, currIdx);
        currIdx++;
      }  
      return currIdx;
  }

    private List<UnaryRule>[] buildRuleDB() {
      List<UnaryRule>[] res = new List[states.size()];
      for (int i = 0; i < res.length; i++) {
        res[i] = new ArrayList<UnaryRule>();
      }
      return res;
    }

    UnaryGrammar(List<State> states,List<UnaryRule> unarys) {
      this.states = states;
      topDownOrdering = new State[states.size()];
      bottomUpOrdering = new State[states.size()];
      unariesByParentList = buildRuleDB();
      unariesByChildList = buildRuleDB();
      for (UnaryRule unary : unarys) {
        addUnaryRule(unary);
      }
      unariesByParent = new UnaryRule[states.size()][];
      unariesByChild = new UnaryRule[states.size()][];
      for (int s = 0; s < states.size(); ++s) {
        unariesByParent[s] = unariesByParentList[s].toArray(new UnaryRule[0]);
        unariesByChild[s] = unariesByChildList[s].toArray(new UnaryRule[0]); 
      }
    }

    private void addUnaryRule(UnaryRule ur) {
      unariesByParentList[ur.parent.index].add(ur);
      unariesByChildList[ur.child.index].add(ur);
    }
 }
