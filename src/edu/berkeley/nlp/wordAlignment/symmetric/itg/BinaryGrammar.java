package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 */
public class BinaryGrammar {
  private List<BinaryRule>[] binaryRulesByParentList;
  private List<BinaryRule>[] binaryRulesByLeftChildList;
  private List<BinaryRule>[] binaryRulesByRightChildList;
  private List<State> states;

  public BinaryRule[][] binaryRulesByParent;
  public BinaryRule[][] binaryRulesByLeftChild;
  public BinaryRule[][] binaryRulesByRightChild;


  public BinaryRule[][] normalBinaryRulesByParent;
  public BinaryRule[][] inverseBinaryRulesByParent;
  public BinaryRule[][] normalBinaryRulesByLeftChild;
  public BinaryRule[][] inverseBinaryRulesByLeftChild;
  public BinaryRule[][] normalBinaryRulesByRightChild;
  public BinaryRule[][] inverseBinaryRulesByRightChild;

  public List<BinaryRule>[] normalBinaryRulesByParentList;
  public List<BinaryRule>[] inverseBinaryRulesByParentList;
  public List<BinaryRule>[] normalBinaryRulesByLeftChildList;
  public List<BinaryRule>[] inverseBinaryRulesByLeftChildList;
  public List<BinaryRule>[] normalBinaryRulesByRightChildList;
  public List<BinaryRule>[] inverseBinaryRulesByRightChildList;


  List<BinaryRule>[] buildRuleList() {
    List<BinaryRule>[] res = new List[states.size()];
    for (int i = 0; i < res.length; i++) {
      res[i] = new ArrayList<BinaryRule>();
    }
    return res;
  }

  private static void writeToArray(BinaryRule[][] binarys, List<BinaryRule>[] binarysLists) {
    for (int i = 0; i < binarysLists.length; i++) {
      List<BinaryRule> binaryRulesList =  binarysLists[i];
      binarys[i] = binaryRulesList.toArray(new BinaryRule[0]);
    }
  }

  public BinaryGrammar(List<State> states,List<BinaryRule> binarys) {
    this.states = states;
    binaryRulesByParentList = buildRuleList();
    binaryRulesByLeftChildList = buildRuleList();
    binaryRulesByRightChildList = buildRuleList();
    normalBinaryRulesByParentList = buildRuleList();;
    inverseBinaryRulesByParentList = buildRuleList();;
    normalBinaryRulesByLeftChildList = buildRuleList();;
    inverseBinaryRulesByLeftChildList = buildRuleList();;
    normalBinaryRulesByRightChildList = buildRuleList();;
    inverseBinaryRulesByRightChildList = buildRuleList();;

    for (BinaryRule binary : binarys) {
      addBinaryRule(binary);
    }
    binaryRulesByParent = new BinaryRule[states.size()][];
    writeToArray(binaryRulesByParent,binaryRulesByParentList);
    binaryRulesByLeftChild = new BinaryRule[states.size()][];
    writeToArray(binaryRulesByLeftChild,binaryRulesByLeftChildList);
    binaryRulesByRightChild = new BinaryRule[states.size()][];
    writeToArray(binaryRulesByRightChild,binaryRulesByRightChildList);

    normalBinaryRulesByParent = new BinaryRule[states.size()][];
    writeToArray(normalBinaryRulesByParent,normalBinaryRulesByParentList);
    inverseBinaryRulesByParent = new BinaryRule[states.size()][];
    writeToArray(inverseBinaryRulesByParent,inverseBinaryRulesByParentList);
    normalBinaryRulesByLeftChild = new BinaryRule[states.size()][];;
    writeToArray(normalBinaryRulesByLeftChild,normalBinaryRulesByLeftChildList);
    inverseBinaryRulesByLeftChild = new BinaryRule[states.size()][];;
    writeToArray(inverseBinaryRulesByLeftChild,inverseBinaryRulesByLeftChildList);
    normalBinaryRulesByRightChild = new BinaryRule[states.size()][];;
    writeToArray(normalBinaryRulesByRightChild,normalBinaryRulesByRightChildList);
    inverseBinaryRulesByRightChild = new BinaryRule[states.size()][];;
    writeToArray(inverseBinaryRulesByRightChild,inverseBinaryRulesByRightChildList);    
  }

  private void add(State s, BinaryRule br, List<BinaryRule>[] db) {
    db[s.index].add(br);
  }

  private void addBinaryRule(BinaryRule br) {
    add(br.parent, br, binaryRulesByParentList);
    add(br.lchild, br, binaryRulesByLeftChildList);
    add(br.rchild, br, binaryRulesByRightChildList);

    if (br.isNormal()) {
      add(br.parent,br, normalBinaryRulesByParentList);
      add(br.lchild,br, normalBinaryRulesByLeftChildList);
      add(br.rchild,br, normalBinaryRulesByRightChildList);
    } else {
      add(br.parent,br, inverseBinaryRulesByParentList);
      add(br.lchild,br, inverseBinaryRulesByLeftChildList);
      add(br.rchild,br, inverseBinaryRulesByRightChildList);      
    }
  }

}
