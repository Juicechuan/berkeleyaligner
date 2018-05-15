package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 15, 2008
 */
public class SimpleGrammarBuilder implements GrammarBuilder{


  @Override
public Grammar buildGrammar() {
    State X = new State(true,false,"X");
    State XNormal = new State(true,false,"XNormal");
    State XInverted = new State(false,false,"XInverted");

    State AlignTerm = new State(true,true,"AlignTerm");
    State FrNullTerm = new State(true,true,"FrNullTerm");
    State EnNullTerm = new State(true,true,"EnNullTerm");
    State Root = new State(true,false,"Root");
    State RootNulls = new State(true,false,"RootNulls");
    
    List<UnaryRule> unarys = new ArrayList<UnaryRule>();
    List<BinaryRule> binarys = new ArrayList<BinaryRule>();
    
    binarys.add(new BinaryRule(Root,RootNulls,X));    
    binarys.add(new BinaryRule(RootNulls,RootNulls,FrNullTerm));
    binarys.add(new BinaryRule(RootNulls,RootNulls,EnNullTerm));
    binarys.add(new BinaryRule(X,X,FrNullTerm));
    binarys.add(new BinaryRule(X,X,EnNullTerm));
    unarys.add(new UnaryRule(RootNulls,AlignTerm));
    unarys.add(new UnaryRule(Root,RootNulls));
    
    unarys.add(new UnaryRule(X,XNormal));
    unarys.add(new UnaryRule(X,XInverted));
    unarys.add(new UnaryRule(X,AlignTerm));

    binarys.add(new BinaryRule(XNormal,X,X));
    binarys.add(new BinaryRule(XInverted,X,X));

    return new Grammar(unarys,binarys,Root,AlignTerm,FrNullTerm,EnNullTerm,3);
  }
}
