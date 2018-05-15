package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 29, 2008
 */
public class NormalFormGrammarBuilder implements GrammarBuilder {
  public State newState(boolean isNormal, boolean isTerminal, String id) {
    return new State(isNormal,isTerminal,id);
  }

  @Override
public Grammar buildGrammar() {
    State Root = newState(true,false,"Root");
    State NormalNoNull = newState(true,false,"NormalNoNull");
    State NormalFNull = newState(true,false,"NormalFrNull");
    State NormalENull = newState(false,false,"NormalEnNull");
    State NormalDone = newState(true,false,"NormalDone");
    State NormalBar = newState(true,false,"NormalBar");

    State InvertedNoNull = newState(false,false,"InvertedNoNull");
    State InvertedFNull = newState(true,false,"InvertedFrNull");
    State InvertedENull = newState(true,false,"InvertedEnNull");
    State InvertedDone = newState(true,false,"InvertedDone");
    State InvertedBar = newState(false,false,"InvertedBar");

    State AlignTerm = newState(true,true,"AlignTerm");
    State FrNullTerm = newState(true,true,"FrNullTerm");
    State EnNullTerm = newState(true,true,"EnNullTerm");    

    State RootENull = newState(true,false,"RootENull");
    State RootFNull = newState(true,false,"RootFNull");    

    List<UnaryRule> urs = new ArrayList<UnaryRule>();
    //    X_N' --> X_I
    //    X_I' -|-> X_N
    //    X_N --> <e,f>
    //    X_I --> <e,f>
    //  urs.add(new UnaryRule(Root,NormalDone));

    urs.add(new UnaryRule(NormalNoNull,AlignTerm));
    urs.add(new UnaryRule(InvertedNoNull,AlignTerm));
    urs.add(new UnaryRule(NormalBar,InvertedNoNull));
    urs.add(new UnaryRule(InvertedBar,NormalNoNull));

    // Null Unaries
    urs.add(new UnaryRule(NormalFNull,NormalNoNull));
    urs.add(new UnaryRule(NormalENull,NormalFNull));
    urs.add(new UnaryRule(NormalDone,NormalENull));

    urs.add(new UnaryRule(InvertedFNull,InvertedNoNull));
    urs.add(new UnaryRule(InvertedENull,InvertedFNull));
    urs.add(new UnaryRule(InvertedDone,InvertedENull));

    List<BinaryRule> brs = new ArrayList<BinaryRule>();

    brs.add(new BinaryRule(NormalBar, InvertedDone, NormalBar));
    brs.add(new BinaryRule(InvertedBar, NormalDone, InvertedBar));
    brs.add(new BinaryRule(NormalNoNull, InvertedDone, NormalBar));
    brs.add(new BinaryRule(InvertedNoNull, NormalDone, InvertedBar)); // inverted

    // NULL Binaries
    brs.add(new BinaryRule(NormalFNull,NormalFNull,FrNullTerm)); // add f-nulls downward
    brs.add(new BinaryRule(NormalENull,NormalENull,EnNullTerm)); // inverted, meaning add e-nulls leftward
    brs.add(new BinaryRule(InvertedFNull,InvertedFNull,FrNullTerm)); // add f-nulls downward
    brs.add(new BinaryRule(InvertedENull,InvertedENull,EnNullTerm)); // add e-nulls rightward

    brs.add(new BinaryRule(RootFNull,RootFNull,FrNullTerm)); // add f-nulls downward
    brs.add(new BinaryRule(RootENull,RootENull,EnNullTerm)); // add e-nulls rightward
    urs.add(new UnaryRule(RootFNull,NormalNoNull));
    urs.add(new UnaryRule(RootENull,RootFNull));
    urs.add(new UnaryRule(Root,RootENull));

    return new Grammar(urs,brs,Root,AlignTerm,FrNullTerm,EnNullTerm,3);    
  }
}
