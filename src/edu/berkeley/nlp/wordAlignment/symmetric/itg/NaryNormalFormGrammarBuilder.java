package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 29, 2008
 */
public class NaryNormalFormGrammarBuilder implements GrammarBuilder {
  public State newState(boolean isNormal, boolean isTerminal, String id) {
    return new State(isNormal,isTerminal,id);
  }

  @Override
public Grammar buildGrammar() {
    State Root = newState(true,false,"Root");
    State NormalDone = newState(true,false,"NormalDone");
    State NormalBar = newState(true,false,"NormalBar");
    State InvertedDone = newState(false,false,"InvertedDone");
    State InvertedBar = newState(false,false,"InvertedBar");
    State Aligning = newState(true,false,"Aligning");
    State RootNulls = newState(true,false,"RootNulls");
    
    State AlignTerm = newState(true,true,"AlignTerm");
    State FrNullTerm = newState(true,true,"FrNullTerm");
    State EnNullTerm = newState(true,true,"EnNullTerm");    
        
    List<UnaryRule> urs = new ArrayList<UnaryRule>();
    List<BinaryRule> brs = new ArrayList<BinaryRule>();

    //    X_N' --> X_I
    //    X_I' -|-> X_N
    //    X_N --> <e,f>
    //    X_I --> <e,f>
    //  urs.add(new UnaryRule(Root,NormalDone));

    // Terminals & nulls
    urs.add(new UnaryRule(Aligning,AlignTerm));
    brs.add(new BinaryRule(Aligning,Aligning,FrNullTerm)); // add f-nulls 
    brs.add(new BinaryRule(Aligning,Aligning,EnNullTerm)); // add e-nulls
    
    urs.add(new UnaryRule(NormalDone,Aligning));
    urs.add(new UnaryRule(InvertedDone,Aligning));

    // Build N' out of I, N'
    brs.add(new BinaryRule(NormalBar, InvertedDone, NormalBar));

    // Build I' out of N,I'
    brs.add(new BinaryRule(InvertedBar, NormalDone, InvertedBar));
    
    // Build N' and I' out of Dones
    urs.add(new UnaryRule(NormalBar,InvertedDone));
    urs.add(new UnaryRule(InvertedBar,NormalDone));

    brs.add(new BinaryRule(NormalDone, InvertedDone, NormalBar));
    brs.add(new BinaryRule(InvertedDone, NormalDone, InvertedBar)); // inverted
    
    // Root unaries
    brs.add(new BinaryRule(Root,RootNulls,NormalDone));
    brs.add(new BinaryRule(Root,RootNulls,InvertedDone));
    brs.add(new BinaryRule(RootNulls,RootNulls,FrNullTerm));
    brs.add(new BinaryRule(RootNulls,RootNulls,EnNullTerm));
    urs.add(new UnaryRule(RootNulls,AlignTerm));
    urs.add(new UnaryRule(Root, RootNulls));
    
    return new Grammar(urs,brs,Root,AlignTerm,FrNullTerm,EnNullTerm,3);    
  }
  
}
