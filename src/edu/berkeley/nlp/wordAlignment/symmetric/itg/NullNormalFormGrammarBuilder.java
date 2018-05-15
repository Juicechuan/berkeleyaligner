package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.util.CollectionUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 15, 2008
 */
public class NullNormalFormGrammarBuilder implements GrammarBuilder {


	public State newState(boolean isNormal, boolean isTerminal, String id) {
	    return new State(isNormal,isTerminal,id);
	  }

  @Override
public Grammar buildGrammar() {
    State Root = newState(true,false,"Root");
    State X = newState(true,false,"X");
    
    State NormalNoNull = newState(true,false,"NormalNoNull");
    State NormalFNull = newState(true,false,"NormalFrNull");
    State NormalENull = newState(true,false,"NormalEnNull");
    State NormalDone = newState(true,false,"NormalDone");

    State InvertedNoNull = newState(true,false,"InvertedNoNull");
    State InvertedFNull = newState(true,false,"InvertedFrNull");
    State InvertedENull = newState(true,false,"InvertedEnNull");
    State InvertedDone = newState(false,false,"InvertedDone");

    State RootENull = newState(true,false,"RootEnNull");
    State RootFNull = newState(true,false,"RootFrNull");
    State RootNoNull = newState(true,false,"RootNoNull");

    State AlignTerm = newState(true,true,"AlignTerm");
    State FrNullTerm = newState(true,true,"FrNullTerm");
    State EnNullTerm = newState(true,true,"EnNullTerm");    

    List<UnaryRule> urs = new ArrayList<UnaryRule>();
    urs.add(new UnaryRule(X, NormalDone));
    urs.add(new UnaryRule(X, InvertedDone));

    // Null Unaries
    urs.add(new UnaryRule(NormalNoNull,AlignTerm));
    urs.add(new UnaryRule(NormalFNull,NormalNoNull));
    urs.add(new UnaryRule(NormalENull,NormalFNull));
    urs.add(new UnaryRule(NormalDone,NormalENull));

    urs.add(new UnaryRule(InvertedNoNull,AlignTerm));
    urs.add(new UnaryRule(InvertedFNull,InvertedNoNull));
    urs.add(new UnaryRule(InvertedENull,InvertedFNull));
    urs.add(new UnaryRule(InvertedDone,InvertedENull));
    
    List<BinaryRule> brs = new ArrayList<BinaryRule>();

    brs.add(new BinaryRule(NormalDone, X, X));
    brs.add(new BinaryRule(InvertedDone, X, X));

    // NULL Binaries
    brs.add(new BinaryRule(NormalFNull,NormalFNull,FrNullTerm)); // add f-nulls downward
    brs.add(new BinaryRule(NormalENull,NormalENull,EnNullTerm)); // inverted, meaning add e-nulls leftward
    brs.add(new BinaryRule(InvertedFNull,InvertedFNull,FrNullTerm)); // add f-nulls downward
    brs.add(new BinaryRule(InvertedENull,InvertedENull,EnNullTerm)); // add e-nulls rightward

    // nulls from root
    urs.add(new UnaryRule(RootNoNull,AlignTerm));
    urs.add(new UnaryRule(RootFNull,RootNoNull));
    
    brs.add(new BinaryRule(RootFNull,RootFNull,FrNullTerm));
    urs.add(new UnaryRule(RootENull,RootFNull));
    
    brs.add(new BinaryRule(RootENull,RootENull,EnNullTerm));
    urs.add(new UnaryRule(Root, RootENull));
    brs.add(new BinaryRule(Root, RootENull, X));
    
    return new Grammar(urs,brs,Root,AlignTerm,FrNullTerm,EnNullTerm,3);    
  }

  public static void main(String[] args) {
    int n = 4;
    double[][] m = new double[n][n];
    for (double[] row : m) {
      Arrays.fill(row,Double.NEGATIVE_INFINITY);
    }
    m[0][2] = m[1][0] = m[2][3] = m[3][1] = 0.0;
    double[] nullPots = new double[n];
    Arrays.fill(nullPots,Double.NEGATIVE_INFINITY);
    List<Grammar> grammars = CollectionUtils.makeList(
        new SimpleGrammarBuilder().buildGrammar(),
        new NormalFormGrammarBuilder().buildGrammar(),
        new NullNormalFormGrammarBuilder().buildGrammar()
    );
    for (Grammar g : grammars) {
      ITGParser p = new ITGParser(g);
      p.setMode(ITGParser.Mode.MAX);
      p.setInput(m,nullPots,nullPots);
      double logZ = p.getLogZ();
      System.out.printf("logZ = %.5f\n",logZ);
    }
  }
}
