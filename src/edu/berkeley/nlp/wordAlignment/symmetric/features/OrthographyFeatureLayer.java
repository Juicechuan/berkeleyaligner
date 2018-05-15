package edu.berkeley.nlp.wordAlignment.symmetric.features;

import java.util.HashSet;
import java.util.List;

import edu.berkeley.nlp.fig.basic.LogInfo;
import edu.berkeley.nlp.fig.basic.PriorityQueue;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wa.mt.SentencePairReader.PairDepot;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.FeatureLayer;

public class OrthographyFeatureLayer extends FeatureLayer {

    public OrthographyFeatureLayer() { }

    @Override
    public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
      int m = sp.getForeignWords().size();
      int n = sp.getEnglishWords().size();
      
      for (int i = 0; i < m; ++i) {
    	  String frWord = sp.getForeignWords().get(i);
    	  for (int j = 0; j < n; ++j) {
    		  String enWord = sp.getEnglishWords().get(j);

    		  boolean begin1Gram = false; boolean begin2Gram = false;
    		  boolean begin3Gram = false; boolean begin4Gram = false;
    		  if ((enWord.length() > 0) && (frWord.length() > 0)) 
    			  begin1Gram = enWord.substring(0, 1).equals(frWord.substring(0,1));
    		  if ((enWord.length() > 1) && (frWord.length() > 1))
    			  begin2Gram = enWord.substring(0, 2).equals(frWord.substring(0,2)); 
    		  if ((enWord.length() > 2) && (frWord.length() > 2))
    			  begin3Gram = enWord.substring(0, 3).equals(frWord.substring(0,3));
    		  if ((enWord.length() > 3) && (frWord.length() > 3))
    			  begin4Gram = enWord.substring(0, 4).equals(frWord.substring(0,4)); 
    		      		  
    		  boolean end1Gram = false; boolean end2Gram = false;
    		  boolean end3Gram = false; boolean end4Gram = false;
    		  if ((enWord.length() > 0) && (frWord.length() > 0)) 
    			  end1Gram = enWord.substring(enWord.length()-1, enWord.length()).equals(frWord.substring(frWord.length()-1,frWord.length()));
    		  if ((enWord.length() > 1) && (frWord.length() > 1))
    			  end2Gram = enWord.substring(enWord.length()-2, enWord.length()).equals(frWord.substring(frWord.length()-2,frWord.length()));
    		  if ((enWord.length() > 2) && (frWord.length() > 2))
    			  end3Gram = enWord.substring(enWord.length()-3, enWord.length()).equals(frWord.substring(frWord.length()-3,frWord.length()));
    		  if ((enWord.length() > 3) && (frWord.length() > 3))
    			  end4Gram = enWord.substring(enWord.length()-4, enWord.length()).equals(frWord.substring(frWord.length()-4,frWord.length()));
    		  
    		  if (begin1Gram) alignFeats[i][j].add(new FeatureValuePair("begin1Gram", 5.0));  
    		  if (begin2Gram) alignFeats[i][j].add(new FeatureValuePair("begin2Gram", 5.0));
    		  if (begin3Gram) alignFeats[i][j].add(new FeatureValuePair("begin3Gram", 5.0));
    		  if (begin4Gram) alignFeats[i][j].add(new FeatureValuePair("begin4Gram", 5.0));
    		  
    		  if (end1Gram) alignFeats[i][j].add(new FeatureValuePair("end1Gram", 5.0));  
    		  if (end2Gram) alignFeats[i][j].add(new FeatureValuePair("end2Gram", 5.0));
    		  if (end3Gram) alignFeats[i][j].add(new FeatureValuePair("end3Gram", 5.0));
    		  if (end4Gram) alignFeats[i][j].add(new FeatureValuePair("end4Gram", 5.0));
    	  }
       }
    }
    
}
