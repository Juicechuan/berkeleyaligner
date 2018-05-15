package edu.berkeley.nlp.wordAlignment.symmetric.features;

import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.FeatureLayer;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.Utils;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DictionaryFeatureLayer extends FeatureLayer {

	HashSet<String> validPairs = new HashSet<String>();
	
	@Opt
	public String dictionary;

    @Opt(required = true)
    public int itgBlockSize = 0;

	private Map<String, Set<String>> frEnDictionary = null;
	private Map<String, Set<String>> enFrDictionary = null;
	
	public DictionaryFeatureLayer(boolean lowercaseWords) {
		GlobalOptionParser.fillOptions(this);
		frEnDictionary = Utils.loadDictionary(dictionary,lowercaseWords,false);
		enFrDictionary = Utils.reverseDictionary(frEnDictionary);
	}
	
	 @Override
	 public void addEnBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] enBlockFeats,
	                                 List<FeatureValuePair>[][] alignmentFeats,
	                                 List<FeatureValuePair>[] frNullFeats,
	                                 List<FeatureValuePair>[] enNullFeats) {

	    int m = sp.getForeignWords().size();
	    int n = sp.getEnglishWords().size();
	    
	    // Find the percent of foreign block items which are a best match to the English
	    for (int i = 0; i < m; ++i) {
	    	String frWord = sp.getForeignWords().get(i);
	    	for (int j = 0; j < n; ++j) {
	    		for (int k = 2; k <= itgBlockSize && j + k <= n; k++) {
	    			double partialMatchCount = 0.0;

	    			// Check first word
	    			String enWord = sp.getEnglishWords().get(j);
                    if (enFrDictionary.containsKey(enWord)) {
                    	if (enFrDictionary.get(enWord).contains(frWord)) {
                    		partialMatchCount++;
                    	}
                    }
	    		    
                    // Check rest of phrase
	    			StringBuffer enPhrase = new StringBuffer(sp.getEnglishWords().get(j));
	    			for (int ell = 1; ell < k; ++ell) {
	    				enPhrase.append(" ");
	    				enPhrase.append(sp.getEnglishWords().get(j+ell));
                       
	    				enWord = sp.getEnglishWords().get(j+ell);
                        if (enFrDictionary.containsKey(enWord)) {
                        	if (enFrDictionary.get(enWord).contains(frWord)) {
                        		partialMatchCount++;
                        	}
                        }
	    			}
			
	    			// Phrase or partial match
                    if (enFrDictionary.containsKey(enPhrase.toString())) {
  	    			  if (enFrDictionary.get(enPhrase.toString()).contains(frWord)) {
  	    				  enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-exact_match", k));
  	    			  } else {
  	    				  if (partialMatchCount > 0) {
  	    					  enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-partial-match-count", partialMatchCount));
  	    				  }
  	    				  enBlockFeats[i][j][k].add(new FeatureValuePair("en-dict-contains-phrase", k));
  	    			  }
                    } else if (partialMatchCount > 0) {
	    				enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-partial-match-count", partialMatchCount));
	    			}
                  
	    		}
	    	}
	    }
	 }

	 @Override
	 public void addFrBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] frBlockFeats,
	                                 List<FeatureValuePair>[][] alignmentFeats,
	                                 List<FeatureValuePair>[] frNullFeats,
	                                 List<FeatureValuePair>[] enNullFeats) {

	    int m = sp.getForeignWords().size();
	    int n = sp.getEnglishWords().size();
	    
	    // Find the percent of foreign block items which are a best match to the English
	    for (int i = 0; i < m; ++i) {
    		StringBuffer frPhrase = new StringBuffer(sp.getForeignWords().get(i));
    		for (int k = 2; k <= itgBlockSize && i + k <= m; k++) {
    			
    			// Check if first word in phrase is there
    			String frWord = sp.getForeignWords().get(i);    			
    			for (int j = 0; j < n; ++j) {
        			double partialMatchCount = 0.0;
    				String enWord = sp.getEnglishWords().get(j);
                    if (frEnDictionary.containsKey(frWord)) {
	    	    		  if (frEnDictionary.get(frWord).contains(enWord)) {
	    	    			  partialMatchCount++;
	    	    		  }
                    }
    	        
                    // Check rest of phrase and build up phrase
                    for (int ell = 1; ell < k; ++ell) {
                    	frPhrase.append(sp.getForeignWords().get(i+ell));
                    	frWord = sp.getForeignWords().get(i);
                        if (frEnDictionary.containsKey(frWord)) {
    	    	    		  if (frEnDictionary.get(frWord).contains(enWord)) {
    	    	    			  partialMatchCount++;
    	    	    		  }
                        }
    	    		}

                    // Phrase or partial match
                    if (frEnDictionary.containsKey(frPhrase.toString())) {
  	    			  if (frEnDictionary.get(frPhrase.toString()).contains(enWord)) {
  	    				  frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-exact_match", k));
  	    			  } else { 
  	    				  if (partialMatchCount > 0) {
  	    					  frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-partial-match-count", partialMatchCount));
  	    				  }
  	    				  frBlockFeats[i][j][k].add(new FeatureValuePair("fr-dict-contains-phrase", k));
  	    			  }
                    } else if (partialMatchCount > 0) {
	    				frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-partial-match-count", partialMatchCount));
	    			}
	    		}
    		}
	    }

	 }
	
    @Override
    public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
      int m = sp.getForeignWords().size();
      int n = sp.getEnglishWords().size();
      
      String frWordUG = new String();
      String frWordBG = new String();
      String frWordTG = new String();
      String frWord4G = new String();
      String frWord5G = new String();
      
      for (int i = 0; i < m; ++i) {
    	  String frWord = sp.getForeignWords().get(i);
    	  if (i > 3) frWord5G = frWord4G + frWord;
    	  if (i > 2) frWord4G = frWordTG + frWord;
    	  if (i > 1) frWordTG = frWordBG + frWord;
    	  if (i > 0) frWordBG = frWordUG + frWord;

    	  for (int j = 0; j < n; ++j) {
    		  String enWord = sp.getEnglishWords().get(j);

    		  if (frEnDictionary.containsKey(frWord)) {
    			  if (frEnDictionary.get(frWord).contains(enWord)) {
    				  alignFeats[i][j].add(new FeatureValuePair("exact_match", 1.0));
    			  }
    		  }
    		  boolean longMatch = false;
    		  if (frEnDictionary.containsKey(frWordBG)) {
    			  if (frEnDictionary.get(frWordBG).contains(enWord)) {
    				  longMatch = true;
    			  }
    		  }
    		  if (frEnDictionary.containsKey(frWordTG)) {
    			  if (frEnDictionary.get(frWordTG).contains(enWord)) {
    				  longMatch = true;
    			  }
    		  }
    		  if (frEnDictionary.containsKey(frWord4G)) {
    			  if (frEnDictionary.get(frWord4G).contains(enWord)) {
    				  longMatch = true;
    			  }
    		  }
    		  if (frEnDictionary.containsKey(frWord5G)) {
    			  if (frEnDictionary.get(frWord5G).contains(enWord)) {
    				  longMatch = true;
    			  }
    		  }
			  if (longMatch) alignFeats[i][j].add(new FeatureValuePair("long_match", 1.0));
    	  }
    	  
    	  frWordUG = frWord;
      }
      
    }
}
