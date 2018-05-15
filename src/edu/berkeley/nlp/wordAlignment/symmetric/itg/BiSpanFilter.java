package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.wa.mt.SentencePair;

/**
 * User: aria42
 * Date: Nov 7, 2008
 * Time: 2:30:00 PM
 */
public interface BiSpanFilter {
    public boolean[][][][] getBiSpanFilter(SentencePair sp);    
    public String report();
}
