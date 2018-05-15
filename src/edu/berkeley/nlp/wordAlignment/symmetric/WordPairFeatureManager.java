package edu.berkeley.nlp.wordAlignment.symmetric;

import edu.berkeley.nlp.classify.FeatureExtractor;
import edu.berkeley.nlp.fig.basic.Interner;
import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.MapFactory;
import edu.berkeley.nlp.wa.mt.SentencePair;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Oct 10, 2008
 * Time: 2:02:14 PM
 */
public class WordPairFeatureManager {

  private final Feature queryFeat = new Feature();

  private final Interner<Feature> featInfoInterner = new Interner<Feature>(new Interner.CanonicalFactory<Feature>() {
            public Feature build(Feature feat) {
              Feature canonical = new Feature();
              canonical.feature = feat.feature;
              canonical.index = featInfoInterner.size();
              return canonical;
            }
          });

  private final Map<String, CounterMap<String, Feature>> onWordPairFeatureMap = new HashMap<String, CounterMap<String, Feature>>();

  public Feature getFeature(Object feat) {
    queryFeat.feature = feat;
    Feature result = featInfoInterner.getCanonical(queryFeat);
    return result != queryFeat ? result : null;
  }

  public Map<String, CounterMap<String, Feature>> getWordPairFeatureMap() {
    return Collections.unmodifiableMap(onWordPairFeatureMap);
  }

  private void makeFeatures(Iterable<SentencePair> sentPairIterable,
                            FeatureExtractor<Pair<String, String>, Object> wordPairExtractor,
                            Map<String, CounterMap<String, Feature>> featureMap) {
    Map<String, Set<String>> cooc = new HashMap<String, Set<String>>();
    for (SentencePair sentPair : sentPairIterable) {
      for (String srcWord : sentPair.getForeignWords()) {
        for (String trgWord : sentPair.getEnglishWords()) {
          CollectionUtils.addToValueSet(cooc, srcWord, trgWord);
        }
      }
    }
    for (Map.Entry<String, Set<String>> entry : cooc.entrySet()) {
      String srcWord = entry.getKey();
      Set<String> trgWords = entry.getValue();
      CounterMap<String, Feature> innerMap = new CounterMap<String, Feature>(
          new MapFactory.HashMapFactory<String,Counter<Feature>>(),
          new MapFactory.IdentityHashMapFactory<Feature,Double>()
      );
      featureMap.put(srcWord, innerMap);
      for (String trgWord : trgWords) {
        Counter<Object> featCounter = wordPairExtractor.extractFeatures(Pair.makePair(srcWord, trgWord));
        for (Map.Entry<Object, Double> centry : featCounter.entrySet()) {
          Object featStr = centry.getKey();
          double count = centry.getValue();
          queryFeat.feature = featStr;
          Feature fi = featInfoInterner.intern(queryFeat);
          innerMap.setCount(trgWord,fi,count);
        }
      }
    }

  }

  public void makeFeatures(Iterable<SentencePair> sentPairIterable,
                           FeatureExtractor<Pair<String, String>, Object> wordPairExtractor) {
    makeFeatures(sentPairIterable, wordPairExtractor, onWordPairFeatureMap);
  }

  public Set<Feature> getFeatures() {
    return featInfoInterner.getCanonicalElements();
  }

  public int getNumFeatures() {
    return featInfoInterner.size();
  }

}
