package edu.berkeley.nlp.wordAlignment.symmetric;

import edu.berkeley.nlp.fig.basic.Interner;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.BiSpanFilter;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.FeatureLayer;
import edu.berkeley.nlp.wordAlignment.symmetric.itg.Grammar;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 15, 2008
 */
public class  SentencePairFeatureExtractor {
  public Interner<Feature> featureInterner;
  final public BiSpanFilter biSpanFilter;

  @Opt
  public int itgBlockSize;

  final protected Grammar grammar;
  final protected List<FeatureLayer> featLayers;

  public SentencePairFeatureExtractor(Grammar grammar,
                                      BiSpanFilter biSpanFilter,
                                      List<FeatureLayer> featLayers) {
    this.grammar = grammar;
    this.featLayers = featLayers;
    this.biSpanFilter = biSpanFilter;
    GlobalOptionParser.fillOptions(this);
  }

  protected List<FeatureValuePair>[][] extractAlignmentFeatures(SentencePair sp) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    List<FeatureValuePair>[][] alignFeats = new ArrayList[m][n];
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        alignFeats[i][j] = new ArrayList<FeatureValuePair>();
      }
    }
    for (FeatureLayer featLayer : featLayers) {
      featLayer.addAlignmentFeatures(sp, alignFeats);
    }
    return alignFeats;
  }

  protected List<FeatureValuePair>[] extractFrNullFeatures(SentencePair sp) {
    int m = sp.getForeignWords().size();
    List<FeatureValuePair>[] frNullFeats = new ArrayList[m];
    for (int i = 0; i < m; ++i) {
      frNullFeats[i] = new ArrayList<FeatureValuePair>();
    }
    for (FeatureLayer featLayer : featLayers) {
      featLayer.addFrNullFeatures(sp, frNullFeats);
    }
    return frNullFeats;
  }

  protected List<FeatureValuePair>[] extractEnNullFeatures(SentencePair sp) {
    int n = sp.getEnglishWords().size();
    List<FeatureValuePair>[] enNullFeats = new ArrayList[n];
    for (int i = 0; i < n; ++i) {
      enNullFeats[i] = new ArrayList<FeatureValuePair>();
    }
    for (FeatureLayer featLayer : featLayers) {
      featLayer.addEnNullFeatures(sp, enNullFeats);
    }
    return enNullFeats;
  }

  private boolean hasBiSpanFeatures() {
    for (FeatureLayer featLayer : featLayers) {
      if (featLayer.hasBiSpanFeatures()) return true;
    }
    return false;
  }

  private boolean hasRuleFeatures() {
    for (FeatureLayer featLayer : featLayers) {
      if (featLayer.hasRuleFeatures()) {
        return true;
      }
    }
    return false;
  }

  protected List<FeatureValuePair>[][][] extractFrBlockFeatures(SentencePair sp,
                                                                List<FeatureValuePair>[][] alignmentFeats,
                                                                List<FeatureValuePair>[] frNullFeats,
                                                                List<FeatureValuePair>[] enNullFeats) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    List<FeatureValuePair>[][][] frBlockFeats = new List[m][n][itgBlockSize + 1];
    for (List<FeatureValuePair>[][] listss : frBlockFeats) {
      for (List<FeatureValuePair>[] lists : listss) {
        for (int i = 0; i < lists.length; i++) {
          lists[i] = new ArrayList<FeatureValuePair>();          
        }
      }
    }
    for (FeatureLayer featLayer : featLayers) {
      featLayer.addFrBlockFeatures(sp, frBlockFeats, alignmentFeats, frNullFeats, enNullFeats);
    }
    return frBlockFeats;
  }

  protected List<FeatureValuePair>[][][] extractEnBlockFeatures(SentencePair sp,
                                                                List<FeatureValuePair>[][] alignmentFeats,
                                                                List<FeatureValuePair>[] frNullFeats,
                                                                List<FeatureValuePair>[] enNullFeats) {
    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();
    List<FeatureValuePair>[][][] enBlockFeats = new List[m][n][itgBlockSize + 1];
    for (List<FeatureValuePair>[][] listss : enBlockFeats) {
      for (List<FeatureValuePair>[] lists : listss) {
        for (int i = 0; i < lists.length; i++) {
          lists[i] = new ArrayList<FeatureValuePair>();
        }
      }
    }
    for (FeatureLayer featLayer : featLayers) {
      featLayer.addEnBlockFeatures(sp, enBlockFeats, alignmentFeats, frNullFeats, enNullFeats);
    }
    return enBlockFeats;
  }


//  protected List<FeatureValuePair>[][][][] extractBiSpanFeatures(SentencePair sp) {
//    if (!hasBiSpanFeatures()) return null;
//    int m = sp.getForeignWords().size() + 1;
//    int n = sp.getEnglishWords().size() + 1;
//    List<FeatureValuePair>[][][][] bispanFeats = new List[m + 1][m + 1][n + 1][n + 1];
////    for (int frStart = 0; frStart <= m; ++frStart) {
////      for (int frStop = frStart; frStop <= m; ++frStop) {
////        for (int enStart = 0; enStart <= n; ++enStart) {
////          for (int enStop = enStart; enStop <= n; ++enStop) {
////            //bispanFeats[frStart][frStop][enStart][enStop] = new ArrayList<FeatureValuePair>();
////          }
////        }
////      }
////    }
//    for (FeatureLayer featLayer : featLayers) {
//      if (featLayer.hasBiSpanFeatures()) {
//        featLayer.addBiSpanFeatures(sp, bispanFeats);
//      }
//    }
//    return bispanFeats;
//  }

//  protected List<FeatureValuePair>[][][][][] extractRuleFeatures(SentencePair sp) {
//    if (!hasRuleFeatures()) {
//      return null;
//    }
//    int m = sp.getForeignWords().size() + 1;
//    int n = sp.getEnglishWords().size() + 1;
//    int numRules = grammar.rules.size();
//    List<FeatureValuePair>[][][][][] ruleFeats = new List[m + 1][m + 1][n + 1][n + 1][numRules];
//    for (int frStart = 0; frStart <= m; ++frStart) {
//      for (int frStop = frStart; frStop <= m; ++frStop) {
//        for (int enStart = 0; enStart <= n; ++enStart) {
//          for (int enStop = enStart; enStop <= n; ++enStop) {
//            for (int r = 0; r < numRules; ++r) {
//              ruleFeats[frStart][frStop][enStart][enStop][r] = new ArrayList<FeatureValuePair>();
//            }
//          }
//        }
//      }
//    }
//    for (FeatureLayer featLayer : featLayers) {
//      featLayer.addRuleFeatures(sp, ruleFeats);
//    }
//    return ruleFeats;
//  }

  public Grammar getGrammar() {
    return grammar;
  }

//  public void internRuleFeatures(List<FeatureValuePair> rulesFeats[]) {
//    for (List<FeatureValuePair> rulesFeat : rulesFeats) {
//      intern(rulesFeat);
//    }
//  }

  private void intern(List<FeatureValuePair> fvps, boolean createNew) {
    if (fvps == null) return;
    boolean anyToRemove = false;
    for (FeatureValuePair fvp : fvps) {
      assert !fvp.feat.feature.toString().matches("^.*\\t+.*$");
      fvp.feat =  createNew ?
        featureInterner.intern(fvp.feat) : featureInterner.getCanonical(fvp.feat);
      if (fvp.feat == null) {
        anyToRemove = true;
        throw new RuntimeException("null feature");
      }
    }
    if (!createNew && anyToRemove) {
      Iterator<FeatureValuePair> it = fvps.iterator();
      while (it.hasNext()) {
        FeatureValuePair fvp = it.next();
        if (fvp.feat == null) it.remove();
      }
    }    
  }


  private void intern(SentencePairFeatures spf, boolean createNew) {
    for (List<FeatureValuePair>[] featLists : spf.alignmentFeats) {
      for (List<FeatureValuePair> featList : featLists) {
        intern(featList,createNew);
      }
    }
    for (List<FeatureValuePair> fvps : spf.frNullFeats) {
      intern(fvps,createNew);
    }
    for (List<FeatureValuePair> fvps : spf.enNullFeats) {
      intern(fvps,createNew);
    }
    if (spf.bispanFeats != null) {
      for (List<FeatureValuePair>[][][] feats3d : spf.bispanFeats) {
        for (List<FeatureValuePair>[][] feats2d : feats3d) {
          for (List<FeatureValuePair>[] featsLists : feats2d) {
            for (List<FeatureValuePair> featsList : featsLists) {
              intern(featsList,createNew);
            }
          }
        }
      }
    }
    if (spf.ruleFeats != null) {
      for (List<FeatureValuePair>[][][][] feats4d : spf.ruleFeats) {
        for (List<FeatureValuePair>[][][] feats3d : feats4d) {
          for (List<FeatureValuePair>[][] feats2d : feats3d) {
            for (List<FeatureValuePair>[] featsLists : feats2d) {
              for (List<FeatureValuePair> fvps : featsLists) {
                intern(fvps,createNew);
              }
            }
          }
        }
      }
    }

    for (List<FeatureValuePair>[][] m3 : spf.frBlockFeats) {
      for (List<FeatureValuePair>[] m2 : m3) {
        for (List<FeatureValuePair> fvps : m2) {
          intern(fvps,createNew);
        }
      }
    }

    for (List<FeatureValuePair>[][] m3 : spf.enBlockFeats) {
      for (List<FeatureValuePair>[] m2 : m3) {
        for (List<FeatureValuePair> fvps : m2) {
          intern(fvps,createNew);
        }
      }
    }

  }

//  static int count = 0;
//  static long extractTime = 0L;
//  static long filterTime = 0L;

  public SentencePairFeatures getSentencePairFeatures(SentencePair sp, boolean createNew) {
    long start = System.currentTimeMillis();
    List<FeatureValuePair>[][] alignmentFeats = extractAlignmentFeatures(sp);
    List<FeatureValuePair>[] frNullFeats = extractFrNullFeatures(sp);
    List<FeatureValuePair>[] enNullFeats = extractEnNullFeatures(sp);
    //List<FeatureValuePair>[][][][][] ruleFeats = extractRuleFeatures(sp);
    ///List<FeatureValuePair>[][][][] bispanFeats = extractBiSpanFeatures(sp);
    List<FeatureValuePair>[][][] frBlockFeats = extractFrBlockFeatures(sp,alignmentFeats,frNullFeats,enNullFeats);
    List<FeatureValuePair>[][][] enBlockFeats = extractEnBlockFeatures(sp,alignmentFeats,frNullFeats,enNullFeats);

    //extractTime += (System.currentTimeMillis()-start);
    start = System.currentTimeMillis();

    boolean[][][][] filter = biSpanFilter != null ? biSpanFilter.getBiSpanFilter(sp) : null;

//    filterTime += (System.currentTimeMillis()-start);
//    if (++count % 100 == 0) {
//      Logger.logs("filterSecs: %.3f",filterTime/1000.0);
//      Logger.logs("extractSecs: %.3f",extractTime/1000.0);
//    }

    SentencePairFeatures sfs = new SentencePairFeatures(sp, alignmentFeats, frNullFeats, enNullFeats, null, null,
        frBlockFeats, enBlockFeats, filter);
    intern(sfs, createNew);
    sfs.goldAlignment = sp.getAlignment();
    return sfs;
  }

}
      