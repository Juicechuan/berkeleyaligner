package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.fig.basic.IOUtils;
import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.util.Logger;
import edu.berkeley.nlp.util.Triple;
import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.symmetric.MatchingWordAligner;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;

/**
 * User: aria42
 * Date: Feb 3, 2009
 */
public class Utils {

  public static Map<String, Set<String>> loadDictionary(String dictFile,
                                                        boolean lowercaseWords,
                                                        boolean splitDefinitions) {
      BufferedReader dictReader = IOUtils.openInEasy(dictFile);
      Map<String,Set<String>> dictionary = new HashMap<String, Set<String>>();
      if (dictReader == null) {
        Logger.logs("Didn't load dictionary");
        return null;
      } else {
        Logger.logs("Loading dictionary from %s", dictFile);

        try {
          while (dictReader.ready()) {
            String[] words = dictReader.readLine().split("\\t");
            String[] translations = words[1].split("/");
            for (int i = 1; i < translations.length; i++) {
              String translation = translations[i];
              if (lowercaseWords) {
                translation = translation.toLowerCase();
              }
              if (splitDefinitions) {
                String[] transwords = translation.split(" ");
                int len = transwords.length;
                for (int j = 0; j < len; j++) {
                  if (!dictionary.containsKey(words[0])) {
                    dictionary.put(words[0], new HashSet<String>());
                  }
                  dictionary.get(words[0]).add(transwords[j]);
                }
              } else {
                if (!dictionary.containsKey(words[0])) {
                  dictionary.put(words[0], new HashSet<String>());
                }
                dictionary.get(words[0]).add(translation);
              }
            }
          }
        }
        catch (IOException e) {
          Logger.err("Problem loading dictionary file: " + dictFile);
          return null;
        }
      }
      Logger.logs("Dictionary has %d entries", dictionary.size());
      return dictionary;
    }

  
  public static Map<String, Set<String>> reverseDictionary(Map<String, Set<String>> dictionary) {
	  HashMap<String, Set<String>> rd = new HashMap<String, Set<String>>();
	  for (String key : dictionary.keySet()) {
		  Set<String> vals = dictionary.get(key);
		  for (String val : vals) {
			  if (!rd.containsKey(val)) {
				  rd.put(val,new HashSet<String>());
			  }
			  rd.get(val).add(key);
		  }
	  }
		
	  return rd;
  }
  
  public static class AlignmentEvalCounts {
    public int proposedSureCount, proposedPossibleCount, proposedCount, sureCount = 0, numCompleted = 0;

    public int totalErrors = 0;

    public void merge(AlignmentEvalCounts other) {
      this.proposedCount += other.proposedCount;
      this.proposedPossibleCount += other.proposedPossibleCount;
      this.proposedSureCount += other.proposedSureCount;
      this.sureCount += other.sureCount;
      this.totalErrors += other.totalErrors;
      this.numCompleted += other.numCompleted;
    }

    @Override
	public String toString() {
      double prec = proposedPossibleCount / (double) proposedCount;
      double recall = proposedSureCount / (double) sureCount;
      double f1 = (prec + recall > 0.0 ? 2 * prec * recall / (prec + recall) : 0.0);
      StringBuilder sb = new StringBuilder();
      sb.append("AER Precision: " + prec + "\n");
      sb.append("AER Recall: " + recall+ "\n");
      sb.append("AER: " + (1.0 - (proposedSureCount + proposedPossibleCount) / (double) (sureCount + proposedCount))+ "\n");
      sb.append("F1: " + f1 + "\n");
      sb.append("Total Errors: " + totalErrors);
      return sb.toString();
      
    }
  }

  public static void updateCounts(AlignmentEvalCounts evalCounts, Alignment referenceAlignment, Alignment proposedAlignment) {
    if (referenceAlignment == null)
      throw new RuntimeException("No reference alignment found ");
    for (int frenchPosition = 0; frenchPosition < proposedAlignment.getForeignLength(); frenchPosition++) {
      for (int englishPosition = 0; englishPosition < proposedAlignment.getEnglishLength(); englishPosition++) {
        boolean proposed = proposedAlignment.containsSureAlignment(englishPosition, frenchPosition);
        boolean sure = referenceAlignment.containsSureAlignment(englishPosition, frenchPosition);
        boolean possible = referenceAlignment.containsPossibleAlignment(englishPosition, frenchPosition);
        if (proposed && sure) evalCounts.proposedSureCount += 1;
        if (proposed && possible) evalCounts.proposedPossibleCount += 1;
        if (proposed) evalCounts.proposedCount += 1;
        if (sure) evalCounts.sureCount += 1;
        if (sure && !proposed) {
          evalCounts.totalErrors ++;
        }
        if (proposed && !possible) {
          evalCounts.totalErrors++;
        }
      }
    }
    evalCounts.numCompleted++;
  }

  public static boolean[][] getSures(Alignment align) {
    int m = align.getForeignLength();
    int n = align.getEnglishLength();
    boolean[][] sures = new boolean[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        sures[i][j] = align.containsSureAlignment(j,i);
      }
    }
    return sures;
  }

  public static boolean[][] getPossibles(Alignment align) {
    int m = align.getForeignLength();
    int n = align.getEnglishLength();
    boolean[][] sures = new boolean[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        sures[i][j] = align.containsPossibleAlignment(j,i);    
      }
    }
    return sures;
  }


  public static boolean isItg(Alignment alignment, int maxBlockSize) {
    int m = alignment.getForeignLength();
    int n = alignment.getEnglishLength();
//    Utils.AlignmentDecomposition alignDecomp = Utils.decomposeAlignment(alignment,maxBlockSize);
//    double[][] alignPots = new double[m][n];
//    for (double[] row : alignPots) {
//      Arrays.fill(row,Double.NEGATIVE_INFINITY);
//    }
//    return false;
    double[][] alignmentPotentials = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        alignmentPotentials[i][j] = alignment.containsSureAlignment(j,i) ?
            0.0 : Double.NEGATIVE_INFINITY;
      }
    }
    double[] frNullPotentials = new double[m];
    for (int i = 0; i < m; i++) {
      frNullPotentials[i] = alignment.getNullAlignedForeignIndices().contains(i) ?
          0.0 : Double.NEGATIVE_INFINITY;
    }
    double[] enNullPotentials = new double[n];
    for (int j = 0; j < n; j++) {
      enNullPotentials[j] = alignment.getNullAlignedEnglishIndices().contains(j) ?
          0.0 : Double.NEGATIVE_INFINITY;
    }
    Grammar grammar = new SimpleGrammarBuilder().buildGrammar();
    ITGParser itgParser = new ITGParser(grammar);
    itgParser.setMode(ITGParser.Mode.MAX);
    itgParser.setInput(alignmentPotentials,frNullPotentials,enNullPotentials);
    double logZ = itgParser.getLogZ() ;
    return logZ > Double.NEGATIVE_INFINITY;
  }

  public static void fixITGAlignment(SentencePair sp, Alignment alignment) {
    int m = alignment.getForeignLength();
    int n = alignment.getEnglishLength();
    double[][] alignPotentials = new double[m][n] ;    
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (alignment.containsSureAlignment(j,i)) {
          alignPotentials[i][j] = 1.0;
        }  else {
          alignPotentials[i][j] = Double.NEGATIVE_INFINITY;
        }
      }
    }
    ITGParser parser = new ITGParser(new SimpleGrammarBuilder().buildGrammar());
    parser.setMode(ITGParser.Mode.MAX);
    parser.setInput(alignPotentials,new double[m], new double[n]);
    Alignment itgAlignment = MatchingWordAligner.makeAlignment(sp, parser.getMatching());
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (alignment.containsSureAlignment(j,i) && !itgAlignment.containsSureAlignment(j,i)) {
          alignment.removeAlignment(j,i);
        }
      }
    }
  }

  public static class AlignmentDecomposition {
    public List<Pair<Integer,Integer>> isolatedAlignments;
    public List<Triple<Integer,Integer,Integer>> frBlocks;
    public List<Triple<Integer,Integer,Integer>> enBlocks;
    public List<Integer> frNulls;
    public List<Integer> enNulls;
  }

  private static boolean aligned(Alignment align, int i, int j, boolean allowSure) {
    return align.containsSureAlignment(j,i) ||
           (allowSure && align.containsPossibleAlignment(j,i));
  }

  public static double[][][][] getSumSures(double[][] alignPosts, boolean pad) {
    int m = alignPosts.length  + (pad ? 1 : 0);
    int n = alignPosts[0].length + (pad ? 1 : 0);
    double[][][][] sumSures = new double[m + 1][m + 1][n + 1][n + 1];
    for (int i = 0; i <= m; i++) {
      for (int j = i; j <= m; j++) {
        for (int s = 0; s <= n; s++) {
          for (int t = s; t <= n; t++) {
            double sum = 0;
            for (int i1 = i; i1 < j; i1++) {
              for (int j1 = s; j1 < t; j1++) {
                sum += alignPosts[i1][j1];
              }
            }
            sumSures[i][j][s][t] = sum;
          }
        }
      }
    }
    return sumSures;

  }

  public static int[][][][] getNumSures(Alignment alignment, boolean pad) {
    int m = alignment.getForeignLength() + (pad ? 1 : 0);
    int n = alignment.getEnglishLength() + (pad ? 1 : 0);
    int[][][][] numSures = new int[m + 1][m + 1][n + 1][n + 1];
    for (int i = 0; i <= m; i++) {
      for (int j = i; j <= m; j++) {
        for (int s = 0; s <= n; s++) {
          for (int t = s; t <= n; t++) {
            int num = 0;
            // Upper Left
            for (edu.berkeley.nlp.fig.basic.Pair<Integer, Integer> pair : alignment.getSureAlignments()) {
              int aj = pair.getFirst()  + (pad ? 1 : 0);
              int ai = pair.getSecond() + (pad ? 1 : 0);
              if (ai >= i && ai < j && aj >= s && aj < t) num++;
            }
            numSures[i][j][s][t] = num;
          }
        }
      }
    }
    return numSures;
  }

  public static Pair<List<Triple<Integer,Integer,Integer>>,
                     List<Triple<Integer,Integer,Integer>>> findBlocks(Alignment align, int maxBlockSize, boolean allowPossible)
  {
    int m = align.getForeignLength();
    int n = align.getEnglishLength();
    List<Triple<Integer,Integer,Integer>> frBlocks = new ArrayList<Triple<Integer, Integer, Integer>>();
    List<Triple<Integer,Integer,Integer>> enBlocks = new ArrayList<Triple<Integer, Integer, Integer>>();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (!aligned(align,i,j,allowPossible)) continue;
        int frRight = 0;
        while (i + frRight < m && frRight <= maxBlockSize &&
            aligned(align,i+frRight,j,allowPossible)) {
          frRight++;
        }
        int enRight = 0;
        while (j + enRight < n && enRight <= maxBlockSize &&
            aligned(align,i,j+enRight,allowPossible)) {            
          enRight++;
        }
        int frLen = frRight;
        int enLen = enRight;
        if (frLen > 1) {
          Triple t = Triple.makeTriple(i,j,frLen);
          frBlocks.add(t);
        }
        if (enLen > 1) {
          Triple t = Triple.makeTriple(i,j,enLen);
          enBlocks.add(t);
        }
      }
    }
    return Pair.newPair(frBlocks,enBlocks);
  }


  public static AlignmentDecomposition decomposeAlignment(Alignment align, int maxBlockSize) {
    int m = align.getForeignLength();
    int n = align.getEnglishLength();
    AlignmentDecomposition ad = new AlignmentDecomposition();
    ad.isolatedAlignments = new ArrayList<Pair<Integer, Integer>>();
    ad.frBlocks = new ArrayList<Triple<Integer, Integer, Integer>>();
    ad.enBlocks = new ArrayList<Triple<Integer, Integer, Integer>>();
    ad.frNulls = new ArrayList<Integer>();
    ad.enNulls = new ArrayList<Integer>();
    boolean[][] marked = new boolean[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (!align.containsSureAlignment(j,i) || marked[i][j]) continue;
        int frRight = 0;
        while (i + frRight < m && frRight <= maxBlockSize &&            
            align.containsSureAlignment(j,i+frRight) && !marked[i+frRight][j]) {
          frRight++;
        }
        int enRight = 0;
        while (j + enRight < n && enRight <= maxBlockSize &&            
            align.containsSureAlignment(j+enRight,i) && !marked[i][j+enRight]) {
          enRight++;
        }
        int frLen = frRight;
        int enLen = enRight;
        if (frLen == 1 && enLen == 1)  {
          ad.isolatedAlignments.add(Pair.newPair(i,j));
          marked[i][j] = true;
          continue;
        }
        boolean frBlock = frLen >= enLen;
        if (frBlock) {
          Triple t = Triple.makeTriple(i,j,frLen);
          for (int k=0; k < frLen; ++k) {
            marked[i+k][j] = true;
          }
          ad.frBlocks.add(t);
          continue;
        } else {
          Triple t = Triple.makeTriple(i,j,enLen);
          for (int k=0; k < enLen; ++k) {
            marked[i][j+k] = true;
          }
          ad.enBlocks.add(t);
          continue;
        }
      }
    }
    for (int i : align.getNullAlignedForeignIndices()) {
      ad.frNulls.add(i);
    }
    for (Integer j : align.getNullAlignedEnglishIndices()) {
      ad.enNulls.add(j);
    }
    return ad;
  }
}
