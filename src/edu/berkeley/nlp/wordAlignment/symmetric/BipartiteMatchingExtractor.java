package edu.berkeley.nlp.wordAlignment.symmetric;

import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.optimize.BipartiteMatchings;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.PriorityQueue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BipartiteMatchingExtractor implements MatchingExtractor {

//  private void filterTopN(double[][] potentials, int N) {
//    if (N == Integer.MAX_VALUE) {
//      return;
//    }
//    List<Set<Integer>> indicesToKeep = new ArrayList<Set<Integer>>();
//    for (int i = 0; i < potentials.length; ++i) {
//      Counter<Integer> counts = new Counter<Integer>();
//      for (int j = 0; j < potentials[i].length; ++j) {
//        counts.setCount(j, potentials[i][j]);
//      }
//      PriorityQueue<Integer> pq = counts.asPriorityQueue();
//      Set<Integer> set = new HashSet<Integer>();
//      for (int k = 0; k < N && pq.hasNext(); ++k) {
//        set.add(pq.next());
//      }
//      indicesToKeep.add(set);
//    }
//    for (int i = 0; i < potentials.length; ++i) {
//      Set<Integer> set = indicesToKeep.get(i);
//      for (int j = 0; j < potentials[i].length; ++j) {
//        if (!set.contains(j)) {
//          potentials[i][j] = Double.NEGATIVE_INFINITY;
//        }
//      }
//    }
//  }
//
//  private double[][] padMatrix(double[][] potentials) {
//    int m = potentials.length;
//    int n = potentials[0].length;
//    int max = Math.max(m,n);
//    double[][] padded = new double[max][max];
//    for (int i=0; i < max; ++i) {
//      for (int j=0; j < max; ++j) {
//        if (i < m && j < n) {
//          padded[i][j] = i < m && j < n ? potentials[i][j] : Double.NEGATIVE_INFINITY;
//        }
//      }
//    }
//    return padded;
//  }

  private static void negateInPlace(double[][] matchingPotentials) {
    for (double[] row: matchingPotentials) {
      DoubleArrays.scale(row,-1.0);
    }
  }

  private static void padZeros(double[][] matchingPotentials, int m, int n) {
	  for (int i = m; i < matchingPotentials.length; ++i) {
		  for (int j = n; j < matchingPotentials.length; ++j) {
			  matchingPotentials[i][j] = 0.0;
		  }
	  }
  }

//  private static void negateAndShiftInPlace(double[][] matchingPotentials) {
//    for (double[] row: matchingPotentials) {
//      DoubleArrays.scale(row,-1.0);
//    }
//    
//    double min = Double.POSITIVE_INFINITY;
//    for (double[] row: matchingPotentials) {
//    	for (double entry : row) {
//    		if (entry < min) {
//    			min = entry;
//    		}
//    	}
//    }
//    
//    for (int i = 0; i < matchingPotentials.length; ++i) {
//    	for (int j = 0; j < matchingPotentials[i].length; ++j) {
//    		matchingPotentials[i][j] -= min;
//    	}
//    }
//    
//  }
  
  public int[] extractMatching(double[][] matchingPotentials) {
	negateInPlace(matchingPotentials);
    BipartiteMatchings bm = new BipartiteMatchings();
//    if (matchingPotentials.length < 10) {
//    	assert (matchingPotentials.length < 5);
//    }

    return bm.getMaxMatching(matchingPotentials);    
  }

}
