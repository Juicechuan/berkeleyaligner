package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.util.Logger;
import edu.berkeley.nlp.util.Triple;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.Alignment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: aria42
 * Date: Feb 16, 2009
 */
public class ConstrainedITGParser {

  float[][][][][][] iScores;
  float[][][][][][] oScores;
  boolean[][][][] filter;
  double[][] alignPots;
  double[] frNullPots, enNullPots;
  double[][][] frBlockPots, enBlockPots;
  boolean[] frSures, enSures;
  boolean[] frPosbls, enPosbls;
  boolean[][][] frBlocks, enBlocks;
  boolean[][] sures, posbls;
  boolean hiddenPossibles;
  int alignState, frNullState, enNullState;
  int[][][][] numSures;
  Alignment align;

  int[][][] minStart, minStop, maxStart, maxStop;

  double[][] alignPosts;
  double[] frNullPosts, enNullPosts;
  double[][][] frBlockPosts, enBlockPosts;

  private boolean doneInside, doneOutside, donePosteriors;

  private int maxBlockSize;
  private double logZ = Double.NEGATIVE_INFINITY;
  private int finalP;

  public void setHiddenPossibles(boolean hiddenPossibles) {
    this.hiddenPossibles = hiddenPossibles;
  }

  public void setFilter(boolean[][][][] filter) {
    this.filter = filter;
    buildExtents();
  }

  private void buildExtents() {
    minStart = new int[m + 1][][];
    minStop = new int[m + 1][][];
    maxStop = new int[m + 1][][];
    maxStart = new int[m + 1][][];
    for (int i = 0; i <= m; i++) {
      minStart[i] = new int[m + 1][numStates];
      for (int[] row : minStart[i]) {
        Arrays.fill(row, n + 1);
      }
      minStop[i] = new int[m + 1][numStates];
      for (int[] row : minStop[i]) {
        Arrays.fill(row, n + 1);
      }
      maxStop[i] = new int[m + 1][numStates];
      for (int[] row : maxStop[i]) {
        Arrays.fill(row, 0);
      }
      maxStart[i] = new int[m + 1][numStates];
      for (int[] row : maxStart[i]) {
        Arrays.fill(row, 0);
      }
    }
  }


  @Opt
  public void setITGBlockSize(int maxBlockSize) {
    this.maxBlockSize = maxBlockSize;
  }

  public int getNumDroppedAlignment() {
    return finalP;
  }

  int maxPenalty = 6;
  int m, n, numStates;
  Grammar grammar;
  double[] scratch;
  int scratchIndex;

  private void clearComputation() {
    scratchIndex = 0;
  }

  private void updateComputation(double val) {
    scratch[scratchIndex++] = val;
  }

  private double doComputation() {
    return SloppyMath.logAdd(scratch, scratchIndex);
  }

  private double combineComputation(double x, double y) {
    return SloppyMath.logAdd(x, y);
  }

  public ConstrainedITGParser(Grammar grammar) {
    this.grammar = grammar;
    this.numStates = grammar.states.size();
    this.alignState = grammar.alignTerm.index;
    this.frNullState = grammar.frNullTerm.index;
    this.enNullState = grammar.enNullTerm.index;
  }

  private boolean prune(int i, int j, int s, int t) {
    if (filter == null) return false;
    if (i == 0 || j == 0 || s == 0 || t == 0) return false;
    return filter[i - 1][j - 1][s - 1][t - 1];
  }

  private float[][][][][] createSingleScores(int p) {
    float[][][][][] scores = new float[m + 1][][][][];
      for (int i = 0; i <= m; i++) {
        scores[i] = new float[m + 1][][][];
        for (int j = i; j <= m; j++) {
          scores[i][j] = new float[n + 1][][];
          for (int s = 0; s <= n; s++) {
            scores[i][j][s] = new float[n + 1][];
            for (int t = s; t <= n; t++) {
              if (prune(i, j, s, t)) continue;
              scores[i][j][s][t] = new float[numStates];
              Arrays.fill(scores[i][j][s][t], Float.NEGATIVE_INFINITY);
            }
        }
      }
    }

    return scores;

  }

  private float[][][][][][] createScores() {
    float[][][][][][] scores = new float[maxPenalty][m + 1][][][][];
    for (int p = 0; p < maxPenalty; p++) {
      for (int i = 0; i <= m; i++) {
        scores[p][i] = new float[m + 1][][][];
        for (int j = i; j <= m; j++) {
          scores[p][i][j] = new float[n + 1][][];
          for (int s = 0; s <= n; s++) {
            scores[p][i][j][s] = new float[n + 1][];
            for (int t = s; t <= n; t++) {
              if (prune(i, j, s, t)) continue;
              scores[p][i][j][s][t] = new float[numStates];
              Arrays.fill(scores[p][i][j][s][t], Float.NEGATIVE_INFINITY);              
            }
          }
        }
      }
    }

    return scores;
  }

  public void setInput(Alignment alignment,
                       double[][] alignPots,
                       double[] frNullPots,
                       double[] enNullPots,
                       double[][][] frBlockPots,
                       double[][][] enBlockPots) {
    setup(alignment, alignPots, frNullPots, enNullPots, frBlockPots, enBlockPots);
    doneInside = doneOutside = donePosteriors = false;
  }

  private void setup(Alignment alignment,
                     double[][] alignPots,
                     double[] frNullPots,
                     double[] enNullPots,
                     double[][][] frBlockPots,
                     double[][][] enBlockPots) {
    this.align = alignment;
    this.m = alignPots.length + 1;
    this.n = alignPots[0].length + 1;
    this.alignPots = alignPots;
    this.frNullPots = frNullPots;
    this.enNullPots = enNullPots;
    this.frBlockPots = frBlockPots;
    this.enBlockPots = enBlockPots;
    this.frBlocks = new boolean[m - 1][n - 1][maxBlockSize + 1];
    this.enBlocks = new boolean[m - 1][n - 1][maxBlockSize + 1];
    this.sures = new boolean[m - 1][n - 1];
    this.posbls = new boolean[m - 1][n - 1];
    this.scratch = new double[m * n * numStates * maxPenalty * maxPenalty];
    this.frPosbls = new boolean[m - 1];
    this.enPosbls = new boolean[n - 1];
    this.frSures = new boolean[m - 1];
    this.enSures = new boolean[n - 1];

    for (Pair<Integer, Integer> pair : alignment.getSureAlignments()) {
      int i = pair.getSecond();
      int j = pair.getFirst();
      this.frSures[i] = true;
      this.enSures[j] = true;
      sures[i][j] = true;
    }
    for (Pair<Integer, Integer> pair : alignment.getPossibleAlignments()) {
      int i = pair.getSecond();
      int j = pair.getFirst();
      this.frPosbls[i] = true;
      this.enPosbls[j] = true;
      posbls[i][j] = true;
    }

    edu.berkeley.nlp.fig.basic.Pair<List<Triple<Integer, Integer, Integer>>, List<Triple<Integer, Integer, Integer>>> blockPairs =
        Utils.findBlocks(alignment, maxBlockSize, hiddenPossibles);
    List<Triple<Integer, Integer, Integer>> frBlockTriples = blockPairs.getFirst();
    List<Triple<Integer, Integer, Integer>> enBlockTriples = blockPairs.getSecond();
    for (Triple<Integer, Integer, Integer> triple : frBlockTriples) {
      int i = triple.getFirst();
      int j = triple.getSecond();
      int k = triple.getThird();
      if (k <= maxBlockSize) frBlocks[i][j][k] = true;
    }
    for (Triple<Integer, Integer, Integer> triple : enBlockTriples) {
      int i = triple.getFirst();
      int j = triple.getSecond();
      int k = triple.getThird();
      if (k <= maxBlockSize) enBlocks[i][j][k] = true;
    }
    numSures = Utils.getNumSures(alignment, true);
  }

  private void doInsideUnaryPass(int i, int j, int s, int t, State state, int p) {
    clearComputation();
    UnaryRule[] unarys = grammar.ug.unariesByParent[state.index];
    if (unarys.length == 0) return;
    for (int ruleIndex = 0; ruleIndex < unarys.length; ++ruleIndex) {
      UnaryRule rule = unarys[ruleIndex];
      double childScore = iScores[p][i][j][s][t][rule.child.index];
      updateComputation(childScore);
    }
    double unaryComputation = doComputation();
    double binaryComputation = iScores[p][i][j][s][t][state.index];
    double score = combineComputation(unaryComputation, binaryComputation);
    iSet(i, j, s, t, state.index, p, score);
    //iScores[i][j][s][t][state.index][p] = (float) score;
  }

  void iSet(int i, int j, int s, int t, int state, int p, double val) {
    iScores[p][i][j][s][t][state] = (float) val;
    fastMin(minStart, i, j, state, s);
    fastMax(maxStart, i, j, state, s);
    fastMin(minStop, i, j, state, t);
    fastMax(maxStop, i, j, state, t);
  }

  private static void fastMin(int[][][] arr, int i, int j, int s, int x) {
    if (x < arr[i][j][s]) arr[i][j][s] = x;
  }

  private static void fastMax(int[][][] arr, int i, int j, int s, int x) {
    if (x > arr[i][j][s]) arr[i][j][s] = x;
  }


  private static int fastMax(int x, int y) {
    return x > y ? x : y;
  }

  private static int fastMin(int x, int y) {
    return x < y ? x : y;
  }

  void initInsidePass() {
    iSet(0, 1, 0, 1, alignState, 0, 0.0f);
    //iScores[0][1][0][1][alignState][0] = 0.0f;
    for (State state : grammar.ug.bottomUpOrdering) {
      doInsideUnaryPass(0, 1, 0, 1, state, 0);
    }
    // Alignment Pots
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        if (prune(i, i + 1, j, j + 1)) continue;

        if (sures[i - 1][j - 1] || (hiddenPossibles && posbls[i - 1][j - 1])) {
          iSet(i, i + 1, j, j + 1, alignState, 0, alignPots[i - 1][j - 1]);
          //iScores[i][i + 1][j][j + 1][alignState][0] = (float) alignPots[i - 1][j - 1];
          for (State state : grammar.ug.bottomUpOrdering) {
            doInsideUnaryPass(i, i + 1, j, j + 1, state, 0);
          }
        }
        // Blocks
        for (int k = 2; k <= maxBlockSize; ++k) {
          if (frBlocks[i - 1][j - 1][k]) {
            if (prune(i, i + k, j, j + 1)) continue;
            iSet(i, i + k, j, j + 1, alignState, 0, frBlockPots[i - 1][j - 1][k]);
            //iScores[i][i + k][j][j + 1][alignState][0] = (float) frBlockPots[i - 1][j - 1][k];
          }
          if (enBlocks[i - 1][j - 1][k]) {
            if (prune(i, i + 1, j, j + k)) continue;
            iSet(i, i + 1, j, j + k, alignState, 0, enBlockPots[i - 1][j - 1][k]);
            //iScores[i][i + 1][j][j + k][alignState][0] = (float) enBlockPots[i - 1][j - 1][k];
          }
        }
      }
    }

    // Nulls
    for (int i = 1; i < m; i++) {
      //int penalty = frSures[i - 1] ? 1 : 0;
      float nullPot = (float) frNullPots[i - 1];
      for (int j = 0; j <= n; j++) {
        iSet(i, i + 1, j, j, frNullState, 0, nullPot);
        //iScores[i][i + 1][j][j][frNullState][0] = nullPot;
        for (State state : grammar.ug.bottomUpOrdering) {
          doInsideUnaryPass(i, i + 1, j, j, state, 0);
        }
      }
    }
    for (int j = 1; j < n; j++) {
      //int penalty = enSures[j - 1] ? 1 : 0;
      float nullPot = (float) enNullPots[j - 1];
      for (int i = 0; i <= m; i++) {
        iSet(i, i, j, j + 1, enNullState, 0, nullPot);
        //iScores[i][i][j][j + 1][enNullState][0] = nullPot;
        for (State state : grammar.ug.bottomUpOrdering) {
          doInsideUnaryPass(i, i, j, j + 1, state, 0);
        }
      }
    }
  }

  private double getIScore(int i, int j, int s, int t, int state, int penalty) {
    return prune(i, j, s, t) ? Double.NEGATIVE_INFINITY : iScores[penalty][i][j][s][t][state];
  }

  private double getOScore(int i, int j, int s, int t, int state, int penalty) {
    return prune(i, j, s, t) ? Double.NEGATIVE_INFINITY : oScores[penalty][i][j][s][t][state];
  }

  private void doInsideBinaryPass(int i, int j, int s, int t, State state, int p) {
    clearComputation();
    BinaryRule[] binarys = grammar.bg.binaryRulesByParent[state.index];
    if (binarys.length == 0) {
      return;
    }
    for (int frBreak = i; frBreak <= j; ++frBreak) {
      for (int enBreak = s; enBreak <= t; ++enBreak) {

        if (state.isNormal) {
          if (prune(i, frBreak, s, enBreak) || prune(frBreak, j, enBreak, t)) {
            continue;
          }
          int numSureMissing = numSures[i][frBreak][enBreak][t] +
              numSures[frBreak][j][s][enBreak];
          if (numSureMissing > p) continue;
          int penaltyLeft = p - numSureMissing;

          // Normal No Inversion
          for (int r = 0; r < binarys.length; ++r) {
            BinaryRule rule = binarys[r];
            for (int lp = 0; lp <= penaltyLeft; ++lp) {
              int rp = penaltyLeft - lp;
              double lScore = getIScore(i, frBreak, s, enBreak, rule.lchild.index, lp);
              double rScore = getIScore(frBreak, j, enBreak, t, rule.rchild.index, rp);
              updateComputation(lScore + rScore);
            }
          }
        } else {
          if (prune(i, frBreak, enBreak, t) || prune(frBreak, j, s, enBreak)) {
            continue;
          }
          int numSureMissing = numSures[i][frBreak][s][enBreak] +
              numSures[frBreak][j][enBreak][t];
          if (numSureMissing > p) continue;
          int penaltyLeft = p - numSureMissing;
          // Invert
          for (int r = 0; r < binarys.length; ++r) {
            BinaryRule rule = binarys[r];
            for (int lp = 0; lp <= penaltyLeft; ++lp) {
              int rp = penaltyLeft - lp;
              double lScore = getIScore(i, frBreak, enBreak, t, rule.lchild.index, lp);
              double rScore = getIScore(frBreak, j, s, enBreak, rule.rchild.index, rp);
              updateComputation(lScore + rScore);
            }
          }
        }
      }
    }
    float score = (float) doComputation();
    iSet(i, j, s, t, state.index, p, score);
    //iScores[i][j][s][t][state.index][p] = score;
  }

  private class Computation {
    double[] scratch;
    int scratchIndex = 0;

    Computation() {
      this.scratch = new double[m * n * numStates];
    }

    void clear() {
      scratchIndex = 0;
    }

    void updateComputation(double val) {
      scratch[scratchIndex++] = val;
    }

    double doComputation() {
      return SloppyMath.logAdd(scratch, scratchIndex);
    }
  }


  void doNewInsidePass() {
    if (doneInside) return;
    this.iScores = createScores();
    initInsidePass();

    Computation[] comps = new Computation[n + 1];
    for (int i = 0; i <= n; i++) {
      comps[i] = new Computation();
    }

    for (int sumLen = grammar.minSumLen; sumLen <= m + n; ++sumLen) {
      for (int frLen = 0; frLen <= sumLen; ++frLen) {
        int enLen = sumLen - frLen;
        for (int i = 0; i + frLen <= m; ++i) {
          int j = i + frLen;
          for (State state : grammar.ug.bottomUpOrdering) {
            if (state.isTerminal) continue;
            if (state == grammar.root && !(frLen == m && enLen == n)) continue;
            BinaryRule[] brs = grammar.bg.binaryRulesByParent[state.index];
            if (brs.length == 0) continue;
            for (int p = 0; p < maxPenalty; p++) {
              for (Computation comp : comps) {
                comp.clear();
              }
              for (int k = i; k <= j; ++k) {
                for (BinaryRule br : brs) {
                  if (state.isNormal) {
                    int minS = minStart[i][k][br.lchild.index];
                    int maxS = maxStart[i][k][br.lchild.index];
                    for (int s = minS; s <= maxS; s++) {
                      if (!iff(i == 0, s == 0)) continue;
                      int t = s + enLen;
                      if (t > n || prune(i, j, s, t)) continue;
                      int minU = fastMax(minStop[i][k][br.lchild.index], minStart[k][j][br.rchild.index]);
                      int maxU = fastMin(maxStop[i][k][br.lchild.index], maxStart[k][j][br.rchild.index]);
                      minU = fastMax(s, minU);//(s > minU) ? s : minU;
                      maxU = fastMin(t, maxU);//(t < maxU) ? t : maxU;
                      for (int u = minU; u <= maxU; u++) {
                        if (prune(i, k, s, u) || prune(k, j, u, t)) continue;
                        int numSureMissing = numSures[i][k][u][t] + numSures[k][j][s][u];
                        if (numSureMissing > p) continue;
                        int penaltyLeft = p - numSureMissing;
                        for (int lp = 0; lp <= penaltyLeft; lp++) {
                          int rp = penaltyLeft - lp;
                          double lScore = iScores[lp][i][k][s][u][br.lchild.index];
                          double rScore = iScores[rp][k][j][u][t][br.rchild.index];
                          comps[s].updateComputation(lScore + rScore);
                        }
                      }
                    }
                  } else {
                    int minS = minStart[j][k][br.lchild.index];
                    int maxS = maxStart[j][k][br.lchild.index];
                    for (int s = minS; s <= maxS; s++) {
                      if (!iff(i == 0, s == 0)) continue;
                      int t = s + enLen;
                      if (t > n || prune(i, j, s, t)) continue;
                      int minU = (minStop[i][k][br.lchild.index] > minStart[k][j][br.rchild.index]) ?
                          minStop[i][k][br.lchild.index] : minStart[k][j][br.rchild.index];
                      int maxU = (maxStop[i][k][br.lchild.index] < maxStart[k][j][br.rchild.index]) ?
                          maxStop[i][k][br.lchild.index] : maxStart[k][j][br.rchild.index];
                      minU = (s > minU) ? s : minU;
                      maxU = (t < maxU) ? t : maxU;

                      for (int u = minU; u <= maxU; u++) {
                        if (prune(i, k, u, t) || prune(k, j, s, u)) continue;
                        int numSureMissing = numSures[i][k][s][u] + numSures[k][j][u][t];
                        if (numSureMissing > p) continue;
                        int penaltyLeft = p - numSureMissing;
                        for (int lp = 0; lp <= penaltyLeft; lp++) {
                          int rp = penaltyLeft - lp;
                          double lScore = iScores[lp][i][k][u][t][br.lchild.index];
                          double rScore = iScores[rp][k][j][s][u][br.rchild.index];
                          comps[s].updateComputation(lScore + rScore);
                        }
                      }
                    }
                  }
                }
              }
              for (int s = 0; s + enLen <= n; s++) {
                int t = s + enLen;
                if (prune(i, j, s, t)) continue;
                iSet(i, j, s, t, state.index, p, comps[s].doComputation());
                doInsideUnaryPass(i, j, s, t, state, p);
              }
            }
          }
        }
      }
    }
    finalP = -1;
    for (int p = 0; p < maxPenalty; p++) {
      logZ = iScores[p][0][m][0][n][grammar.root.index];
      if (logZ > Double.NEGATIVE_INFINITY) {
        this.finalP = p;
        break;
      }
    }
    if (finalP < 0) {
      logZ = Double.NEGATIVE_INFINITY;
      Logger.logss("Can't find ITG parse with " +
          maxPenalty + " penalties for alignment\n" +
          align.toString());
    }
    doneInside = true;
  }

  void doInsidePass() {
//    if (true) {
//      doNewInsidePass();
//      return;
//    }
    if (doneInside) return;
    finalP = -1;
    this.iScores = new float[maxPenalty][][][][][];
    iScores[0] = createSingleScores(0);
    initInsidePass();
    for (int p = 0; p < maxPenalty; p++) {
      if (p > 0) iScores[p] = createSingleScores(p);
      for (int sumLen = 3; sumLen <= m + n; ++sumLen) {
        for (int frLen = 0; frLen <= sumLen; ++frLen) {
          int enLen = sumLen - frLen;
          if (frLen == 0 && enLen == 0) continue;
          for (int i = 0; i + frLen <= m; ++i) {
            int j = i + frLen;
            for (int s = 0; s + enLen <= n; ++s) {  // start en
              if (!iff(i == 0, s == 0)) continue;
              int t = s + enLen;
              if (prune(i, j, s, t) || numSures[i][j][s][t] < p) {
                continue;
              }
              for (State state : grammar.ug.bottomUpOrdering) {
                if (state.isTerminal) continue;
                if (state == grammar.root && !(frLen == m && enLen == n)) continue;
                doInsideBinaryPass(i, j, s, t, state, p);
                doInsideUnaryPass(i, j, s, t, state, p);
              }
            }
          }
        }
      }
      logZ = iScores[p][0][m][0][n][grammar.root.index];
      if (logZ > Double.NEGATIVE_INFINITY) {
        this.finalP = p;
        break;
      }
    }
    if (finalP < 0) {
      logZ = Double.NEGATIVE_INFINITY;
      Logger.logss("Can't find ITG parse with " +
          maxPenalty + " penalties for alignment\n" +
          align.toString());
    }
    doneInside = true;
  }

  private void doOutsideUnaryPass(int i, int j, int s, int t, State state, int p) {
    clearComputation();
    UnaryRule[] unaryRules = grammar.ug.unariesByChild[state.index];
    for (int r = 0; r < unaryRules.length; ++r) {
      UnaryRule ur = unaryRules[r];
      double result = oScores[p][i][j][s][t][ur.parent.index];
      updateComputation(result);
    }
    float score = (float) doComputation();
    if (score > Float.NEGATIVE_INFINITY) oScores[p][i][j][s][t][state.index] = score;
  }

  private void doOutsideBinaryPass(int i, int j, int s, int t, State state, int p) {
    clearComputation();
    // Left Span of fr (i,j)
    BinaryRule[] rightBinaryRules = grammar.bg.binaryRulesByRightChild[state.index];
    if (rightBinaryRules.length > 0) {
      // Upper Left
      // Normal (k,i),(l,s) + (i,j), (s,t)
      for (int k = 0; k <= i; ++k) {
        for (int l = 0; l <= s; ++l) {
          if (i - k == 0 && s - l == 0) continue;
          if (prune(k, i, l, s) || prune(k, j, l, t)) continue;
          int internalError = numSures[i][j][l][s] + numSures[k][i][s][t];
          if (internalError > p) continue;
          int penaltyLeft = p - internalError;
          for (int ruleIndex = 0; ruleIndex < rightBinaryRules.length; ruleIndex++) {
            BinaryRule br = rightBinaryRules[ruleIndex];
            if (!br.isNormal()) continue;
            for (int ip = 0; ip <= penaltyLeft; ++ip) {
              int op = penaltyLeft - ip;
              double ulInside = getIScore(k, i, l, s, br.lchild.index, ip);
              double outside = getOScore(k, j, l, t, br.parent.index, op);
              updateComputation(ulInside + outside);
            }
          }
        }
      }
      // Lower Left
      // Inverse (k,i),(t,l) + (i,j),(s,t)
      for (int k = 0; k <= i; ++k) {
        for (int l = t; l <= n; ++l) {
          if (i - k == 0 && l - t == 0) continue;
          if (prune(k, i, t, l) || prune(k, j, s, l)) continue;
          int internalError = numSures[k][i][s][t] + numSures[i][j][t][l];
          if (internalError > p) continue;
          int penaltyLeft = p - internalError;
          for (int ruleIndex = 0; ruleIndex < rightBinaryRules.length; ruleIndex++) {
            BinaryRule br = rightBinaryRules[ruleIndex];
            if (br.isNormal()) continue;
            for (int ip = 0; ip <= penaltyLeft; ++ip) {
              int op = penaltyLeft - ip;
              double llInside = getIScore(k, i, t, l, br.lchild.index, ip);
              double outside = getOScore(k, j, s, l, br.parent.index, op);
              updateComputation(llInside + outside);
            }
          }
        }
      }
    }
    BinaryRule[] leftBinaryRules = grammar.bg.binaryRulesByLeftChild[state.index];
    if (leftBinaryRules.length > 0) {
      // Lower Right
      // Normal (i,j),(s,t) + (j,k),(t,l)
      for (int k = j; k <= m; ++k) {
        for (int l = t; l <= n; ++l) {
          if (k - j == 0 && l - t == 0) continue;
          if (prune(j, k, t, l) || prune(i, k, s, l)) continue;
          int internalError = numSures[j][k][s][t] + numSures[i][j][t][l];
          if (internalError > p) continue;
          int penaltyLeft = p - internalError;
          for (int ruleIndex = 0; ruleIndex < leftBinaryRules.length; ruleIndex++) {
            BinaryRule br = leftBinaryRules[ruleIndex];
            if (!br.isNormal()) continue;
            for (int ip = 0; ip <= penaltyLeft; ++ip) {
              int op = penaltyLeft - ip;
              double lrInside = getIScore(j, k, t, l, br.rchild.index, ip);
              double outside = getOScore(i, k, s, l, br.parent.index, op);
              updateComputation(lrInside + outside);
            }
          }
        }
      }
      // Upper Right
      // Inverted (i,j),(s,t) + (j,k),(l,s)
      for (int k = j; k <= m; ++k) {
        for (int l = 0; l <= s; ++l) {
          if (j - k == 0 && s - l == 0) continue;
          if (prune(j, k, l, s) || prune(i, k, l, t)) continue;
          int internalError = numSures[i][j][l][s] + numSures[j][k][s][t];
          if (internalError > p) continue;
          int penaltyLeft = p - internalError;
          for (int ruleIndex = 0; ruleIndex < leftBinaryRules.length; ruleIndex++) {
            BinaryRule br = leftBinaryRules[ruleIndex];
            if (br.isNormal()) continue;
            for (int ip = 0; ip <= penaltyLeft; ++ip) {
              int op = penaltyLeft - ip;
              double urInside = getIScore(j, k, l, s, br.rchild.index, ip);
              double outside = getOScore(i, k, l, t, br.parent.index, op);
              updateComputation(urInside + outside);
            }
          }
        }
      }
    }
    double unaryResult = oScores[p][i][j][s][t][state.index];
    double binaryResult = doComputation();
    //oScores[i][j][s][t][state.index] = combineComputations(unaryResult, binaryResult);
    float oScore = (float) combineComputation(unaryResult, binaryResult);
    if (oScore > Double.NEGATIVE_INFINITY) {
      oScores[p][i][j][s][t][state.index] = oScore;
    }
  }

  private void doOutsidePass() {
    if (doneOutside) return;
    doInsidePass();
    this.oScores = new float[finalP+1][][][][][];
    for (int p = 0; p <= finalP; p++) {
      oScores[p] = createSingleScores(p);
    }
    oScores[0][0][m][0][n][grammar.root.index] = 0.0f;
    for (State state : grammar.ug.topDownOrdering) {
      if (state != grammar.root) {
        doOutsideUnaryPass(0, m, 0, n, state, 0);
      }
    }
    for (int sumLen = m + n - 1; sumLen > 0; --sumLen) {
      for (int frLen = 0; frLen <= sumLen; ++frLen) {
        int enLen = sumLen - frLen;
        for (int i = 0; i + enLen <= m; ++i) {
          int j = i + enLen;
          for (int s = 0; s + frLen <= n; ++s) {
            int t = s + frLen;
            if (prune(i, j, s, t)) {
              continue;
            }
            for (State state : grammar.ug.topDownOrdering) {
              for (int p = 0; p <= finalP; p++) {
                if (iScores[finalP - p][i][j][s][t][state.index] > Double.NEGATIVE_INFINITY) {
                  doOutsideUnaryPass(i, j, s, t, state, p);
                  doOutsideBinaryPass(i, j, s, t, state, p);
                }
              }
            }
          }
        }
      }
    }
    doneOutside = true;
  }

  private boolean isProb(double p) {
    double eps = 0.01;
    return !Double.isNaN(p) && p >= -eps && p <= 1 + eps;
  }

  private void computePosteriors() {
    if (donePosteriors) return;
    doInsidePass();
    doOutsidePass();
    alignPosts = new double[m - 1][n - 1];
    frBlockPosts = new double[m - 1][n - 1][maxBlockSize + 1];
    enBlockPosts = new double[m - 1][n - 1][maxBlockSize + 1];
    frNullPosts = new double[m - 1];
    enNullPosts = new double[n - 1];
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        if (prune(i, i + 1, j, j + 1)) {
          alignPosts[i - 1][j - 1] = 0.0;
        } else {
          double iScore = iScores[0][i][i + 1][j][j + 1][alignState];
          double oScore = oScores[finalP][i][i + 1][j][j + 1][alignState];
          alignPosts[i - 1][j - 1] = Math.exp(iScore + oScore - logZ);
        }
        for (int k = 2; k <= maxBlockSize; k++) {
          if (i + k <= m) {
            if (prune(i, i + k, j, j + 1)) {
              frBlockPosts[i - 1][j - 1][k] = 0.0;
            } else {
              float biScore = iScores[0][i][i + k][j][j + 1][alignState];
              float boScore = oScores[finalP][i][i + k][j][j + 1][alignState];
              frBlockPosts[i - 1][j - 1][k] = Math.exp(biScore + boScore - logZ);
            }
          }
          if (j + k <= n) {
            if (prune(i, i + 1, j, j + k)) {
              enBlockPosts[i - 1][j - 1][k] = 0.0;
            } else {
              float biScore = iScores[0][i][i + 1][j][j + k][alignState];
              float boScore = oScores[finalP][i][i + 1][j][j + k][alignState];
              enBlockPosts[i - 1][j - 1][k] = Math.exp(biScore + boScore - logZ);
            }
          }
        }
      }
    }
    frNullPosts = new double[m - 1];
    for (int i = 1; i < m; i++) {
      double post = 0.0;
      for (int j = 0; j <= n; ++j) {
        double iScore = iScores[0][i][i + 1][j][j][frNullState];
        double oScore = oScores[finalP][i][i + 1][j][j][frNullState];
        post += Math.exp(iScore + oScore - logZ);
      }
      frNullPosts[i - 1] = post;
      //assert isProb(frNullPosts[i-1]);
    }
    enNullPosts = new double[n - 1];
    for (int j = 1; j < n; j++) {
      double post = 0.0;
      for (int i = 0; i <= m; i++) {
        double iScore = iScores[0][i][i][j][j + 1][enNullState];
        double oScore = oScores[finalP][i][i][j][j + 1][enNullState];
        post += Math.exp(iScore + oScore - logZ);
      }
      enNullPosts[j - 1] = post;
      //assert isProb(enNullPosts[j-1]);
    }
    donePosteriors = true;
  }

  private static boolean iff(boolean p, boolean q) {
    if (p && !q) return false;
    if (q && !p) return false;
    return true;
  }

  public double getLogZ() {
    doInsidePass();
    return logZ;
  }

  public double[][] getAlignmentPosteriors() {
    computePosteriors();
    return alignPosts;
  }

  public double[] getFrNullPosteriors() {
    computePosteriors();
    return frNullPosts;
  }

  public double[] getEnNullPosteriors() {
    computePosteriors();
    return enNullPosts;
  }

  public double[][][] getFrBlockPosteriors() {
    computePosteriors();
    return frBlockPosts;
  }

  public double[][][] getEnBlockPosteriors() {
    computePosteriors();
    return enBlockPosts;
  }

  public static void main(String[] args) {
    double[][] alignPots = new double[3][1];
    double[] frNullPots = new double[3];
    double[] enNullPots = new double[1];
    double[][][] frBlockPots = new double[3][1][10];
    double[][][] enBlockPots = new double[3][1][10];
    List<String> frList = new ArrayList<String>();
    frList.add("x1");
    frList.add("x2");
    frList.add("x3");
    List<String> enList = new ArrayList<String>();
    enList.add("y1");
    Alignment align = new Alignment(enList, frList);
    align.addAlignment(0, 0, true);
    align.addAlignment(0, 1, false);
    align.addAlignment(0, 2, true);
    ConstrainedITGParser p = new ConstrainedITGParser(new NormalFormGrammarBuilder().buildGrammar());
    p.hiddenPossibles = false;
    p.maxBlockSize = 3;
    p.maxPenalty = 3;
    p.setInput(align, alignPots, frNullPots, enNullPots, frBlockPots, enBlockPots);
    double logZ = p.getLogZ();
    System.out.println("Z: " + Math.exp(logZ));
    double[][] alignPosts = p.getAlignmentPosteriors();
    double[][][] frBlockPosts = p.getFrBlockPosteriors();
    double[][][] enBlockPosts = p.getEnBlockPosteriors();
  }

  public void setMaxPenalty(int penalty) {
    this.maxPenalty = penalty;
  }

  public AlignmentScores getPosteriors() {
    computePosteriors();
    AlignmentScores alignScores = new AlignmentScores(m - 1, n - 1, maxBlockSize);
    alignScores.alignScores = DoubleArrays.clone(alignPosts);
    alignScores.frNullScores = DoubleArrays.clone(frNullPosts);
    alignScores.enNullScores = DoubleArrays.clone(enNullPosts);
    alignScores.frBlockPots = DoubleArrays.clone(frBlockPosts);
    alignScores.enBlockPots = DoubleArrays.clone(enBlockPosts);
    return alignScores;
  }
}
