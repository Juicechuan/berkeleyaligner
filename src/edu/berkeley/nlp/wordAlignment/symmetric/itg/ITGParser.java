package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.util.Logger;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;

import java.util.Arrays;

/**
 * User: aria42
 * Date: Feb 20, 2009
 */
public class ITGParser {

  private float[][][][][] iScores;
  private float[][][][][] oScores;
  private boolean[][][][] filter;
  private double[][] alignmentPots;
  private double logZ;
  private double[] scratch;
  private int scratchIndex;
  private double[] frNullPots, enNullPots;
  private int m, n;
  private Grammar grammar;
  private int numStates;
  private int maxBlockSize = 2;
  private int alignState;
  private int frNullState;
  private int enNullState;
  private double[][] alignmentPosts;
  private double[] frNullPosts, enNullPosts;

  private boolean doneInside = false;
  private boolean doneOutside = false;
  private boolean donePosteriors = false;
  private boolean doneBispanPosteriors = false;
  private double[][] collaspedAlignPosts;
  private double[][][] frBlockPots;
  private double[][][] frBlockPosts;
  private double[][][] enBlockPots;
  private double[][][] enBlockPosts;
  private float[][][][] bispanPosts;

  // Filter Help
  private int[][][] minStart, minStop, maxStart, maxStop;

  public double getLogZ() {
    doInsidePass();
    return logZ;
  }

  public void setBlockPotentials(double[][][] frBlockPots, double[][][] enBlockPots) {
    this.frBlockPots = frBlockPots;
    this.enBlockPots = enBlockPots;
  }

  public void setFilter(boolean[][][][] filter) {
    this.filter = filter;
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

  public void setScores(AlignmentScores scores) {
//    alignmentPots = scores.alignScores;
//    frNullPots = scores.frNullScores;
//    enNullPots = scores.enNullScores;
//    frBlockPots = scores.frBlockPots;
//    enBlockPots = scores.enBlockPots;
    pinchedMode = false;
    setup(scores.alignScores, scores.frNullScores, scores.enNullScores, scores.frBlockPots, scores.enBlockPots);
    doneInside = doneOutside = donePosteriors = false;
  }

  public AlignmentScores getPosteriors(boolean collapseSingleAlignments) {
    computePosteriors();
    AlignmentScores alignScores = new AlignmentScores(m - 1, n - 1, maxBlockSize);
    alignScores.alignScores = DoubleArrays.clone(collapseSingleAlignments ? collaspedAlignPosts : alignmentPosts);
    alignScores.frNullScores = DoubleArrays.clone(frNullPosts);
    alignScores.enNullScores = DoubleArrays.clone(enNullPosts);
    alignScores.frBlockPots = DoubleArrays.clone(frBlockPosts);
    alignScores.enBlockPots = DoubleArrays.clone(enBlockPosts);
    alignScores.logZ = logZ;
    return alignScores;
  }

  public int getMaxBlockSize() {
    return maxBlockSize;
  }

  public static enum Mode {
    SUM, MAX
  }

  private class Computation {
    double max = Double.NEGATIVE_INFINITY;
    double[] scratch;
    int scratchIndex = 0;

    Computation() {
      if (mode == Mode.SUM) {
        this.scratch = new double[m * n * numStates];
      }
    }

    void clear() {
      scratchIndex = 0;
      max = Double.NEGATIVE_INFINITY;
    }

    void updateComputation(double val) {
      switch (mode) {
        case MAX:
          if (val > max) max = val;
          break;
        case SUM:
          scratch[scratchIndex++] = val;
          break;
      }
    }

    double doComputation() {
      switch (mode) {
        case MAX:
          return max;
        case SUM:
          return SloppyMath.logAdd(scratch, scratchIndex);
      }
      throw new IllegalStateException();
    }
  }

  @Opt
  public void setITGBlockSize(int K) {
    this.maxBlockSize = K;
  }

  private Mode mode = Mode.SUM;
  private double max = Double.NEGATIVE_INFINITY;
  private boolean pinchedMode = false;
  public float[][][][][] pIScores;
  public float[][][][][] pOScores;
  private double[][] pinchedPotentials;

  public AlignmentScores doPinchedInsideOutside(final double[][] pinchedPotentials) {
    doInsidePass();
    doOutsidePass();
    assert pinchedPotentials.length + 1 == m;
    assert pinchedPotentials[0].length + 1 == n;
    doneInside = doneOutside = donePosteriors = false;
    this.pinchedMode = true;
    this.pinchedPotentials = pinchedPotentials;
    doInsidePass();
    doOutsidePass();
    computePosteriors();
    pinchedMode = false;
    return getPosteriors(true);
  }

  private void iSet(int i, int j, int s, int t, State state, double val) {
    iSet(i, j, s, t, state.index, val);
  }

  private void iSet(int i, int j, int s, int t, int state, double val) {
    float[][][][][] m = pinchedMode ? pIScores : iScores;
    m[i][j][s][t][state] = (float) val;
    if (!pinchedMode && val > Double.NEGATIVE_INFINITY) {
      fastMin(minStart, i, j, state, s);
      fastMax(maxStart, i, j, state, s);
      fastMin(minStop, i, j, state, t);
      fastMax(maxStop, i, j, state, t);
    }
  }

  private static void fastMin(int[][][] arr, int i, int j, int s, int x) {
    if (x < arr[i][j][s]) arr[i][j][s] = x;
  }

  private static void fastMax(int[][][] arr, int i, int j, int s, int x) {
    if (x > arr[i][j][s]) arr[i][j][s] = x;
  }

  private double iRead(int i, int j, int s, int t, State state) {
    return iRead(i, j, s, t, state.index);
  }

  private double iRead(int i, int j, int s, int t, int state) {
    float[][][][][] m = pinchedMode ? pIScores : iScores;
    return prune(i, j, s, t) ? Double.NEGATIVE_INFINITY : m[i][j][s][t][state];
  }

  private void oSet(int i, int j, int s, int t, State state, double val) {
    oSet(i, j, s, t, state.index, val);
  }

  private void oSet(int i, int j, int s, int t, int state, double val) {
    float[][][][][] m = pinchedMode ? pOScores : oScores;
    m[i][j][s][t][state] = (float) val;
  }

//  private double read(float[][][][][] m, int i, int j, int s, int t, int state) {
//    return prune(i,j,s,t) ? Double.NEGATIVE_INFINITY : m[i][j][s][t][state];
//  }

  private double oRead(int i, int j, int s, int t, State state) {
    float[][][][][] m = pinchedMode ? pOScores : oScores;
    return prune(i, j, s, t) ? Double.NEGATIVE_INFINITY : m[i][j][s][t][state.index];
  }

  private double oRead(int i, int j, int s, int t, int state) {
    float[][][][][] m = pinchedMode ? pOScores : oScores;
    return prune(i, j, s, t) ? Double.NEGATIVE_INFINITY : m[i][j][s][t][state];
  }

  private void clearComputation() {
    switch (mode) {
      case SUM:
        scratchIndex = 0;
        break;
      case MAX:
        max = Double.NEGATIVE_INFINITY;
        break;
    }
  }

  private void updateComputation(double val) {
    switch (mode) {
      case SUM:
        scratch[scratchIndex++] = val;
        break;
      case MAX:
        if (val > max) max = val;
        break;
    }
  }


  private float doComputation() {
    switch (mode) {
      case SUM:
        return (float) SloppyMath.logAdd(scratch, scratchIndex);
      case MAX:
        return (float) max;
    }
    throw new RuntimeException("No mode set");
  }

  private float combineComputations(double x, double y) {
    switch (mode) {
      case SUM:
        return (float) SloppyMath.logAdd(x, y);
      case MAX:
        return (float) ((x > y) ? x : y);
    }
    throw new RuntimeException();
  }

  private float[][][][][] createScores() {
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

  public ITGParser(Grammar grammar) {
    setGrammar(grammar);
    GlobalOptionParser.fillOptions(this);
  }

  public void setGrammar(Grammar grammar) {
    this.grammar = grammar;
    this.numStates = grammar.states.size();
    this.alignState = grammar.alignTerm.index;
    this.frNullState = grammar.frNullTerm.index;
    this.enNullState = grammar.enNullTerm.index;
  }

  public void setInput(double[][] alignmentPots,
                       double[] frNullPots,
                       double[] enNullPots) {
    pinchedMode = false;
    setup(alignmentPots, frNullPots, enNullPots, null, null);
    doneInside = doneOutside = donePosteriors = false;
  }

  public double[][] getAlignmentPosteriors() {
    computePosteriors();
    return alignmentPosts;
  }

  public double[] getFrNullPosteriors() {
    computePosteriors();
    return frNullPosts;
  }

  public double[] getEnNullPosteriors() {
    computePosteriors();
    return enNullPosts;
  }

  private double alignProb(int i, int j) {
    return blockProb(i, j, 1, false);
  }

  private double blockProb(int frStart, int enStart, int blockLen, boolean frBlock) {
    int frStop = frBlock ? frStart + blockLen : frStart + 1;
    int enStop = !frBlock ? enStart + blockLen : enStart + 1;
    double ins = iScores[frStart][frStop][enStart][enStop][alignState];
    double outs = oScores[frStart][frStop][enStart][enStop][alignState];
    if (pinchedMode) {
      double pins = pIScores[frStart][frStop][enStart][enStop][alignState];
      double pouts = pOScores[frStart][frStop][enStart][enStop][alignState];
      double logNumer = combineComputations(ins + pouts, pins + outs);
      return Math.exp(logNumer - logZ);
    } else {
      return Math.exp(ins + outs - logZ);
    }
  }

  private double nullProb(int frStart, int enStart, boolean frNull) {
    int frStop = frNull ? frStart + 1 : frStart;
    int enStop = !frNull ? enStart + 1 : enStart;
    int state = frNull ? frNullState : enNullState;
    double ins = iScores[frStart][frStop][enStart][enStop][state];
    double outs = oScores[frStart][frStop][enStart][enStop][state];
    if (pinchedMode) {
      double pins = pIScores[frStart][frStop][enStart][enStop][state];
      double pouts = pOScores[frStart][frStop][enStart][enStop][state];
      double logNumer = combineComputations(ins + pouts, pins + outs);
      return Math.exp(logNumer - logZ);
    } else {
      return Math.exp(ins + outs - logZ);
    }
  }


  public double[][][] getFrBlockPosteriors() {
    computePosteriors();
    return frBlockPosts;
  }

  public double[][][] getEnBlockPosteriors() {
    computePosteriors();
    return enBlockPosts;
  }

  private double prob(int i, int j, int s, int t) {
    double maxProb = 0.0;
    for (State state : grammar.states) {
      double iScore = iScores[i][j][s][t][state.index];
      double oScore = oScores[i][j][s][t][state.index];
      double curProb;
      if (pinchedMode) {
        double pIScore = pIScores[i][j][s][t][state.index];
        double pOScore = pOScores[i][j][s][t][state.index];
        double logNumer = SloppyMath.logAdd(iScore + pOScore, pIScore + oScore);
        curProb = Math.exp(logNumer - logZ);
      } else {
        curProb = Math.exp(iScore + oScore - logZ);
      }
      maxProb = Math.max(curProb, maxProb);
    }
    return maxProb;
  }

  public float[][][][] getBispanPosteriors() {
    computeBispanPosteriors();
    return bispanPosts;
  }

  private void computeBispanPosteriors() {
    if (doneBispanPosteriors) return;
    doInsidePass();
    doOutsidePass();
    bispanPosts = new float[m][][][];
    for (int i = 1; i <= m; i++) {
      bispanPosts[i - 1] = new float[m][][];
      for (int j = i; j <= m; j++) {
        bispanPosts[i - 1][j - 1] = new float[n][];
        for (int s = 1; s <= n; s++) {
          bispanPosts[i - 1][j - 1][s - 1] = new float[n];
          for (int t = s; t <= n; t++) {
            bispanPosts[i - 1][j - 1][s - 1][t - 1] = (float) prob(i, j, s, t);
          }
        }
      }
    }
    doneBispanPosteriors = true;
  }

  private void computePosteriors() {
    if (donePosteriors) return;
    doInsidePass();
    doOutsidePass();
    alignmentPosts = new double[m - 1][n - 1];
    collaspedAlignPosts = new double[m - 1][n - 1];
    frBlockPosts = new double[m - 1][n - 1][maxBlockSize + 1];
    enBlockPosts = new double[m - 1][n - 1][maxBlockSize + 1];
    for (int i = 1; i < m; ++i) {
      for (int j = 1; j < n; ++j) {
        if (prune(i, i + 1, j, j + 1)) {
          alignmentPosts[i - 1][j - 1] = 0.0;
        } else {
          alignmentPosts[i - 1][j - 1] = alignProb(i, j);
          collaspedAlignPosts[i - 1][j - 1] += alignmentPosts[i - 1][j - 1];
        }
        for (int k = 2; k <= maxBlockSize && i + k <= m; ++k) {
          if (prune(i, i + k, j, j + 1)) {
            frBlockPosts[i - 1][j - 1][k] = 0.0;
          } else {
            frBlockPosts[i - 1][j - 1][k] = blockProb(i, j, k, true);
            for (int k1 = 0; k1 < k; ++k1) collaspedAlignPosts[i - 1 + k1][j - 1] += frBlockPosts[i - 1][j - 1][k];
          }
        }
        for (int k = 2; k <= maxBlockSize && j + k <= n; k++) {
          if (prune(i, i + 1, j, j + k)) {
            enBlockPosts[i - 1][j - 1][k] = 0.0;
          } else {
            enBlockPosts[i - 1][j - 1][k] = blockProb(i, j, k, false);
            for (int k1 = 0; k1 < k; ++k1) collaspedAlignPosts[i - 1][j - 1 + k1] += enBlockPosts[i - 1][j - 1][k];
          }
        }
      }
    }
    frNullPosts = new double[m - 1];
    for (int i = 1; i < m; i++) {
      double post = 0.0;
      for (int j = 0; j <= n; ++j) {
        post += nullProb(i, j, true);
      }
      frNullPosts[i - 1] = post;
      // Way 2
//      double way2 = 0.0;
//      for (int j=1; j < n; ++j) {
//        way2 += collaspedAlignPosts[i-1][j-1];
//      }
//      way2 = 1.0-way2;
//      if (Math.abs(way2-post) > 0.001) {
//        throw new RuntimeException("1: " + way2 + " 2: " + post);
//      }
    }
    enNullPosts = new double[n - 1];
    for (int j = 1; j < n; j++) {
      double post = 0.0;
      for (int i = 0; i <= m; i++) {
        post += nullProb(i, j, false);
      }
      enNullPosts[j - 1] = post;
    }
    donePosteriors = true;
  }

  public double[][] getCollapsedAlignmentPosteriors() {
    computePosteriors();
    return collaspedAlignPosts;
  }

  private void setup(double[][] alignmentPots, double[] frNullPots, double[] enNullPots,
                     double[][][] frBlockPots, double[][][] enBlockPots) {
    this.m = alignmentPots.length + 1;
    this.n = alignmentPots[0].length + 1;
    this.alignmentPots = alignmentPots;
    this.frNullPots = frNullPots;
    this.enNullPots = enNullPots;
    this.scratch = new double[m * n * numStates];
    this.filter = null;
    this.frBlockPots = frBlockPots;
    this.enBlockPots = enBlockPots;

    buildExtents();
  }

  private static boolean iff(boolean p, boolean q) {
    if (p && !q) return false;
    if (q && !p) return false;
    return true;
  }

  private boolean prune(int i, int j, int s, int t) {
    if (filter == null) return false;
    if (i == 0 || j == 0 || s == 0 || t == 0) return false;
    return filter[i - 1][j - 1][s - 1][t - 1];
  }

  private void doInsidePass() {
    if (true) {
      doNewInsidePass();
      return;
    }

    if (doneInside) return;
    if (pinchedMode) {
      this.pIScores = createScores();
    } else {
      this.iScores = createScores();
    }
    initInsidePass();
    for (int sumLen = grammar.minSumLen; sumLen <= m + n; ++sumLen) {
      for (int frLen = 0; frLen <= sumLen; ++frLen) {
        int enLen = sumLen - frLen;
        if (frLen == 0 && enLen == 0) continue;
        for (int i = 0; i + frLen <= m; ++i) {
          int j = i + frLen;
          for (int s = 0; s + enLen <= n; ++s) {  // start en
            if (!iff(i == 0, s == 0)) continue;
            int t = s + enLen;
            if (prune(i, j, s, t)) {
              continue;
            }
            for (State state : grammar.ug.bottomUpOrdering) {
              if (state.isTerminal) continue;
              if (state == grammar.root && !(frLen == m && enLen == n)) continue;
              doInsideBinaryPass(i, j, s, t, state);
              doInsideUnaryPass(i, j, s, t, state);
            }
          }
        }
      }
    }
    logZ = iRead(0, m, 0, n, grammar.root);
    //iScores[0][m][0][n][grammar.root.index];
    doneInside = true;
  }

  private static int min(int x, int y) {
    return x < y ? x : y;
  }

  private static int max(int x, int y) {
    return x > y ? x : y;
  }

  private void doNewInsidePass() {
    if (doneInside) return;
    if (pinchedMode) {
      this.pIScores = createScores();
    } else {
      this.iScores = createScores();
    }
    initInsidePass();
    Computation[] comps = new Computation[n + 1];
    for (int k = 0; k < comps.length; k++) {
      comps[k] = new Computation();
    }
    for (int sumLen = grammar.minSumLen; sumLen <= m + n; ++sumLen) {
      for (int frLen = 0; frLen <= sumLen; frLen++) {
        int enLen = sumLen - frLen;
        // Binary (frLen,enLen)
        for (int i = 0; i + frLen <= m; i++) {
          int j = i + frLen;
          for (State state : grammar.ug.bottomUpOrdering) {
            if (state.isTerminal) continue;
            if (state == grammar.root && sumLen < m + n) continue;
            for (Computation comp : comps) {
              comp.clear();
            }
            for (int k = i; k <= j; k++) {
              // Normal
              if (state.isNormal) {
                BinaryRule[] brs = grammar.bg.binaryRulesByParent[state.index];
                for (BinaryRule br : brs) {
                  int minS = minStart[i][k][br.lchild.index];
                  int maxS = maxStart[i][k][br.lchild.index];
                  for (int s = minS; s <= maxS; s++) {
                    int t = s + enLen;
                    if (t > n || prune(i, j, s, t)) continue;
                    int minU = max(minStop[i][k][br.lchild.index], minStart[k][j][br.rchild.index]);
                    int maxU = min(maxStop[i][k][br.lchild.index], maxStart[k][j][br.rchild.index]);
                    minU = max(s, minU);
                    maxU = min(t, maxU);
                    for (int u = minU; u <= maxU; u++) {
                      if (prune(i, k, s, u) || prune(k, j, u, t)) continue;
                      double lScore = iScores[i][k][s][u][br.lchild.index];
                      double rScore = iScores[k][j][u][t][br.rchild.index];
                      if (pinchedMode) {
                        double pLScore = pIScores[i][k][s][u][br.lchild.index];
                        double pRScore = pIScores[k][j][u][t][br.rchild.index];
                        comps[s].updateComputation(combineComputations(lScore + pRScore, pLScore + rScore));
                      } else {
                        comps[s].updateComputation(lScore + rScore);
                      }
                    }
                  }
                }
              }
              // Invert
              else {
                BinaryRule[] brs = grammar.bg.binaryRulesByParent[state.index];
                for (BinaryRule br : brs) {
                  int minS = minStart[k][j][br.rchild.index];
                  int maxS = maxStart[k][j][br.rchild.index];
                  for (int s = minS; s <= maxS; s++) {
                    int t = s + enLen;
                    if (t > n || prune(i, j, s, t)) continue;
                    int minU = max(minStart[i][k][br.lchild.index], minStop[k][j][br.rchild.index]);
                    int maxU = min(maxStart[i][k][br.lchild.index], maxStop[k][j][br.rchild.index]);
                    minU = max(s, minU);
                    maxU = min(t, maxU);
                    for (int u = minU; u <= maxU; u++) {
                      if (prune(i, k, u, t) || prune(k, j, s, u)) continue;
                      double lScore = iScores[i][k][u][t][br.lchild.index];
                      double rScore = iScores[k][j][s][u][br.rchild.index];
                      if (pinchedMode) {
                        double pLScore = pIScores[i][k][u][t][br.lchild.index];
                        double pRScore = pIScores[k][j][s][u][br.rchild.index];
                        comps[s].updateComputation(combineComputations(lScore + pRScore, pLScore + rScore));
                      } else {
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
              iSet(i, j, s, t, state, comps[s].doComputation());
              doInsideUnaryPass(i, j, s, t, state);
            }
          }
        }
        // Unary (frLen,enLen)
      }
    }
    logZ = iRead(0, m, 0, n, grammar.root);
    doneInside = true;
  }

//  static Counter<Pair<Integer,Integer>> timeBySpans = new Counter();
//  static int parseCount = 0;

  private void doInsideUnaryPass(int i, int j, int s, int t, State state) {
    clearComputation();
    UnaryRule[] unarys = grammar.ug.unariesByParent[state.index];
    if (unarys.length == 0) return;
    for (int ruleIndex = 0; ruleIndex < unarys.length; ++ruleIndex) {
      UnaryRule rule = unarys[ruleIndex];
      double childScore = iRead(i, j, s, t, rule.child);//iScores[i][j][s][t][childIndex];
      updateComputation(childScore);
    }
    double unaryComputation = doComputation();
    double binaryComputation = iRead(i, j, s, t, state);//iScores[i][j][s][t][state.index];    
    iSet(i, j, s, t, state, combineComputations(unaryComputation, binaryComputation));
  }

  private Pair<BinaryRule[], BinaryRule[]> splitRules(BinaryRule[] brs) {
    int kNormal = 0;
    int kInvert = 0;
    for (int i = 0; i < brs.length; i++) {
      BinaryRule br = brs[i];
      if (br.isNormal()) kNormal++;
      else kInvert++;
    }
    int normalIndex = 0;
    int invertIndex = 0;
    BinaryRule[] normals = new BinaryRule[kNormal];
    BinaryRule[] inverts = new BinaryRule[kInvert];
    for (int i = 0; i < brs.length; i++) {
      BinaryRule br = brs[i];
      if (br.isNormal()) normals[normalIndex++] = br;
      else inverts[invertIndex++] = br;
    }
    return Pair.newPair(normals, inverts);
  }

  private void doInsideBinaryPass(int i, int j, int s, int t, State state) {
//    long start = System.currentTimeMillis();
    clearComputation();
    BinaryRule[] binarys = grammar.bg.binaryRulesByParent[state.index];
    if (binarys.length == 0) {
      return;
    }
    if (state.isNormal) {
      for (int frBreak = i; frBreak <= j; ++frBreak) {
        int enBreakStart = s;//minStop != null && s > 0 ? Math.max(s, minStop[i][frBreak]) : s;
        int enBreakStop = t;//maxStart  != null && s > 0 ? Math.min(t, maxStart[frBreak][j]) : t;
        for (int enBreak = enBreakStart; enBreak <= enBreakStop; ++enBreak) {
          if (prune(i, frBreak, s, enBreak) || prune(frBreak, j, enBreak, t)) {
            continue;
          }
          // Normal No Inversion
          for (int r = 0; r < binarys.length; ++r) {
            BinaryRule rule = binarys[r];
            double lScore = iScores[i][frBreak][s][enBreak][rule.lchild.index];
            double rScore = iScores[frBreak][j][enBreak][t][rule.rchild.index];
            if (pinchedMode) {
              double pLScore = pIScores[i][frBreak][s][enBreak][rule.lchild.index];
              double pRScore = pIScores[frBreak][j][enBreak][t][rule.rchild.index];
              updateComputation(combineComputations(lScore + pRScore, pLScore + rScore));
            } else {
              updateComputation(lScore + rScore);
            }
          }
        }
      }
    } else {
      for (int frBreak = i; frBreak <= j; ++frBreak) {
        for (int enBreak = s; enBreak <= t; ++enBreak) {
          if (prune(i, frBreak, enBreak, t) || prune(frBreak, j, s, enBreak)) {
            continue;
          }
          // Invert
          for (int r = 0; r < binarys.length; ++r) {
            BinaryRule rule = binarys[r];
            double lScore = iScores[i][frBreak][enBreak][t][rule.lchild.index];
            double rScore = iScores[frBreak][j][s][enBreak][rule.rchild.index];
            if (pinchedMode) {
              double pLScore = pIScores[i][frBreak][enBreak][t][rule.lchild.index];
              double pRScore = pIScores[frBreak][j][s][enBreak][rule.rchild.index];
              updateComputation(combineComputations(lScore + pRScore, pLScore + rScore));
            } else {
              updateComputation(lScore + rScore);
            }
          }
        }
      }
    }
    iSet(i, j, s, t, state, doComputation());
//    long stop = System.currentTimeMillis();
//    double secs = ((double)(stop-start))/1000.0;
//    timeBySpans.incrementCount(Pair.newPair(j-i,t-s),secs);

  }

  private void initInsidePass() {
    iSet(0, 1, 0, 1, alignState, pinchedMode ? Double.NEGATIVE_INFINITY : 0.0);
    //iScores[0][1][0][1][alignState] = 0.0f;
    for (State state : grammar.ug.bottomUpOrdering) {
      doInsideUnaryPass(0, 1, 0, 1, state);
    }
    // (0,0) is aligned already
    for (int i = 1; i < m; ++i) {
      for (int j = 1; j < n; ++j) {
        double alignPotential = alignmentPots[i - 1][j - 1];
        if (pinchedMode) alignPotential += pinchedPotentials[i - 1][j - 1];
        if (prune(i, i + 1, j, j + 1)) {
          continue;
        }
        iSet(i, i + 1, j, j + 1, alignState, alignPotential);
        //iScores[i][i + 1][j][j + 1][alignState] = (float) alignmentPots[i - 1][j - 1];
        for (State state : grammar.ug.bottomUpOrdering) {
          doInsideUnaryPass(i, i + 1, j, j + 1, state);
        }
        // Blocks
        // Many Fr to 1 En
        if (frBlockPots != null) {
          //assert !pinchedMode;
          for (int k = 2; k <= maxBlockSize && i + k <= m; ++k) {
            double frBlockPot = frBlockPots[i - 1][j - 1][k];
            if (pinchedMode) {
              double pinchPot = 0.0;
              for (int k1 = 0; k1 < k; ++k1) {
                pinchPot = SloppyMath.logAdd(pinchPot, pinchedPotentials[i - 1 + k1][j - 1]);
              }
              frBlockPot += pinchPot;
            }
            if (prune(i, i + k, j, j + 1)) continue;
            iSet(i, i + k, j, j + 1, alignState, frBlockPot);
          }
        }
        // Many En to 1 Fr
        if (enBlockPots != null) {
          //assert !pinchedMode;
          for (int k = 2; k <= maxBlockSize && j + k <= n; ++k) {
            double enBlockPot = enBlockPots[i - 1][j - 1][k];
            if (pinchedMode) {
              double pinchPot = 0.0;
              for (int k1 = 0; k1 < k; ++k1) {
                pinchPot = SloppyMath.logAdd(pinchPot, pinchedPotentials[i - 1][j - 1 + k1]);
              }
              enBlockPot += pinchPot;
            }
            if (prune(i, i + 1, j, j + k)) continue;
            iSet(i, i + 1, j, j + k, alignState, enBlockPot);
          }
        }
      }
    }
    for (int i = 1; i < m; ++i) {
      for (int j = 0; j <= n; ++j) {
        if (prune(i, i + 1, j, j)) continue;
        iSet(i, i + 1, j, j, frNullState, pinchedMode ? Double.NEGATIVE_INFINITY : frNullPots[i - 1]);
        for (State state : grammar.ug.bottomUpOrdering) {
          doInsideUnaryPass(i, i + 1, j, j, state);
        }
      }
    }
    for (int j = 1; j < n; ++j) {
      for (int i = 0; i <= m; ++i) {
        if (prune(i, i, j, j + 1)) continue;
        iSet(i, i, j, j + 1, enNullState, pinchedMode ? Double.NEGATIVE_INFINITY : enNullPots[j - 1]);
        for (State state : grammar.ug.bottomUpOrdering) {
          doInsideUnaryPass(i, i, j, j + 1, state);
        }
      }
    }
  }

  private void doOutsideUnaryPass(int i, int j, int s, int t, State state) {
    clearComputation();
    UnaryRule[] unaryRules = grammar.ug.unariesByChild[state.index];
    for (int r = 0; r < unaryRules.length; ++r) {
      UnaryRule ur = unaryRules[r];
      double result = oRead(i, j, s, t, ur.parent);
      updateComputation(result);
    }

    oSet(i, j, s, t, state, doComputation());
    //oScores[i][j][s][t][state.index] = doComputation();
  }

  private void doOutsideBinaryPass(int i, int j, int s, int t, State state) {
    //if (true) doNewOutsideBinaryPass(i,j,s,t,state);
    doOldOutsideBinaryPass(i, j, s, t, state);
  }

  private void doNewOutsideBinaryPass(int i, int j, int s, int t, State state) {
    clearComputation();
    // Left Span of fr (i,j)
    BinaryRule[] rightBinaryRules = grammar.bg.binaryRulesByRightChild[state.index];
    doRightOutsideBinaryPass(i, j, s, t, rightBinaryRules);

    BinaryRule[] leftBinaryRules = grammar.bg.binaryRulesByLeftChild[state.index];
    doLeftBinaryOutsidePass(i, j, s, t, leftBinaryRules);

    double unaryResult = oRead(i, j, s, t, state);//oScores[i][j][s][t][state.index];
    double binaryResult = doComputation();
    //oScores[i][j][s][t][state.index] = combineComputations(unaryResult, binaryResult);
    float oScore = combineComputations(unaryResult, binaryResult);
    oSet(i, j, s, t, state, oScore);
  }


  private void doLeftBinaryOutsidePass(int i, int j, int s, int t, BinaryRule[] leftBinaryRules) {
    if (leftBinaryRules.length == 0) return;
    Pair<BinaryRule[], BinaryRule[]> splitRules = splitRules(leftBinaryRules);
    BinaryRule[] normalBinarys = splitRules.getFirst();
    BinaryRule[] invertBinarys = splitRules.getSecond();

    // Lower Right
    // Normal (i,j),(s,t) + (j,k),(t,l)
    if (normalBinarys.length > 0) {
      for (int k = j; k <= m; ++k) {
        for (int l = t; l <= n; ++l) {
          if (k - j == 0 && l - t == 0) continue;
          if (prune(j, k, t, l) || prune(i, k, s, l)) continue;
          for (int ruleIndex = 0; ruleIndex < normalBinarys.length; ruleIndex++) {
            BinaryRule br = normalBinarys[ruleIndex];
            //if (!br.isNormal()) continue;
            double lrInside = iScores[j][k][t][l][br.rchild.index];
            double outside = oScores[i][k][s][l][br.parent.index];
            if (pinchedMode) {
              double plrInside = pIScores[j][k][t][l][br.rchild.index];
              double poutside = pOScores[i][k][s][l][br.parent.index];
              float result = combineComputations(lrInside + poutside, plrInside + outside);
              updateComputation(result);
            } else {
              updateComputation(lrInside + outside);
            }
          }
        }
      }
    }
    // Upper Right
    // Inverted (i,j),(s,t) + (j,k),(l,s)
    if (invertBinarys.length > 0) {
      for (int k = j; k <= m; ++k) {
        for (int l = 0; l <= s; ++l) {
          if (j - k == 0 && s - l == 0) continue;
          if (prune(j, k, l, s) || prune(i, k, l, t)) continue;
          for (int ruleIndex = 0; ruleIndex < invertBinarys.length; ruleIndex++) {
            BinaryRule br = invertBinarys[ruleIndex];
            //if (br.isNormal()) continue;
            double urInside = iScores[j][k][l][s][br.rchild.index];
            double outside = oScores[i][k][l][t][br.parent.index];
            if (pinchedMode) {
              double purInside = pIScores[j][k][l][s][br.rchild.index];
              double poutside = pOScores[i][k][l][t][br.parent.index];
              float result = combineComputations(purInside + outside, poutside + urInside);
              updateComputation(result);
            } else {
              updateComputation(urInside + outside);
            }
          }
        }
      }
    }
  }

  private void doOldOutsideBinaryPass(int i, int j, int s, int t, State state) {
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
          for (int ruleIndex = 0; ruleIndex < rightBinaryRules.length; ruleIndex++) {
            BinaryRule br = rightBinaryRules[ruleIndex];
            if (!br.isNormal()) continue;
            double ulInside = iScores[k][i][l][s][br.lchild.index];
            double outside = oScores[k][j][l][t][br.parent.index];
            if (pinchedMode) {
              double pulInside = pIScores[k][i][l][s][br.lchild.index];
              double pOutside = pOScores[k][j][l][t][br.parent.index];
              float result = combineComputations(ulInside + pOutside, pulInside + outside);
              updateComputation(result);
            } else {
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
          for (int ruleIndex = 0; ruleIndex < rightBinaryRules.length; ruleIndex++) {
            BinaryRule br = rightBinaryRules[ruleIndex];
            if (br.isNormal()) continue;
            double llInside = iScores[k][i][t][l][br.lchild.index];
            double outside = oScores[k][j][s][l][br.parent.index];
            if (pinchedMode) {
              double pllInside = pIScores[k][i][t][l][br.lchild.index];
              double pOutside = pOScores[k][j][s][l][br.parent.index];
              float result = combineComputations(llInside + pOutside, pllInside + outside);
              updateComputation(result);
            } else {
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
          for (int ruleIndex = 0; ruleIndex < leftBinaryRules.length; ruleIndex++) {
            BinaryRule br = leftBinaryRules[ruleIndex];
            if (!br.isNormal()) continue;
            double lrInside = iScores[j][k][t][l][br.rchild.index];
            double outside = oScores[i][k][s][l][br.parent.index];
            if (pinchedMode) {
              double plrInside = pIScores[j][k][t][l][br.rchild.index];
              double poutside = pOScores[i][k][s][l][br.parent.index];
              float result = combineComputations(lrInside + poutside, plrInside + outside);
              updateComputation(result);
            } else {
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
          for (int ruleIndex = 0; ruleIndex < leftBinaryRules.length; ruleIndex++) {
            BinaryRule br = leftBinaryRules[ruleIndex];
            if (br.isNormal()) continue;
            double urInside = iScores[j][k][l][s][br.rchild.index];
            double outside = oScores[i][k][l][t][br.parent.index];
            if (pinchedMode) {
              double purInside = pIScores[j][k][l][s][br.rchild.index];
              double poutside = pOScores[i][k][l][t][br.parent.index];
              float result = combineComputations(purInside + outside, poutside + urInside);
              updateComputation(result);
            } else {
              updateComputation(urInside + outside);
            }
          }
        }
      }
    }
    double unaryResult = oRead(i, j, s, t, state);//oScores[i][j][s][t][state.index];
    double binaryResult = doComputation();
    //oScores[i][j][s][t][state.index] = combineComputations(unaryResult, binaryResult);
    float oScore = combineComputations(unaryResult, binaryResult);
    oSet(i, j, s, t, state, oScore);
  }

  private void doRightOutsideBinaryPass(int i, int j, int s, int t, BinaryRule[] rightBinaryRules) {
    if (rightBinaryRules.length == 0) return;
    Pair<BinaryRule[], BinaryRule[]> splitRules = splitRules(rightBinaryRules);
    BinaryRule[] normalBinarys = splitRules.getFirst();
    BinaryRule[] invertBinarys = splitRules.getSecond();
    // Upper Left
    // Normal (k,i),(l,s) + (i,j), (s,t)
    if (normalBinarys.length > 0) {
      for (int k = 0; k <= i; ++k) {
        for (int l = 0; l <= s; ++l) {
          if (i - k == 0 && s - l == 0) continue;
          if (prune(k, i, l, s) || prune(k, j, l, t)) continue;
          for (int ruleIndex = 0; ruleIndex < normalBinarys.length; ruleIndex++) {
            BinaryRule br = normalBinarys[ruleIndex];
            //if (!br.isNormal()) continue;
            double ulInside = iScores[k][i][l][s][br.lchild.index];
            double outside = oScores[k][j][l][t][br.parent.index];
            if (pinchedMode) {
              double pulInside = pIScores[k][i][l][s][br.lchild.index];
              double pOutside = pOScores[k][j][l][t][br.parent.index];
              float result = combineComputations(ulInside + pOutside, pulInside + outside);
              updateComputation(result);
            } else {
              updateComputation(ulInside + outside);
            }
          }
        }
      }
    }
    if (invertBinarys.length > 0) {
      // Lower Left
      // Inverse (k,i),(t,l) + (i,j),(s,t)
      for (int k = 0; k <= i; ++k) {
        for (int l = t; l <= n; ++l) {
          if (i - k == 0 && l - t == 0) continue;
          if (prune(k, i, t, l) || prune(k, j, s, l)) continue;
          for (int ruleIndex = 0; ruleIndex < invertBinarys.length; ruleIndex++) {
            BinaryRule br = invertBinarys[ruleIndex];
            //if (br.isNormal()) continue;
            double llInside = iScores[k][i][t][l][br.lchild.index];
            double outside = oScores[k][j][s][l][br.parent.index];
            if (pinchedMode) {
              double pllInside = pIScores[k][i][t][l][br.lchild.index];
              double pOutside = pOScores[k][j][s][l][br.parent.index];
              float result = combineComputations(llInside + pOutside, pllInside + outside);
              updateComputation(result);
            } else {
              updateComputation(llInside + outside);
            }
          }
        }
      }
    }
  }

  private void doOutsidePass() {
    if (false) {
      doNewOutsidePass();
      return;
    }

    if (doneOutside) return;
    if (pinchedMode) {
      this.pOScores = createScores();
    } else {
      this.oScores = createScores();
    }
    doInsidePass();
    oSet(0, m, 0, n, grammar.root, pinchedMode ? Double.NEGATIVE_INFINITY : 0.0);
    //oScores[0][m][0][n][grammar.root.index] = 0.0f;
    for (State state : grammar.ug.topDownOrdering) {
      if (state != grammar.root) {
        doOutsideUnaryPass(0, m, 0, n, state);
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
              if (iScores[i][j][s][t][state.index] > Double.NEGATIVE_INFINITY) {
                doOutsideUnaryPass(i, j, s, t, state);
                doOutsideBinaryPass(i, j, s, t, state);
              }
            }
          }
        }
      }
    }
    doneOutside = true;
  }

  private void doNewOutsidePass() {
    if (doneOutside) return;
    if (pinchedMode) {
      this.pOScores = createScores();
    } else {
      this.oScores = createScores();
    }
    if (!doneInside) doInsidePass();
    oSet(0, m, 0, n, grammar.root, pinchedMode ? Double.NEGATIVE_INFINITY : 0.0);
    //oScores[0][m][0][n][grammar.root.index] = 0.0f;
    for (State state : grammar.ug.topDownOrdering) {
      if (state != grammar.root) {
        doOutsideUnaryPass(0, m, 0, n, state);
      }
    }

    Computation[] comps = new Computation[n + 1];
    for (int k = 0; k < comps.length; k++) {
      comps[k] = new Computation();
    }

    for (int sumLen = m + n - 1; sumLen > 0; --sumLen) {
      for (int frLen = 0; frLen <= sumLen; ++frLen) {
        int enLen = sumLen - frLen;
        for (int i = 0; i + frLen <= m; ++i) {
          int j = i + frLen;

          for (State state : grammar.ug.topDownOrdering) {
            BinaryRule[] rightBinaryRules = grammar.bg.binaryRulesByRightChild[state.index];
            Pair<BinaryRule[], BinaryRule[]> rightSplitRules = splitRules(rightBinaryRules);
            BinaryRule[] rightNormalBinarys = rightSplitRules.getFirst();
            BinaryRule[] rightInvertBinarys = rightSplitRules.getSecond();

            BinaryRule[] leftBinaryRules = grammar.bg.binaryRulesByLeftChild[state.index];
            Pair<BinaryRule[], BinaryRule[]> leftSplitRules = splitRules(leftBinaryRules);
            BinaryRule[] leftNormalBinarys = leftSplitRules.getFirst();
            BinaryRule[] leftInvertBinarys = leftSplitRules.getSecond();

            // Must do unary pass before binary
            for (int s = 0; s + enLen <= n; ++s) {
              int t = s + enLen;
              if (prune(i, j, s, t)) continue;
              doOutsideUnaryPass(i, j, s, t, state);
            }

            for (Computation comp : comps) {
              comp.clear();
            }

            // Left extend
            for (int k = 0; k <= i; ++k) {
              int minS = minStart[i][j][state.index];
              int maxS = maxStart[i][j][state.index];

              for (int s = minS; s <= maxS; ++s) {
                int t = s + enLen;
                if (t > n || prune(i, j, s, t)) continue;

                // Upper Left
                // Normal (k,i),(l,s) + (i,j), (s,t)
                if (rightNormalBinarys.length > 0) {
                  for (int ruleIndex = 0; ruleIndex < rightNormalBinarys.length; ruleIndex++) {
                    BinaryRule br = rightNormalBinarys[ruleIndex];
                    int minL = max(minStart[k][i][br.lchild.index],minStart[k][j][br.parent.index]);
                    int maxL = min(maxStart[k][i][br.lchild.index],maxStart[k][j][br.parent.index]);
                    maxL = min(maxL,s) ;

                    for (int l = minL; l <= maxL; ++l) {
                      if (prune(k, i, l, s) || prune(k, j, l, t)) continue;
                      double ulInside = iScores[k][i][l][s][br.lchild.index];
                      double outside = oScores[k][j][l][t][br.parent.index];

                      if (pinchedMode) {
                        double pulInside = pIScores[k][i][l][s][br.lchild.index];
                        double pOutside = pOScores[k][j][l][t][br.parent.index];
                        comps[s].updateComputation(combineComputations(ulInside + pOutside, pulInside + outside));
                      } else {
                        comps[s].updateComputation(ulInside + outside);
                      }
                    }
                  }
                }

                // Lower Left
                // Inverse (k,i),(t,l) + (i,j),(s,t)
                if (rightInvertBinarys.length > 0) {
                  for (int ruleIndex = 0; ruleIndex < rightInvertBinarys.length; ruleIndex++) {
                    BinaryRule br = rightInvertBinarys[ruleIndex];

                    int minL = max(minStop[k][i][br.lchild.index],minStop[k][j][br.parent.index]) ;
                    int maxL = min(maxStop[k][i][br.lchild.index],maxStop[k][j][br.parent.index]);
                    minL = max(minL,t) ;

                    for (int l = minL; l <= maxL; ++l) {
                      if (prune(k, i, t, l) || prune(k, j, s, l)) continue;
                      double ulInside = iScores[k][i][t][l][br.lchild.index];
                      double outside = oScores[k][j][s][l][br.parent.index];

                      if (pinchedMode) {
                        double pulInside = pIScores[k][i][t][l][br.lchild.index];
                        double pOutside = pOScores[k][j][s][l][br.parent.index];
                        comps[s].updateComputation(combineComputations(ulInside + pOutside, pulInside + outside));
                      } else {
                        comps[s].updateComputation(ulInside + outside);
                      }
                    }

                  }
                }
              }
            }

            // right extend
            for (int k = j; k <= m; ++k) {
              int minS = minStart[i][j][state.index];
              int maxS = maxStart[i][j][state.index];

              for (int s = minS; s <= maxS; ++s) {
                int t = s + enLen;
                if (t > n || prune(i, j, s, t)) continue;

                // Lower Right
                // Normal (i,j),(s,t) + (j,k),(t,l)
                if (leftNormalBinarys.length > 0) {
                  for (int ruleIndex = 0; ruleIndex < leftNormalBinarys.length; ruleIndex++) {
                    BinaryRule br = leftNormalBinarys[ruleIndex];

                    int minL = max(minStop[j][k][br.rchild.index],minStop[i][k][br.parent.index]);
                    int maxL = min(maxStop[j][k][br.rchild.index],maxStop[i][k][br.parent.index]);
                    minL = max(minL,t);

                    for (int l = minL; l <= maxL; ++l) {
                      if (prune(i, k, s, l) || prune(j, k, t, l)) continue;
                      double ulInside = iScores[j][k][t][l][br.rchild.index];
                      double outside = oScores[i][k][s][l][br.parent.index];

                      if (pinchedMode) {
                        double pulInside = pIScores[j][k][t][l][br.rchild.index];
                        double pOutside = pOScores[i][k][s][l][br.parent.index];
                        comps[s].updateComputation(combineComputations(ulInside + pOutside, pulInside + outside));
                      } else {
                        comps[s].updateComputation(ulInside + outside);
                      }
                    }
                  }
                }

                // Upper Right
                // Inverted (i,j),(s,t) + (j,k),(l,s)
                if (leftInvertBinarys.length > 0) {
                  for (int ruleIndex = 0; ruleIndex < leftInvertBinarys.length; ruleIndex++) {
                    BinaryRule br = leftInvertBinarys[ruleIndex];

                    int minL = max(minStart[j][k][br.rchild.index],minStart[i][k][br.parent.index]);
                    int maxL = min(maxStart[j][k][br.rchild.index],maxStart[i][k][br.parent.index]);
                    maxL = min(maxL, s);

                    for (int l = minL; l <= maxL; ++l) {
                      if (prune(i, k, l, t) || prune(j, k, l, s)) continue;
                      double ulInside = iScores[j][k][l][s][br.rchild.index];
                      double outside = oScores[i][k][l][t][br.parent.index];

                      if (pinchedMode) {
                        double pulInside = pIScores[j][k][l][s][br.rchild.index];
                        double pOutside = pOScores[i][k][l][t][br.parent.index];
                        comps[s].updateComputation(combineComputations(ulInside + pOutside, pulInside + outside));
                      } else {
                        comps[s].updateComputation(ulInside + outside);
                      }
                    }
                  }
                }
              }
            }

            for (int s = 0; s + enLen <= n; s++) {
              int t = s + enLen;
              if (prune(i, j, s, t)) continue;
              double unaryComputation = oRead(i, j, s, t, state);
              oSet(i, j, s, t, state, combineComputations(unaryComputation, comps[s].doComputation()));
            }
          }
        }
      }
    }

    doneOutside = true;
  }

  private void backtrackAlign(boolean[][] matching,
                              int i, int j,
                              int s, int t,
                              State state) {
    double goal = iScores[i][j][s][t][state.index];
    double tol = 0.0001;
    String msg = String.format("[%d,%d,%d,%d,%s] goal: %.5f", i, j, s, t, state.toString(), goal);
    if (goal == Double.NEGATIVE_INFINITY) {
      throw new RuntimeException("No goal: " + msg);
    }
    if (state == grammar.frNullTerm || state == grammar.enNullTerm) {
      return;
    }
    if (state == grammar.alignTerm) {
      if (i == 0) return;
      for (int i1 = i; i1 < j; i1++) {
        for (int j1 = s; j1 < t; j1++) {
          matching[i1 - 1][j1 - 1] = true;
        }
      }
      //assert matching[i] == -1;
      // For the $ root
      //matching[i] = s-1;
      return;
    }
    for (int k = i; k <= j; ++k) { // fr break  k
      for (int l = s; l <= t; ++l) { // en break  l
        BinaryRule[] rules = grammar.bg.binaryRulesByParent[state.index];
        for (int ruleIndex = 0; ruleIndex < rules.length; ++ruleIndex) {
          BinaryRule rule = rules[ruleIndex];
          // Normal
          // (i,k),(s,l) + (k,j),(l,t) = (i,j),(s,t)
          if (state.isNormal) {
            if (prune(i, k, s, l) || prune(k, j, l, t)) continue;
            float left = iScores[i][k][s][l][rule.lchild.index];
            float right = iScores[k][j][l][t][rule.rchild.index];
            float curIScore = left + right;
            if (Math.abs(goal - curIScore) < tol) {
              backtrackAlign(matching, i, k, s, l, rule.lchild);
              backtrackAlign(matching, k, j, l, t, rule.rchild);
              return;
            }
          } else {
            // Invert
            // (i,k),(l,t) + (k,j),(s,l)
            if (prune(i, k, l, t) || prune(k, j, s, l)) continue;
            float left = iScores[i][k][l][t][rule.lchild.index];
            float right = iScores[k][j][s][l][rule.rchild.index];
            float curIScore = left + right;
            if (Math.abs(goal - curIScore) < tol) {
              backtrackAlign(matching, i, k, l, t, rule.lchild);
              backtrackAlign(matching, k, j, s, l, rule.rchild);
              return;
            }
          }
        }
      }
    }
    // Try Unaries
    UnaryRule[] unaries = grammar.ug.unariesByParent[state.index];
    for (int ruleIndex = 0; ruleIndex < unaries.length; ruleIndex++) {
      UnaryRule unary = unaries[ruleIndex];
      float curIScore = iScores[i][j][s][t][unary.child.index];
      if (Math.abs(goal - curIScore) < tol) {
        backtrackAlign(matching, i, j, s, t, unary.child);
        return;
      }
    }
    throw new RuntimeException("Error in " + msg);
  }

  public boolean[][] getAlignmentMatrix() {
    if (mode != Mode.MAX) {
      throw new RuntimeException();
    }
    doInsidePass();
    boolean[][] matching = new boolean[m - 1][n - 1];
    backtrackAlign(matching, 0, m, 0, n, grammar.root);
    return matching;
  }

  public int[] getMatching() {
    doInsidePass();
    boolean[][] matchingMatrix = getAlignmentMatrix();
    int[] matching = new int[matchingMatrix.length];
    for (int i = 0; i < matchingMatrix.length; i++) {
      for (int j = 0; j < matchingMatrix[i].length; j++) {
        if (matchingMatrix[i][j]) {
          matching[i] = j;
        }
      }
    }
    return matching;
  }

  public Mode getMode() {
    return mode;
  }

  public void setMode(Mode mode) {
    this.mode = mode;
  }

  public static void main(String[] args) {
    if (false) {
      int n = 2;
      long start = System.currentTimeMillis();
      double[][] ap = new double[n][n];
      ITGParser p = new ITGParser(new NormalFormGrammarBuilder().buildGrammar());
      p.maxBlockSize = 3;
      p.setMode(ITGParser.Mode.SUM);
      //double[] frNull = {0.0, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
      double[] nullPots = DoubleArrays.constantArray(0.0, n);
      double[][][] blockPots = new double[n][n][p.maxBlockSize + 1];
      p.setInput(ap, nullPots, nullPots);
      p.setBlockPotentials(blockPots, blockPots);
      //    DoubleArrays.constantArray(0, 2));
      //p.getAlignmentMatrix();
      System.out.println("Z: " + Math.exp(p.getLogZ()));
      double[] frNullPosts = p.getFrNullPosteriors();
      System.out.println("nullposts: " + frNullPosts[0] * Math.exp(p.getLogZ()));
      double[][] alignPosts = p.getAlignmentPosteriors();
      double[][][] frBlockPosts = p.getFrBlockPosteriors();
      for (double[] row : alignPosts) {
        System.out.println("row: " + Arrays.toString(row));
      }
      long stop = System.currentTimeMillis();
      System.out.printf("frBlockPosteriors[0][0][2]=%.5f\n", frBlockPosts[0][0][2]);
      double[][] collapsedPosts = p.getCollapsedAlignmentPosteriors();
      System.out.println("time: " + (stop - start));
//      System.out.println("matching: " + Arrays.toString(m[0]));
//      System.out.println("matching: " + Arrays.toString(m[1]));
    }
    if (true) {
      int n = 20;
      double[][] ap = new double[n][n];
      for (double[] row : ap) {
        Arrays.fill(row, Double.NEGATIVE_INFINITY);
      }
      for (int i = 0; i < n; i++) {
        ap[i][i] = 0.0;
      }
      double[] np = new double[n];
      ITGParser p = new ITGParser(new NormalFormGrammarBuilder().buildGrammar());
      p.mode = ITGParser.Mode.SUM;
      p.setInput(ap, np, np);
      Logger.startTrack("Normal Inside Outside");
      System.out.println("UnPinched Z: " + Math.exp(p.getLogZ()));
      Logger.endTrack();
      Logger.startTrack("Pinced Inside Outside");
      p.doPinchedInsideOutside(new double[n][n]);
      System.out.println("Pinched Z: " + Math.exp(p.logZ) + " Theoretical: " + (n * Math.pow(2, n - 1)));
      Logger.endTrack();

      double prob = ((n + 1.0) * Math.pow(2, n - 2)) / (n * Math.pow(2, n - 1));
      System.out.println("theoretical prob: " + prob);
      System.out.println("observed prob: " + p.getAlignmentPosteriors()[0][0]);
    }
  }
}

