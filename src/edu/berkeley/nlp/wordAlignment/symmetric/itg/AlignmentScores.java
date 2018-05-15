package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wordAlignment.symmetric.FeatureValuePair;
import edu.berkeley.nlp.wordAlignment.symmetric.SentencePairFeatures;

import java.util.List;

/**
 * Just a container for arrays to go to parser (potentials)
 * or come back from parser (posteriors)
 */
public class AlignmentScores {

  public int m, n;
  public double logZ;
  public double[][] alignScores;
  public double[] frNullScores;
  public double[] enNullScores;
  public double[][][] frBlockPots;
  public double[][][] enBlockPots;
  public int itgBlockSize;

  private double getActivation(List<FeatureValuePair> fvps, double[] weights) {
    double sum = 0.0;
    for (FeatureValuePair fvp : fvps) {
      sum += weights[fvp.feat.index] * fvp.value;
    }
    return sum;
  }

  public AlignmentScores(int m, int n, int itgBlockSize) {
    this.itgBlockSize = itgBlockSize;
    this.m = m;
    this.n = n;
    this.alignScores = new double[m][n];
    this.frNullScores = new double[m];
    this.enNullScores = new double[n];
    this.frBlockPots = new double[m][n][itgBlockSize + 1];
    this.enBlockPots = new double[m][n][itgBlockSize + 1];
  }

  public AlignmentScores(double[][] alignScores, double[] frNullScores, double[] enNullScores,
                         double[][][] frBlockScores, double[][][] enBlockScores) {
    this.m = alignScores.length;
    this.n = alignScores[0].length;
    this.itgBlockSize = frBlockScores[0][0].length;
    this.alignScores = DoubleArrays.clone(alignScores);
    this.frNullScores = DoubleArrays.clone(frNullScores);
    this.enNullScores = DoubleArrays.clone(enNullScores);
    this.frBlockPots = DoubleArrays.clone(frBlockScores);
    this.enBlockPots = DoubleArrays.clone(enBlockScores);
  }

  public void addModelScores(double[] weights, SentencePairFeatures spf, double c) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        alignScores[i][j] += c * getActivation(spf.alignmentFeats[i][j], weights);
        // Block
        for (int k = 2; k <= itgBlockSize; ++k) {
          if (i + k <= spf.m)
            frBlockPots[i][j][k] += c * getActivation(spf.frBlockFeats[i][j][k], weights);
          if (j + k <= spf.n)
            enBlockPots[i][j][k] += c * getActivation(spf.enBlockFeats[i][j][k], weights);
        }
      }
    }
    for (int i = 0; i < spf.m; i++) {
      frNullScores[i] += c * getActivation(spf.frNullFeats[i], weights);
    }
    for (int j = 0; j < spf.n; j++) {
      enNullScores[j] += c * getActivation(spf.enNullFeats[j], weights);
    }
  }

  public void addLoss(Alignment align, double c) {
    int m = frNullScores.length;
    int n = enNullScores.length;
    double[][] loss = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        boolean sure = align.containsSureAlignment(j, i);
        boolean possible = sure || align.containsPossibleAlignment(j, i);
        // Should Be On
        if (sure) {
          alignScores[i][j] -= c;
          loss[i][j] = -c;
        }
        // Should Be off
        if (!possible) {
          alignScores[i][j] += c;
          loss[i][j] = c;
        }
      }
    }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 2; k <= itgBlockSize; k++) {
          for (int k1 = 0; k1 < k; k1++) {
            if (i + k1 < m) frBlockPots[i][j][k] += loss[i + k1][j];
            if (j + k1 < n) enBlockPots[i][j][k] += loss[i][j + k1];
          }
        }
      }
    }
  }

  public void updateGradient(double[] grad, SentencePairFeatures spf, double c) {
    for (int i = 0; i < spf.m; i++) {
      for (int j = 0; j < spf.n; j++) {
        updateGrad(grad, spf.alignmentFeats[i][j], c* alignScores[i][j]);
        for (int k = 2; k <= itgBlockSize && i + k <= spf.m; k++) {
          updateGrad(grad, spf.frBlockFeats[i][j][k], c*frBlockPots[i][j][k]);
        }
        for (int k = 2; k <= itgBlockSize && j + k <= spf.n; k++) {
          updateGrad(grad, spf.enBlockFeats[i][j][k], c*enBlockPots[i][j][k]);
        }
      }
    }
    for (int i = 0; i < spf.m; i++) {
      updateGrad(grad, spf.frNullFeats[i], c* frNullScores[i]);
    }
    for (int j = 0; j < spf.n; j++) {
      updateGrad(grad, spf.enNullFeats[j], c* enNullScores[j]);
    }
  }

  private void updateGrad(double[] grad, List<FeatureValuePair> fvps, double c) {
    if (Math.abs(c) < 0.00000001) return;
    for (FeatureValuePair fvp : fvps) {
      grad[fvp.feat.index] += fvp.value * c;
    }
  }
}