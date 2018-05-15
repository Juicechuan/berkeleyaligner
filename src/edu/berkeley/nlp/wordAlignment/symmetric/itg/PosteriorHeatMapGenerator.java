package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.math.DoubleArrays;

/**
 * User: aria42
 * Date: Feb 6, 2009
 */
public class PosteriorHeatMapGenerator {
  public static void main(String[] args) {
    ITGParser p = new ITGParser(new NaryNormalFormGrammarBuilder().buildGrammar());

    HeatMapImage hmi = new HeatMapImage(true,Gradient.GRADIENT_HEAT);
    int n = 10;
    hmi.pixelSize = 100;
    double[][] alignmentPotentials = new double[n][n];
    double[] nullPots = DoubleArrays.constantArray(0.0, n);
    double[][] alignmentPosteriors = null;

    p.setGrammar(new NaryNormalFormGrammarBuilder().buildGrammar());
    p.setInput(alignmentPotentials,nullPots,nullPots);
    alignmentPosteriors = p.getAlignmentPosteriors();
    System.out.println("nary alignPost:");
    for (double[] row : alignmentPosteriors) {
      System.out.println(DoubleArrays.toString(row));
    }
    hmi.writePNGImage(alignmentPosteriors,"narynormal.png");
    System.out.println();
    
    p.setGrammar(new NormalFormGrammarBuilder().buildGrammar());
    p.setInput(alignmentPotentials,nullPots,nullPots);
    alignmentPosteriors = p.getAlignmentPosteriors();
    System.out.println("normal alignPost:");
    for (double[] row : alignmentPosteriors) {
      System.out.println(DoubleArrays.toString(row));
    }
    hmi.writePNGImage(alignmentPosteriors,"normal.png");
    System.out.println();

    p.setGrammar(new NullNormalFormGrammarBuilder().buildGrammar());
    p.setInput(alignmentPotentials,nullPots,nullPots);
    alignmentPosteriors = p.getAlignmentPosteriors();
    System.out.println("null-normal alignPost:");
    for (double[] row : alignmentPosteriors) {
      System.out.println(DoubleArrays.toString(row));
    }
    hmi.writePNGImage(alignmentPosteriors,"nullnormal.png");
    System.out.println();

    p.setGrammar(new SimpleGrammarBuilder().buildGrammar());
    p.setInput(alignmentPotentials,nullPots,nullPots);
    alignmentPosteriors = p.getAlignmentPosteriors();
    System.out.println("simple alignPost:");
    for (double[] row : alignmentPosteriors) {
      System.out.println(DoubleArrays.toString(row));
    }
    hmi.writePNGImage(alignmentPosteriors,"simple.png");
    System.out.println();

    System.out.println("***** Null-normal test *******");
//
//    // Null-normal test:  4x4 array with exactly 1 derivation
    alignmentPotentials = new double[3][3];
    for (double[] row : alignmentPotentials) {
    	DoubleArrays.initialize(row, Double.NEGATIVE_INFINITY);
    }
//    alignmentPotentials[0][0] = 0.0;
//    alignmentPotentials[1][1] = 0.0; // Double.NEGATIVE_INFINITY;
//    alignmentPotentials[2][2] = 0.0; // Double.NEGATIVE_INFINITY;
//    alignmentPotentials[3][3] = 0.0; // Double.NEGATIVE_INFINITY;
    
    double[] frNullPots = DoubleArrays.constantArray(0.0, 3);
    //frNullPots[1] = Double.NEGATIVE_INFINITY;
    double[] enNullPots = DoubleArrays.constantArray(0.0, 3);
    //enNullPots[1] = Double.NEGATIVE_INFINITY;
    
    p = new ITGParser(new NaryNormalFormGrammarBuilder().buildGrammar());
    p.setInput(alignmentPotentials,frNullPots,enNullPots);
    alignmentPosteriors = p.getAlignmentPosteriors();
    System.out.println("simple alignPost:");
    for (double[] row : alignmentPosteriors) {
      System.out.println(DoubleArrays.toString(row));
    }
    System.out.println("Z: " + Math.exp(p.getLogZ()));
    System.out.println();
    //hmi.writePNGImage(alignmentPosteriors,"simple-nulltest.png");
  }
  
}
