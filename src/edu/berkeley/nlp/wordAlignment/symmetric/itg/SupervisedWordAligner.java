package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.fig.basic.IOUtils;
import edu.berkeley.nlp.fig.basic.Interner;
import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.math.*;
import edu.berkeley.nlp.util.*;
import edu.berkeley.nlp.util.functional.Function;
import edu.berkeley.nlp.util.functional.FunctionalUtils;
import edu.berkeley.nlp.util.optionparser.Experiment;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wordAlignment.WordAligner;
import edu.berkeley.nlp.wordAlignment.symmetric.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 15, 2008
 */
public class SupervisedWordAligner extends WordAligner {

  Interner<Feature> featInterner = featInterner = new Interner<Feature>(new Interner.CanonicalFactory<Feature>() {
      @Override
	public Feature build(Feature object) {
        Feature canonical = new Feature(object.feature);
        canonical.index = featInterner.size();
        return canonical;
      }
    });

  List<Feature> featIndexer;
  SentencePairFeatureExtractor featExtractor;
  double[] weights;

  List<SentencePairFeatures> trainData;
  List<SentencePairFeatures> testData;

  Grammar grammar;
  Options opts = new Options();
  CallbackFunction evalCallback;
  boolean inTraining = false;
  Map<Integer, SentencePairFeatures> testSPFIntern = new ConcurrentHashMap();
  ConcurrentMap<Integer,boolean[][][][]> idToFilter = new ConcurrentHashMap();

  private double[] matchingWeights;

  public void dumpCached() {
    trainData = null;
    testData = null;
    testSPFIntern.clear();
    idToFilter.clear();
    System.gc();
  }

  public enum InferenceMode {
    VITERBI_ITG, POSTERIOR_ITG, MAXMATCHING
  }

  public enum TrainObjectiveMode {
    LOGLIKE, MIRA, NONE
  }

  public class Options {

    @Opt(gloss = "variance of the Gaussian prior on weights (for LOGLIKE) [1.0]")
    public double sigmaSquared = 1.0;

    @Opt(gloss = "How aggressive are we for Chiang (EMNLP 2008)-style MIRA updates? [0.5]")
    public double miraModelGoldMixtureWeight = 0.5;

    @Opt(gloss = "How many threads are used? [# of available processors]")
    public int numThreads = Runtime.getRuntime().availableProcessors();

    @Opt(gloss = "Inference mode: VITERBI_ITG, POSTERIOR_ITG, MAXMATCHING", required=true)
    public InferenceMode inferenceMode = InferenceMode.VITERBI_ITG;

    @Opt(gloss = "Run for at least this many iterations [0]")
    public int minIters = 0;

    @Opt(gloss = "Run for at most this many iterations [0]")
    public int maxIters = 25;

    @Opt(gloss = "Objective mode: LOGLIKE, MIRA, NONE (weights loaded from file) [LOGLIKE]", required = true)
    public TrainObjectiveMode objMode = TrainObjectiveMode.LOGLIKE;

    @Opt(gloss = "MIRA slack parameter [0.1]")
    public double maxTau = 0.1;

    @Opt(gloss = "How many ITG violations will we allow per sentence before we throw it out? (only valid for log-likelihood) [4]")
    public int allowedITGViolations = 4;

    @Opt(gloss = "How often should we evaluate the model (on training & testing) [5]")
    public int evalCycle = 5;

    @Opt(gloss = "For MIRA, seed to permute permute the data before we start learning [0]")
    public int randSeed = 0;

    @Opt(gloss = "MIRA loss computed with possibles (as don't-cares)? [true]")
    public boolean miraSeePossible = true;

    @Opt(gloss = "Recall errors get multiplied by this constant [1]")
    public double miraRecallBias = 1.0;

    @Opt(gloss = "Log-likelihood treats possibles as latent [true]")
    public boolean doHiddenPossible = true;

    @Opt(gloss = "What is k for (k x 1) and (1 x k) blocks? [1]")
    public int itgBlockSize = 1;

    @Opt(gloss = "Size of history for approximating the Hessian in quai-Newton [5]")
    public int lBFGSMaxHistorySize = 5;

    @Opt(gloss = "Dump history before lBFGS says converged [true]")
    public boolean dumpHistBeforConverge = true;

    @Opt(gloss = "Threshold for posterior decoding at test time [0.5]")
    public double posteriorTestThresh = 0.5;

    @Opt(gloss = "From where do we load weights? [null]")
    public String loadWeightsFile = null;

    @Opt(gloss = "Where do we write weights? [null]")
    public String writeWeightsFile = null;
  }

  private Random rand = new Random();

  private boolean inITGMode() {
    return opts.inferenceMode == InferenceMode.VITERBI_ITG ||
        opts.inferenceMode == InferenceMode.POSTERIOR_ITG;
  }

  private void validate() {
    if (opts.objMode == TrainObjectiveMode.LOGLIKE && !inITGMode()) {
      throw new RuntimeException("In LogLike, need ITG inference mode...");
    }
  }

  void initInference() {
    // TODO
  }

  void initTrain(Iterable<SentencePair> trainPairs,
            Iterable<SentencePair> testPairs,
            SentencePairFeatureExtractor featExtractor,
            CallbackFunction evalCallback)
  {
    validate();
    rand = opts.randSeed >= 0 ? new Random(opts.randSeed) : new Random();
    this.featExtractor = featExtractor;
    this.grammar = featExtractor.getGrammar();
    this.featExtractor.featureInterner = featInterner;
    this.evalCallback = evalCallback;

    Iterable<Alignment> trainAligns = FunctionalUtils.map(trainPairs,
        new Function<SentencePair, Alignment>() {
          @Override
		public Alignment apply(SentencePair input) {
            return input.getAlignment();
          }
        });
    trainData = new ArrayList<SentencePairFeatures>();
    extractFeatures(trainPairs, trainAligns, trainData);

    Iterable<Alignment> testAligns = FunctionalUtils.map(testPairs,
        new Function<SentencePair, Alignment>() {
          @Override
		public Alignment apply(SentencePair input) {
            return input.getAlignment();
          }
        });
    testData = new ArrayList<SentencePairFeatures>();
    extractFeatures(testPairs, testAligns, testData);
    featIndexer = new ArrayList(featInterner.getCanonicalElements());
    Collections.sort(featIndexer, new Comparator<Feature>() {
      @Override
	public int compare(Feature feature, Feature feature1) {
        return feature.index-feature1.index;
      }
    });
  }

  private void cacheSPFs() {
    for (SentencePairFeatures spf : trainData) {
      testSPFIntern.put(spf.sp.getSentenceID(),spf);
    }
    for (SentencePairFeatures spf : testData) {
      testSPFIntern.put(spf.sp.getSentenceID(),spf);
    }    
  }

  /**
   * only for infernece
   */
  SupervisedWordAligner(SentencePairFeatureExtractor featExtractor) {
    // assume loadWeights
    GlobalOptionParser.fillOptions(opts);
    if (opts.loadWeightsFile == null) {
      throw new RuntimeException("Inference Only Constructor called with 'loadWeightsFile' option specified");
    }
    List<Pair<String,Double>> featAndWeights = readFeatureAndWeights(opts.loadWeightsFile);
    weights = new double[featAndWeights.size()];
    int weightIndex = 0;
    for (Pair<String,Double> pair: featAndWeights) {
      featInterner.intern(new Feature(pair.getFirst()));
      weights[weightIndex++] = pair.getSecond();
    }
    this.featExtractor = featExtractor;
    this.grammar = featExtractor.getGrammar();
    this.featExtractor.featureInterner = featInterner;
    featIndexer = new ArrayList(featInterner.getCanonicalElements());
    Collections.sort(featIndexer, new Comparator<Feature>() {
      @Override
	public int compare(Feature feature, Feature feature1) {
        return feature.index-feature1.index;
      }
    });
    // featIndexer now doesn't change
    // nor does featInterner nor weights
  }


  SupervisedWordAligner(Iterable<SentencePair> trainPairs,
                        Iterable<SentencePair> testPairs,
                        SentencePairFeatureExtractor featExtractor,
                        CallbackFunction evalCallback) {
    GlobalOptionParser.fillOptions(opts);
    initTrain(trainPairs,testPairs,featExtractor,evalCallback);
    cacheSPFs();
    train();
  }

  private void writeWeights(double[] weights, String outpath) {
    List<String> outLines = new ArrayList();
    for (int i = 0; i < weights.length; i++) {
      double weight = weights[i];
      Feature f = featIndexer.get(i);
      outLines.add(String.format("%s\t%.10f",f.feature,weight));
    }
    IOUtils.writeLinesHard(outpath,outLines);
  }

  private void extractFeatures(Iterable<SentencePair> trainPairs,
                               Iterable<Alignment> aligns, List<SentencePairFeatures> data) {
    Logger.startTrack("Feature extraction and interning");
    Iterable<SentencePairFeatures> trainPairFeats = FunctionalUtils.map(
        trainPairs,
        new Function<SentencePair, SentencePairFeatures>() {
          @Override
		public SentencePairFeatures apply(SentencePair input) {
            SentencePairFeatures spf = featExtractor.getSentencePairFeatures(input, true);
            return spf;
          }
        });

    Iterable<Pair<SentencePairFeatures, Alignment>> trainPairsIterables =
        Iterables.zip(trainPairFeats, aligns);
    for (Pair<SentencePairFeatures, Alignment> pair : trainPairsIterables) {
      SentencePairFeatures spf = pair.getFirst();
      Alignment goldAlignment = pair.getSecond();
      spf.goldAlignment = goldAlignment;
      data.add(spf);
    }

    Logger.endTrack();
  }

  public Counter weightCounter() {
    return smallWeightCounter(weights);
  }

  public Counter weightCounter(double[] weights) {
	    Counter counts = new Counter();
	    Collection<Feature> feats = featInterner.getCanonicalElements();
	    for (Feature feat : feats) {
	      if(weights[feat.index] != 0.0) counts.setCount(feat.feature, weights[feat.index]);
	    }
	    return counts;
  }

  public Counter smallWeightCounter(double[] weights) {
	    Counter negCounts = new Counter();
	    Counter posCounts = new Counter();
	    Collection<Feature> feats = featInterner.getCanonicalElements();
	    for (Feature feat : feats) {
	      if(weights[feat.index] > 0.0) posCounts.setCount(feat.feature, weights[feat.index]);
	      if(weights[feat.index] < 0.0) negCounts.setCount(feat.feature, weights[feat.index]);
	    }
	        
	    negCounts.keepBottomNKeys(15);
	    posCounts.keepTopNKeys(15);
	    
	    Counter counts = new Counter(negCounts);
	    for (Object o : posCounts.keySet()) {
	    	counts.put(o, posCounts.getCount(o), false);
	    	
	    }
	    
	    return counts;
  }

  void train() {
    inTraining = true;
    Logger.startTrack("Training with " + opts.objMode, true);
    weights = null;
    double[] initWeights = opts.loadWeightsFile != null ? readWeights(opts.loadWeightsFile)  : new double[featInterner.size()];
    if (opts.loadWeightsFile != null) {
      setWeights(initWeights);
      if (evalCallback != null) {
        Logger.startTrack("Eval for Read Weights");
        evalCallback.callback(this);
        Logger.endTrack();
      }
    }
    switch (opts.objMode) {
      case NONE:
        setWeights(initWeights);
        break;
      case LOGLIKE:
        doLogLikeTrain(initWeights);
        break;
      case MIRA:
        doMiraTrain(initWeights);
        break;
    }
    inTraining = false;
    Logger.endTrack();
  }

  private void doMiraTrain(double[] initWeights) {
    weights = initWeights;
    new MiraObjectiveFunction().train(true);
    setWeights(weights);
  }

  private void doLogLikeTrain(double[] initWeights) {
    Logger.startTrack("Log-Likelihood Objective");
    Logger.logs("sigmaSquared:" + opts.sigmaSquared);
    Logger.logs("doHiddenPossible: " + opts.doHiddenPossible);
    Logger.endTrack();
    List<ObjectiveItemDifferentiableFunction<SentencePairFeatures>> items = new ArrayList();
    for (int k = 0; k < opts.numThreads; k++) {
      items.add(new LogLikelihoodObjectiveItem());
    }
    DifferentiableFunction objFn = new CachingObjectiveDifferentiableFunction(trainData, items, new L2Regularizer(opts.sigmaSquared));
    LBFGSMinimizer minimizer = getMinimizer(true);
    Logger.startTrack("Begin iteration %d", 0);
    weights = minimizer.minimize(objFn, initWeights, 0.0001, false);
    Logger.endTrack();
    setWeights(weights);
  }
  
  private double[] readWeights(String weightFile) {
    Logger.startTrack("Reading weights from %s",weightFile);
    List<String> lines = IOUtils.readLinesHard(weightFile);
    double[] weights = new double[featInterner.size()];
    Feature queryFeat = new Feature();
    for (String line : lines) {
      String[] pieces = line.split("\t");
      assert pieces.length == 2;
      String featName = pieces[0];
      queryFeat.feature = featName;
      Feature f = featInterner.getCanonical(queryFeat);
      assert f != null && f != queryFeat;
      weights[f.index] = Double.parseDouble(pieces[1]);
    }
    Logger.endTrack();
    return weights;
  }

  private List<Pair<String,Double>> readFeatureAndWeights(String weightFile) {
    Logger.startTrack("Reading features/weights from %s",weightFile);
    List<String> lines = IOUtils.readLinesHard(weightFile);
    List<Pair<String,Double>> result = new ArrayList();
    for (String line : lines) {
      String[] pieces = line.split("\t");
      assert pieces.length == 2;
      String featName = pieces[0];
      //queryFeat.feature = featName;
      //Feature f = featInterner.getCanonical(queryFeat);
      //assert f != null && f != queryFeat;
      //weights[f.index] = Double.parseDouble(pieces[1]);
      result.add(Pair.newPair(featName,Double.parseDouble(pieces[1])));
    }
    Logger.endTrack();
    return result;
  }



  private LBFGSMinimizer getMinimizer(final boolean writeWeights) {
    final LBFGSMinimizer minimizer = new LBFGSMinimizer();
    minimizer.setMaxIterations(opts.maxIters);
    minimizer.setMaxHistorySize(opts.lBFGSMaxHistorySize);
    minimizer.setMinIteratons(opts.minIters);
    minimizer.setDumpHistoryBeforeConverge(opts.dumpHistBeforConverge);
    if (evalCallback != null) {
      minimizer.setIterationCallbackFunction(new CallbackFunction() {
        @Override
		public void callback(Object... args) {
          double[] curGuess = (double[]) args[0];
          int iterDone = (Integer) args[1];
          double val = (Double) args[2];
          double[] deriv = (double[]) args[3];
          setWeights(curGuess);
          Logger.endTrack();
          if (iterDone % opts.evalCycle == 0) {
            evalCallback.callback(SupervisedWordAligner.this);
          }
          if (writeWeights) {
            String outpath = opts.writeWeightsFile != null ?
                opts.writeWeightsFile : IOUtil.getPath(Experiment.getResultDir(), "weights.txt");                 
            writeWeights(curGuess, outpath);
            Logger.logs("Wrote weight file %s",outpath);
          }
          Logger.startTrack("Begin iter %d with val %.5f and grad len %.5f",iterDone + 1, val, DoubleArrays.vectorLength(deriv));
        }
      });
    }
    return minimizer;
  }

  void setWeights(double[] x) {
    this.weights = DoubleArrays.clone(x);
  }

  private <T> List<T> shuffle(List<T> lst) {
    List<T> shuffled = new ArrayList<T>();
    shuffled.addAll(lst);
    Collections.shuffle(shuffled, rand);
    return shuffled;
  }

  class MiraObjectiveFunction {

    private int iter;

    void train(boolean verbose) {
      if (verbose) Logger.startTrack("training mira with %d workers", opts.numThreads);
      weights = new double[dimension()];
      double[] sumWeights = new double[dimension()];
      for (iter = 0; iter < opts.maxIters; ++iter) {
        if (verbose) Logger.startTrack("Iter: " + iter);
        weights = doIter(sumWeights, weights);

        if (verbose) Logger.logss("Weights: " + weightCounter());
        if (verbose && evalCallback != null) {
          evalCallback.callback(SupervisedWordAligner.this);
        }
        if (opts.inferenceMode == InferenceMode.VITERBI_ITG) {
          double[] avgWeights = DoubleArrays.clone(sumWeights);
          DoubleArrays.scale(avgWeights, 1.0 / (opts.maxIters * trainData.size()));
        }
        if (verbose) Logger.endTrack();
      }
      if (verbose) Logger.endTrack();
      DoubleArrays.scale(sumWeights, 1.0 / (opts.maxIters * trainData.size()));
      weights = DoubleArrays.clone(sumWeights);
      if ( (opts.writeWeightsFile != null) && (opts.inferenceMode != SupervisedWordAligner.InferenceMode.MAXMATCHING) ) { 
    	  Logger.logss("Writing weights file to %s",opts.writeWeightsFile);
    	  writeWeights(weights, opts.writeWeightsFile);
    	  weights = readWeights(opts.writeWeightsFile);	  
      }
      
    }

    private double[] doIter(double[] sumWeights, double[] weights) {
      List<SentencePairFeatures> randomData = shuffle(trainData);
      for (SentencePairFeatures spf : randomData) {
        weights = update(spf, weights);
        DoubleArrays.addInPlace(sumWeights, weights);
      }
      return DoubleArrays.clone(weights);
    }

    double[] features(Alignment alignment, SentencePairFeatures spf) {
      Utils.AlignmentDecomposition alignDecomp = Utils.decomposeAlignment(alignment, opts.itgBlockSize);
      double[] feats = new double[dimension()];
      for (Pair<Integer, Integer> sureAlign : alignDecomp.isolatedAlignments) {
        int i = sureAlign.getFirst();
        int j = sureAlign.getSecond();
        for (FeatureValuePair fvp : spf.alignmentFeats[i][j]) {
          feats[fvp.feat.index] += fvp.value;
        }
      }
      for (int i : alignDecomp.frNulls) {
        for (FeatureValuePair fvp : spf.frNullFeats[i]) {
          feats[fvp.feat.index] += fvp.value;
        }
      }
      for (int i : alignDecomp.enNulls) {
        for (FeatureValuePair fvp : spf.enNullFeats[i]) {
          feats[fvp.feat.index] += fvp.value;
        }
      }
      if (opts.inferenceMode == InferenceMode.MAXMATCHING) {
        for (Triple<Integer, Integer, Integer> block : alignDecomp.frBlocks) {
          int i = block.getFirst();
          int j = block.getSecond();
          int size = block.getThird();
          for (int k = 0; k < size; k++) {
            for (FeatureValuePair fvp : spf.alignmentFeats[i + k][j]) {
              feats[fvp.feat.index] += fvp.value;
            }
          }
        }
        for (Triple<Integer, Integer, Integer> block : alignDecomp.enBlocks) {
          int i = block.getFirst();
          int j = block.getSecond();
          int size = block.getThird();
          for (int k = 0; k < size; k++) {
            for (FeatureValuePair fvp : spf.alignmentFeats[i][j + k]) {
              feats[fvp.feat.index] += fvp.value;
            }
          }
        }
        return feats;
      } else {
        for (Triple<Integer, Integer, Integer> block : alignDecomp.frBlocks) {
          int i = block.getFirst();
          int j = block.getSecond();
          int size = block.getThird();
          for (FeatureValuePair fvp : spf.frBlockFeats[i][j][size]) {
            feats[fvp.feat.index] += fvp.value;
          }
        }
        for (Triple<Integer, Integer, Integer> block : alignDecomp.enBlocks) {
          int i = block.getFirst();
          int j = block.getSecond();
          int size = block.getThird();
          for (FeatureValuePair fvp : spf.enBlockFeats[i][j][size]) {
            feats[fvp.feat.index] += fvp.value;
          }
        }
        return feats;
      }
    }

    double loss(SentencePair sentencePair, Alignment referenceAlignment, Alignment proposedAlignment) {
      int proposedSureCount = 0;
      int proposedPossibleCount = 0;
      int proposedCount = 0;
      int sureCount = 0;
      for (int frenchPosition = 0; frenchPosition < sentencePair.getForeignWords().size(); frenchPosition++) {
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          boolean proposed = proposedAlignment.containsSureAlignment(englishPosition, frenchPosition);
          boolean sure = referenceAlignment.containsSureAlignment(englishPosition, frenchPosition);
          boolean possible = referenceAlignment.containsPossibleAlignment(englishPosition, frenchPosition);
          if (proposed && sure) proposedSureCount += 1;
          if (proposed && possible) proposedPossibleCount += 1;
          if (proposed) proposedCount += 1;
          if (sure) sureCount += 1;
        }
      }
      int numRecallErrors = sureCount - proposedSureCount;
      int numPrecErrors = opts.miraSeePossible ?
          proposedCount - proposedPossibleCount :
          proposedCount - proposedSureCount;
      return opts.miraRecallBias * numRecallErrors + numPrecErrors;
    }

    double[] update(SentencePairFeatures spf, double[] weights) {
      Alignment pseudoGoldAlignment = getPseudoGoldAlignment(spf, weights);
      Alignment guessAlignment = getGuessAlignment(spf, weights);
      spf.guessedAlignments.add(guessAlignment);
      ensureNoDuplicates(spf);
      assert (spf.guessedAlignments.size() <= 1);

      double[] goldFeatVec = features(pseudoGoldAlignment, spf);
      double[][] guessFeatVec = new double[spf.guessedAlignments.size()][];
      double[][] diffFeatVec = new double[spf.guessedAlignments.size()][];

      int i = 0;
      double bestScore = Double.NEGATIVE_INFINITY;

      double[] losses = new double[spf.guessedAlignments.size()];
      for (Alignment guessedAlignment : spf.guessedAlignments) {
        guessFeatVec[i] = features(guessedAlignment, spf);
        diffFeatVec[i] = DoubleArrays.subtract(goldFeatVec, guessFeatVec[i]);
        Counter goldCounter = weightCounter(goldFeatVec);
        Counter guessCounter = weightCounter(guessFeatVec[i]);
        Counter diffCOunter = weightCounter(diffFeatVec[i]);
        // losses[i] = loss(spf.sp, spf.goldAlignment, guessedAlignment);
        losses[i] = loss(spf.sp, pseudoGoldAlignment, guessedAlignment);
        double score = DoubleArrays.innerProduct(weights, guessFeatVec[i]);
        if (score > bestScore) {
          bestScore = score;
        }
        ++i;
      }
      double[] residualLosses = new double[spf.guessedAlignments.size()];
      for (int j = 0; j < losses.length; ++j) {
        double weightDotFeatDiff = DoubleArrays.innerProduct(weights, diffFeatVec[j]);
        residualLosses[j] = losses[j] - weightDotFeatDiff;
      }
      double[] alphas = QPSolver.hildreth(diffFeatVec, residualLosses, opts.maxTau);
      for (int j = 0; j < alphas.length; ++j) {
        weights = DoubleArrays.addMultiples(weights, 1.0, diffFeatVec[j], alphas[j]);
      }
      return DoubleArrays.clone(weights);
    }

    private Alignment getGuessAlignment(SentencePairFeatures spf, double[] weights) {
      AlignmentScores pots = new AlignmentScores(spf.m, spf.n, opts.itgBlockSize);
      pots.addModelScores(weights, spf, 1.0);
      pots.addLoss(spf.goldAlignment, coef());
      return alignSentencePair(spf, opts.inferenceMode, pots);
    }

    private double coef() {
      return (iter * opts.miraModelGoldMixtureWeight) / (opts.maxIters);
    }

    private Alignment getPseudoGoldAlignment(SentencePairFeatures spf, double[] weights) {
      AlignmentScores pots = new AlignmentScores(spf.m, spf.n, opts.itgBlockSize);
      pots.addModelScores(weights, spf, coef());
      pots.addLoss(spf.goldAlignment, -1.0);
      return alignSentencePair(spf, opts.inferenceMode, pots);
    }

    private void ensureNoDuplicates(SentencePairFeatures spf) {
      for (Alignment a1 : spf.guessedAlignments) {
        int dupCount = 0;
        for (Alignment a2 : spf.guessedAlignments) {
          if (a1.equals(a2)) dupCount++;
        }
        assert (dupCount == 1);
      }
    }

    int dimension() {
      return featInterner.size();
    }
  }

  class LogLikelihoodObjectiveItem implements ObjectiveItemDifferentiableFunction<SentencePairFeatures> {

    class EmpiricalModule {

      ConstrainedITGParser parser;

      EmpiricalModule() {
        parser = new ConstrainedITGParser(grammar);
        parser.setITGBlockSize(opts.itgBlockSize);
        parser.setHiddenPossibles(opts.doHiddenPossible);
      }

      double update(SentencePairFeatures spf,
                    Alignment alignment,
                    AlignmentScores pots,
                    double[] grad) {
        parser.setMaxPenalty(spf.cachedPenalty > 0 ? spf.cachedPenalty + 1 : opts.allowedITGViolations);
        parser.setInput(alignment, pots.alignScores, pots.frNullScores, pots.enNullScores, pots.frBlockPots, pots.enBlockPots);
        parser.setFilter(spf.filter);
        double logZ = parser.getLogZ();
        if (logZ == Double.NEGATIVE_INFINITY) {
          spf.cachedSkip = true;
          return Double.NEGATIVE_INFINITY;
        }
        if (spf.cachedPenalty < 0) spf.cachedPenalty = parser.getNumDroppedAlignment() + 1;
        AlignmentScores posteriors = parser.getPosteriors();
        posteriors.updateGradient(grad, spf, 1.0);
        return logZ;
      }
    }

    class ExpectedModule {

      ITGParser parser;

      ExpectedModule() {
        parser = new ITGParser(grammar);
        parser.setITGBlockSize(opts.itgBlockSize);
      }

      double update(SentencePairFeatures spf,
                    AlignmentScores pots,
                    double[] grad) {
        parser.setScores(pots);
        parser.setFilter(spf.filter);
        AlignmentScores posteriors = parser.getPosteriors(false);
        posteriors.updateGradient(grad, spf, -1.0);
        return parser.getLogZ();
      }
    }

    EmpiricalModule empModule = new EmpiricalModule();
    ExpectedModule expModule = new ExpectedModule();

    @Override
	public void setWeights(double[] weights) {
      SupervisedWordAligner.this.setWeights(weights);
    }

    @Override
	public double update(SentencePairFeatures spf, double[] grad) {
      double logLike = 0.0;
      double[] localGrad = new double[dimension()];
      if (spf.cachedSkip) return 0.0;
      AlignmentScores pots = new AlignmentScores(spf.m, spf.n, opts.itgBlockSize);
      pots.addModelScores(weights, spf, 1.0);
      double logNumer = empModule.update(spf, spf.goldAlignment, pots, localGrad);
      // Skipping
      if (logNumer == Double.NEGATIVE_INFINITY) {
        return 0.0;
      }
      double logDenom = expModule.update(spf, pots, localGrad);
      assert !Double.isNaN(logNumer) && !Double.isInfinite(logNumer) : "logNumer: " + logNumer;
      assert !Double.isNaN(logDenom) && !Double.isInfinite(logDenom) : "logDenom: " + logDenom;

      assert logNumer <= (logDenom + logDenom*0.01) : String.format("LN: %3.3f LD: %3.3f", logNumer, logDenom);
      logLike += (logNumer - logDenom);
      DoubleArrays.addInPlace(grad, localGrad, -1.0);
      return -logLike;
    }

    @Override
	public int dimension() {
      return featInterner.size();
    }

  }

  private double[][] getPaddedMatchingMatrix(double[][] alignPotentials,
                                             double[] frNullPotentials,
                                             double[] enNullPotentials) {
    int m = alignPotentials.length;
    int n = alignPotentials[0].length;
    double[][] paddedPotentials = new double[m + n][m + n];
    for (double[] row : paddedPotentials) Arrays.fill(row, Double.NEGATIVE_INFINITY);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        paddedPotentials[i][j] = alignPotentials[i][j];
      }
    }

    // Fr Nulls
    for (int i = 0; i < m; ++i) {
      paddedPotentials[i][i + n] = frNullPotentials[i];
    }
    for (int i = 0; i < n; ++i) {
      paddedPotentials[i + m][i] = enNullPotentials[i];
    }

    for (int i = m; i < m + n; ++i) {
      for (int j = n; j < m + n; ++j) {
        paddedPotentials[i][j] = 0.0;
      }
    }

    return paddedPotentials;
  }

  private Alignment alignSentencePair(SentencePairFeatures spf,
                                      InferenceMode infMode,
                                      AlignmentScores alignScores) {
    return alignSentencePair(spf, infMode, alignScores.alignScores,
        alignScores.frNullScores, alignScores.enNullScores, alignScores.frBlockPots,
        alignScores.enBlockPots);
  }

  private Alignment alignSentencePair(SentencePairFeatures spf,
                                      InferenceMode infMode,
                                      double[][] alignPots,
                                      double[] frNullPots,
                                      double[] enNullPots,
                                      double[][][] frBlockPots,
                                      double[][][] enBlockPots) {
    switch (infMode) {
      case MAXMATCHING:
        double[][] paddedPotentials = getPaddedMatchingMatrix(alignPots, frNullPots, enNullPots);
        int[] matching = new BipartiteMatchingExtractor().extractMatching(paddedPotentials);
        return MatchingWordAligner.makeAlignment(spf.sp, matching);
      case VITERBI_ITG:
        ITGParser simpleParser = new ITGParser(new SimpleGrammarBuilder().buildGrammar());
        simpleParser.setMode(ITGParser.Mode.MAX);
        simpleParser.setScores(new AlignmentScores(alignPots, frNullPots, enNullPots, frBlockPots, enBlockPots));
        simpleParser.setFilter(spf.filter);
        boolean[][] alignMatrix = simpleParser.getAlignmentMatrix();
        return MatchingWordAligner.makeAlignment(spf.sp, alignMatrix);
      case POSTERIOR_ITG:
        ITGParser posteriorParser = new ITGParser(grammar);
        posteriorParser.setMode(ITGParser.Mode.SUM);
        
        posteriorParser.setITGBlockSize(opts.itgBlockSize);
    	AlignmentScores pots = new AlignmentScores(spf.m, spf.n, opts.itgBlockSize);
        pots.addModelScores(weights, spf, 1.0);
        posteriorParser.setScores(pots);
        
        posteriorParser.setFilter(spf.filter);
        double[][] posts = posteriorParser.getCollapsedAlignmentPosteriors();
          
        Alignment align = new Alignment(spf.sp, false);

        for (int i = 0; i < spf.m; i++) {
          for (int j = 0; j < spf.n; j++) {
            if (posts[i][j] > opts.posteriorTestThresh) {
              align.addAlignment(j, i, true);
            }
            align.setStrength(j,i,posts[i][j]);
          }
        }
        return align;
    }
    throw new RuntimeException();
  }

  private Alignment alignSentencePair(SentencePairFeatures spf,
                                      InferenceMode inferenceMode) {
    AlignmentScores pots = new AlignmentScores(spf.m, spf.n, opts.itgBlockSize);
    pots.addModelScores(weights, spf, 1.0);
    return alignSentencePair(spf, inferenceMode, pots.alignScores, pots.frNullScores, pots.enNullScores, pots.frBlockPots, pots.enBlockPots);
  }

  @Override
public Alignment alignSentencePair(SentencePair sp) {
    SentencePairFeatures cachedSPF =
        opts.objMode == TrainObjectiveMode.NONE ?
          featExtractor.getSentencePairFeatures(sp, false) :
          testSPFIntern.get(sp.getSentenceID());
    if (cachedSPF == null) {
      cachedSPF = featExtractor.getSentencePairFeatures(sp, false);
    }
    return alignSentencePair(cachedSPF, opts.inferenceMode);
  }

  @Override
  public String getName() {
	return "Supervised";
  }
}
