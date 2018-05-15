package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.fig.basic.IOUtils;
import edu.berkeley.nlp.fig.basic.Pair;
import edu.berkeley.nlp.mapper.AsynchronousMapper;
import edu.berkeley.nlp.mapper.SimpleMapper;
import edu.berkeley.nlp.util.CallbackFunction;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Filter;
import edu.berkeley.nlp.util.Logger;
import edu.berkeley.nlp.util.optionparser.Experiment;
import edu.berkeley.nlp.util.optionparser.GlobalOptionParser;
import edu.berkeley.nlp.util.optionparser.Opt;
import edu.berkeley.nlp.wa.mt.Alignment;
import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wa.mt.SentencePairReader;
import edu.berkeley.nlp.wa.mt.SentencePairReader.PairDepot;
import edu.berkeley.nlp.wordAlignment.WordAligner;
import edu.berkeley.nlp.wordAlignment.symmetric.*;
import edu.berkeley.nlp.wordAlignment.symmetric.features.*;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class ITGMain implements Runnable {

  public static enum ModelType {
    LEARNED,
    MINLOSS_ITG,
    MINLOSS_MAXMATCHING,
  }

  public static enum GrammarType {
    SIMPLE, NULLNORMAL, NARYNORMAL, NORMAL
  }

  public static enum FilterType {
    NULL, EXTERNAL_POSTERIOR
  }


  public class Options {

    @Opt(gloss = "Only running alignment, assume weight file given")
    public boolean testOnly = false;

    @Opt(gloss = "Add bias feature for alignment pair? [true]")
    public boolean addBiasFeatures = true;

    @Opt(gloss = "Add lexical features (top k word pairs)? [false]")
    public boolean addLexicalFeatures = false;

    @Opt(gloss = "Add features for external model (HMM) posteriors? [false]")
    public boolean addStoredPosteriorFeature = false;

    @Opt(gloss = "For each (i,j) pair, ask 'Is this the best i for this j?' [false]")
    public boolean addCompetitivePosteriorFeature = false;

    @Opt(gloss = "Add dictionary features? [false]")
    public boolean addDictionaryFeature = false;

    @Opt(gloss = "Add features to encourage matching punctuation items?")
    public boolean addPuncFeatures = false;

    @Opt(gloss = "Add features for posteriors of adjacent words? [false]")
    public boolean addAdjacentPosteriorFeature = false;

    @Opt(gloss = "Add relatve (i,j) distance features? [true]")
    public boolean addDistFeatures = true;

    @Opt(gloss = "Model Type: LEARNED, MINLOSS_ITG, MINLOSS_MAXMATCHING [LEARNED]")
    public ModelType modelType = ModelType.LEARNED;

    @Opt(gloss = "lowercase [true]")
    public boolean lowercase = true;

    @Opt(gloss = "max # of unlabeled sentences [Integer.MAX_VALUE]")
    public int maxUnlabaeledSentences = Integer.MAX_VALUE;

    @Opt(gloss = "max # of training sentences [Integer.MAX_VALUE]")
    public int maxTrainSentences = Integer.MAX_VALUE;

    @Opt(gloss = "max # of test sentences [Integer.MAX_VALUE]")
    public int maxTestSentences = Integer.MAX_VALUE;

    @Opt(gloss = "foreign suffix [f]")
    public String foreignSuffix = "f";

    @Opt(gloss = "English suffix [e]")
    public String englishSuffix = "e";

    @Opt(gloss = "reverse alignments (E->F) instead of (F->E)? [false]")
    public boolean reverseAlignments = false;

    @Opt(gloss = "one-indexed instead of zero-indexed? [false]")
    public boolean oneIndexed = false;

    @Opt(gloss = "max length of training sentences? [Integer.MAX_VALUE]")
    public int maxTrainLength = Integer.MAX_VALUE;

    @Opt(gloss = "max length of test sentences? [Integer.MAX_VALUE]")
    public int maxTestLength = Integer.MAX_VALUE;

    @Opt(gloss = "Produce posterior alignment weight file when aligning training (lots of disk space)")
	public boolean writePosteriors = false;
	@Opt(gloss = "In outputting posteriors, where do we threshold them (0.0 == all posteriors)")
	public double writePosteriorsThreshold = 0.0;
	
    @Opt(gloss = "number of threads? [# of available processors]")
    public int numThreads = Runtime.getRuntime().availableProcessors();

    @Opt(gloss = "For HMM posterior filter, prune (i,j) pairs at what threshold?")
    public double filterThreshold = 0.001;

    @Opt(gloss = "What is k for (k x 1) and (1 x k) blocks? [1]")
    public int itgBlockSize = 1;

    @Opt(gloss = "Directories from which to read training files")
    public void trainSources(String args) {
      trainSources.addAll(Arrays.asList(args.split(";")));
    }

    @Opt(gloss = "Directories from which to read test files")
    public void testSources(String args) {
      testSources.addAll(Arrays.asList(args.split(";")));
    }

    @Opt(gloss = "Directories from which to read unlabeled files")
    public void unlabeledSources(String args) {
      unlabeledSources.addAll(Arrays.asList(args.split(";")));
    }
    
    @Opt(gloss = "Grammar type: SIMPLE, NORMAL [SIMPLE]")
    public void setGrammarType(GrammarType grammarType) {
      switch (grammarType) {
        case NORMAL:
          grammar = new NormalFormGrammarBuilder().buildGrammar();
          break;
        default:
          grammar = new SimpleGrammarBuilder().buildGrammar();
          break;
      }
    }

    @Opt(gloss = "Number of test sentences to evaluate on after each iteration of training? [0]")
    public int numTestToEval = 0;

    @Opt(gloss = "Number of training sentences to evaluate on after each iteration of training? [0]")
    public int numTrainToEval = 0;

    @Opt(gloss = "Filter Type: NULL, EXTERNAL_POSTERIOR [EXTERNAL_POSTERIOR]")
    public FilterType filterType = FilterType.EXTERNAL_POSTERIOR;

    @Opt(gloss = "Remove non-ITG sentences from training data? [false]")
    public boolean filterTrainNonITG = false;

    public Options() {
      GlobalOptionParser.fillOptions(this);
    }
  }

  BiSpanFilter biSpanFilter;

  Grammar grammar = new NormalFormGrammarBuilder().buildGrammar();
  List<String> trainSources = new ArrayList<String>();
  List<String> testSources = new ArrayList<String>();
  List<String> unlabeledSources = new ArrayList<String>();
  
  WordAligner wordAligner;
  PairDepot testPairs = null;
  PairDepot trainingPairs = null;
  
  ExternalPosteriorFeatureLayer externalPosteriorFeatureLayer;
  ExternalPosteriorsWordAligner externalPosteriorsWordAligner;

  class MinLossITG extends WordAligner {

    @Override
	public Alignment alignSentencePair(SentencePair sentencePair) {
      Grammar grammar = new SimpleGrammarBuilder().buildGrammar();
      ITGParser parser = new ITGParser(grammar);
      parser.setMode(ITGParser.Mode.MAX);
//      mode = opts.posteriorITG ? ITGParser.Mode.SUM :
//          ITGParser.Mode.VITERBI;
      double[][] paddedPotentials = getMinLossAlignmentPotentials(sentencePair);

      int m = sentencePair.getForeignWords().size();
      int n = sentencePair.getEnglishWords().size();

      int maxBlockSize = parser.getMaxBlockSize();
      double[][][] frBlockPotentials = new double[m][n][maxBlockSize + 1];
      double[][][] enBlockPotentials = new double[m][n][maxBlockSize + 1];

      double[][] potentials = new double[m][n];
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          potentials[i][j] = paddedPotentials[i][j];
          //fr block
          for (int k = 2; k <= maxBlockSize; k++) {
            for (int k1 = 0; k1 < k && i + k1 < m; k1++) {
              frBlockPotentials[i][j][k] += paddedPotentials[i + k1][j];
            }
          }
          for (int k = 2; k <= maxBlockSize; k++) {
            for (int k1 = 0; k1 < k && j + k1 < n; k1++) {
              enBlockPotentials[i][j][k] += paddedPotentials[i][j + k1];
            }
          }
        }
      }
      double[] frNullPotentials = new double[sentencePair.getForeignWords().size()];
      double[] enNullPotentials = new double[sentencePair.getEnglishWords().size()];
      parser.setInput(potentials, frNullPotentials, enNullPotentials);
      parser.setBlockPotentials(frBlockPotentials, enBlockPotentials);
      boolean[][] matchingMatrix = parser.getAlignmentMatrix();
      return MatchingWordAligner.makeAlignment(sentencePair, matchingMatrix);
    }

	@Override
	public String getName() {
		return "MinLossITG";
	}

  }

  class MinLossMaxMatching extends WordAligner {

    MatchingExtractor matchingExtractor = new BipartiteMatchingExtractor();

    @Override
	public Alignment alignSentencePair(SentencePair sp) {
      double[][] potentials = getMinLossAlignmentPotentials(sp);
      int[] matching = matchingExtractor.extractMatching(potentials);
      return MatchingWordAligner.makeAlignment(sp, matching);
    }

	@Override
	public String getName() {
		return "MinLossMaxMatching";
	}
  }


  private double[][] getMinLossAlignmentPotentials(SentencePair sp) {
    Alignment reference = sp.getAlignment();

    int m = sp.getForeignWords().size();
    int n = sp.getEnglishWords().size();

    double[][] potentials = new double[m + n][m + n];
    for (double[] row : potentials) {
      Arrays.fill(row, 0.0);
    }

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        boolean sure = reference.containsSureAlignment(j, i);
        boolean possible = reference.containsPossibleAlignment(j, i);

        if (sure) {
          potentials[i][j] = 1.0;
        } else if (possible) {
          potentials[i][j] = 0.0;
        } else {
          potentials[i][j] = -1.0;
        }

      }
    }

    // Fr Nulls
    for (int i = 0; i < m; ++i) {
      potentials[i][i + n] = 0.0;
    }
    // En Nulls
    for (int i = 0; i < n; ++i) {
      potentials[i + m][i] = 0.0;
    }
    return potentials;
  }

  class PunctuationFeature extends FeatureLayer {
    private Set<String> punc = new HashSet<String>(CollectionUtils.makeList(
        ".", ",", "?", "-lrb-", "-rrb-", ";", "。", "``", "\"", "''", "'", "`",
        ","
    ));

    private boolean isPunc(String w) {
      return punc.contains(w);
    }

    private Pattern p = Pattern.compile(".*\\d+.*");

    private boolean isDigit(String w) {
      return p.matcher(w).matches();
    }

    private boolean disagree(boolean p, boolean q) {
      return (p && !q) || (!p && q);
    }


    @Override
    public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
      int m = sp.getForeignWords().size();
      int n = sp.getEnglishWords().size();

      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          String f = sp.getForeignWords().get(i);
          boolean fPunc = isPunc(f);
          boolean fDigit = isDigit(f);
          String e = sp.getEnglishWords().get(j);
          boolean ePunc = isPunc(e);
          boolean eDigit = isDigit(e);
          if (fPunc && ePunc) {
            alignFeats[i][j].add(new FeatureValuePair("both-punc", 1.0));
          }
          if (disagree(ePunc, fPunc)) {
            alignFeats[i][j].add(new FeatureValuePair("differ-punc", 1.0));
          }
          if (eDigit && fDigit) {
            alignFeats[i][j].add(new FeatureValuePair("both-digit", 1.0));
          }
          if (disagree(fDigit, eDigit)) {
            alignFeats[i][j].add(new FeatureValuePair("differ-digit", 1.0));
          }
        }
      }
    }
  }

  class BiasFeature extends FeatureLayer {

    public BiasFeature() {}

    @Override
    public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
      int m = sp.getForeignWords().size();
      int n = sp.getEnglishWords().size();

      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            alignFeats[i][j].add(new FeatureValuePair("align-bias", 1.0));
        }
      }

    }
  }

  class RuleFeatureLayer extends FeatureLayer {
    @Override
    public void addRuleFeatures(SentencePair sp, List<FeatureValuePair>[][][][][] ruleFeats) {
      int m = sp.getForeignWords().size() + 1;
      int n = sp.getEnglishWords().size() + 1;


      List<FeatureValuePair>[] generalRuleFeats = new List[grammar.rules.size()];
      for (int r = 0; r < grammar.rules.size(); r++) {
        generalRuleFeats[r] = new ArrayList<FeatureValuePair>();
        Rule rule = grammar.rules.get(r);
        if (rule instanceof BinaryRule && !rule.isNormal())
          generalRuleFeats[r].add(new FeatureValuePair("inversionRule", 1.0));
      }
      for (int frStart = 0; frStart < m; ++frStart) {
        for (int frStop = frStart; frStop <= m; ++frStop) {
          for (int enStart = 0; enStart < n; ++enStart) {
            for (int enStop = enStart; enStop <= n; ++enStop) {
              ruleFeats[frStart][frStop][enStart][enStop] = generalRuleFeats;
            }
          }
        }
      }
    }

    @Override
    public boolean hasRuleFeatures() {
      return true;
    }
  }

  class DistanceBucketFeatures extends FeatureLayer {

    DistanceBucketFeatures() {
    }

    private int getBucket(int d) {
      if (d < 3) return 0;
      if (d < 10) return 1;
      if (d < 20) return 2;
      return 3;
    }

    @Override
    public void addEnBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] enBlockFeats,
                                   List<FeatureValuePair>[][] alignmentFeats,
                                   List<FeatureValuePair>[] frNullFeats,
                                   List<FeatureValuePair>[] enNullFeats) {

      if (opts.itgBlockSize < 2) return;
      
      int m = sp.getForeignWords().size();
      int n = sp.getEnglishWords().size();
      //Utils.AlignmentDecomposition ad = Utils.decomposeAlignment(sp.getAlignment(),itgBlockSize);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
        	double dist = Math.abs((i + 0.0) / (m + 0.0) - (j + 0.0) / (n + 0.0));
        	for (int k = 2; k <= opts.itgBlockSize && j + k <= n; k++) {
              enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-dist-"+k, dist));
              enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-sqrt-relDist-"+k, Math.sqrt(dist)));
              enBlockFeats[i][j][k].add(new FeatureValuePair("en-block-squared-relDist-"+k, dist * dist));
          }
        }
      }
    }

    @Override
    public void addFrBlockFeatures(SentencePair sp, List<FeatureValuePair>[][][] frBlockFeats,
                                   List<FeatureValuePair>[][] alignmentFeats,
                                   List<FeatureValuePair>[] frNullFeats,
                                   List<FeatureValuePair>[] enNullFeats) {
      
      if (opts.itgBlockSize < 2) return;
      
      int m = sp.getForeignWords().size();
      int n = sp.getEnglishWords().size();
      //Utils.AlignmentDecomposition ad = Utils.decomposeAlignment(sp.getAlignment(),itgBlockSize);
      for (int i = 0; i < m; i++) {
    	  for (int j = 0; j < n; j++) {
        
          double dist = Math.abs((i + 0.0) / (m + 0.0) - (j + 0.0) / (n + 0.0));
          for (int k = 2; k <= opts.itgBlockSize && i + k <= m; k++) {
              frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-dist-"+k, dist));
              frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-sqrt-relDist-"+k, Math.sqrt(dist)));
              frBlockFeats[i][j][k].add(new FeatureValuePair("fr-block-squared-relDist-"+k, dist * dist));
          }
        }
      }
    }

    @Override
    public void addAlignmentFeatures(SentencePair sp, List<FeatureValuePair>[][] alignFeats) {
      int m = sp.getForeignWords().size();
      int n = sp.getEnglishWords().size();

      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          double dist = Math.abs((i + 0.0) / (m + 0.0) - (j + 0.0) / (n + 0.0));

          if (Double.isNaN(dist) || Double.isInfinite(dist) || dist > 1.0) throw new RuntimeException();

          alignFeats[i][j].add(new FeatureValuePair("relDist", dist));
          alignFeats[i][j].add(new FeatureValuePair("sqrt-relDist", Math.sqrt(dist)));
          alignFeats[i][j].add(new FeatureValuePair("squared-relDist", dist * dist));
        }
      }
    }
  }

  List<FeatureLayer> getFeatureLayers() {
    List<FeatureLayer> featLayers = new ArrayList<FeatureLayer>();

    if (opts.addStoredPosteriorFeature) {
      externalPosteriorFeatureLayer = new ExternalPosteriorFeatureLayer("hmm_ef","hmm_fe");
      featLayers.add(externalPosteriorFeatureLayer);
      externalPosteriorsWordAligner = new ExternalPosteriorsWordAligner(externalPosteriorFeatureLayer);
      if (!opts.testOnly) {
        test(externalPosteriorsWordAligner, trainingPairs.asList(), false);
        Logger.logs("%d multi-way alignments\n", externalPosteriorsWordAligner.multiWayAlignments);
        //LogInfo.stdout.printf("%d multi-way alignments\n", externalPosteriorsWordAligner.multiWayAlignments);
        test(externalPosteriorsWordAligner, testPairs.asList(), false);

        if (externalPosteriorsWordAligner.multiWayAlignments == 0) {
          Logger.logs("No multi-way alignments: exiting");
          System.exit(0);
        }
        externalPosteriorsWordAligner.multiWayAlignments = 0;
      }
    }

    if (opts.addCompetitivePosteriorFeature) {
      featLayers.add(new CompetitivePosteriorFeatureLayer());
    }

     if (opts.addAdjacentPosteriorFeature) {
      featLayers.add(new AdjacentPosteriorFeatureLayer());
    }

    if (opts.addDictionaryFeature) {
      featLayers.add(new DictionaryFeatureLayer(opts.lowercase));
    }

    if (opts.addPuncFeatures) {
      featLayers.add(new PunctuationFeature());
    }

    if (opts.addDistFeatures) {
        featLayers.add(new DistanceBucketFeatures());
    }

    if (opts.addLexicalFeatures) featLayers.add(new LexicalFeatureLayer(trainingPairs, testPairs));

    if (opts.addBiasFeatures) featLayers.add(new BiasFeature());

    return featLayers;
  }

  Options opts;


  @Override
public void run() {
    // Read (or prepare to read) data
    opts = new Options();
    try {
      loadData();
    } catch (Exception e) {
      //LogInfo.logs("Couldn't load data %s", e.toString());
      e.printStackTrace();
      System.exit(0);
    }

    if (opts.testOnly) {
      SentencePairFeatureExtractor featExtractor = getSentencePairFeatureExtractor();
      wordAligner = new SupervisedWordAligner(featExtractor);
    } else {
      if (trainSources == null) {
        throw new RuntimeException("Must have trainSources when training");
      }
      loadWordAligner();
      List<Pair<Alignment,Alignment>> trainAligns = test(wordAligner, trainingPairs.asList(), true);
      renderToDisk("train.rendered",trainAligns);
      writePharaohToDisk("train.align", trainAligns);
      List<Pair<Alignment,Alignment>> testAligns = test(wordAligner, testPairs.asList(), true);
      renderToDisk("test.rendered",testAligns);
      writePharaohToDisk("test.align", testAligns);
      if (wordAligner instanceof SupervisedWordAligner) {
        ((SupervisedWordAligner)wordAligner).dumpCached();
      }
    }

    // Unlabeled
    if (unlabeledSources.size() > 0) {
    	writeUnlabeledAlignments(wordAligner);
    }

  }

  private void writeUnlabeledAlignments(final WordAligner wordAligner)
  {
    class Worker implements SimpleMapper<String> {
      @Override
	public void map(String file) {
        File f = new File(file);
        File outFile = new File(Experiment.getResultDir(),f.getName() + ".align");
        File renderOutFile = new File(Experiment.getResultDir(),f.getName() + ".rendered");
        SentencePairReader spReader = getSentencePairReader();
        
        Iterator<SentencePair> sentPairs = ( spReader.pairDepotFromSources(unlabeledSources, 0, Integer.MAX_VALUE, getUnlabeledFilter(), false) ).iterator();
        try {
          PrintWriter alignOut = new PrintWriter(new FileWriter(outFile));
          PrintWriter renderAlignOut = new PrintWriter(new FileWriter(renderOutFile));
          while (sentPairs.hasNext()) {
            Alignment align = wordAligner.alignSentencePair(sentPairs.next());
            if (opts.writePosteriors) {
              alignOut.println(align.outputSoft(opts.writePosteriorsThreshold));
            } else {
              alignOut.println(align.outputHard());
            }
            renderAlignOut.println(align.toString());
            alignOut.flush();
            renderAlignOut.flush();
          }
          alignOut.flush();
          alignOut.close();
          renderAlignOut.flush();
          renderAlignOut.close();          
          Logger.logs("Wrote to " + outFile);
          Logger.logs("Wrote to " + renderAlignOut);
        } catch (IOException e) {
          e.printStackTrace();
          System.exit(0);
        }
      }
    }
    List<Worker> workers = new ArrayList<Worker>();
    for (int i = 0; i < unlabeledSources.size(); i++) workers.add(new Worker());
    AsynchronousMapper.doMapping(unlabeledSources,workers);
  }

  private static void renderToDisk(String file, Collection<Pair<Alignment,Alignment>> aligns) {
    if (file == null) return;
    PrintWriter pw = IOUtils.openOutHard(new File(Experiment.getResultDir(),file));
    for (Pair<Alignment,Alignment> pair: aligns) {
      Alignment  goldAlign = pair.getFirst();
      Alignment guessAlign = pair.getSecond();
      pw.println(Alignment.render(goldAlign,guessAlign));
    }
    pw.flush();
    pw.close();
  }

  private static void writePharaohToDisk(String file, Collection<Pair<Alignment,Alignment>> aligns) {
	    if (file == null) return;
	    PrintWriter pw = IOUtils.openOutHard(new File(Experiment.getResultDir(),file));
	    for (Pair<Alignment,Alignment> pair: aligns) {
	      Alignment guessAlign = pair.getSecond();
	      pw.println(guessAlign.outputHard());
	    }
	    pw.flush();
	    pw.close();
  }

  private static void writeGIZAToDisk(String file, Collection<Pair<Alignment,Alignment>> aligns) {
    if (file == null) return;
    PrintWriter pw = IOUtils.openOutHard(new File(Experiment.getResultDir(),file));
    int idx = 0;
    for (Pair<Alignment,Alignment> pair: aligns) {
      Alignment guessAlign = pair.getSecond();
      guessAlign.writeGIZA(pw, idx);
      idx++;
    }
    pw.flush();
    pw.close();
  }

  private void loadWordAligner() {
    CallbackFunction evalCallback = new CallbackFunction() {
      @Override
	public void callback(Object... args) {
        WordAligner wa = (WordAligner) args[0];
        if (opts.numTrainToEval > 0) {
          List<SentencePair> evalTrainPairs = trainingPairs.asList();
          if (opts.numTrainToEval < evalTrainPairs.size()) {
            evalTrainPairs = evalTrainPairs.subList(0, opts.numTrainToEval);
          }
          Logger.startTrack("training set");
          List<Pair<Alignment,Alignment>> trainAligns = test(wa, evalTrainPairs, false);
          renderToDisk("train.aligns", trainAligns);
          Logger.endTrack();
        }
        if (opts.numTestToEval > 0) {
          List<SentencePair> evalPairs = testPairs.asList();
          if (opts.numTestToEval < evalPairs.size()) {
            evalPairs = evalPairs.subList(0, opts.numTestToEval);
          }
          Logger.startTrack("testing set");
          List<Pair<Alignment,Alignment>> testAligns = test(wa, evalPairs, false);
          renderToDisk("test.aligns", testAligns);
          Logger.endTrack();
        }
      }
    };
    // Build Model
    wordAligner = null;
    switch (opts.modelType) {
      case LEARNED:
        SentencePairFeatureExtractor featExtractor = getSentencePairFeatureExtractor();
        wordAligner = new SupervisedWordAligner(trainingPairs, testPairs.asList(), featExtractor, evalCallback);
        break;
      case MINLOSS_MAXMATCHING:
        wordAligner = new MinLossMaxMatching();
        break;
      case MINLOSS_ITG:
        wordAligner = new MinLossITG();
        break;
      default:
        throw new RuntimeException("No such aligner!");
    }
    Logger.logs("Word Aligner: " + wordAligner.getClass().getSimpleName());
  }

  private SentencePairFeatureExtractor getSentencePairFeatureExtractor() {
    List<FeatureLayer> featLayers = getFeatureLayers();
    Logger.startTrack("Loading BiSpan Filter: " + opts.filterType);
    ensureBiSpanFilter();
    Logger.endTrack();
    SentencePairFeatureExtractor featExtractor = new SentencePairFeatureExtractor(grammar, biSpanFilter, featLayers);
    return featExtractor;
  }

  // Uses model 1
  void ensureBiSpanFilter()  {
    try {
    switch (opts.filterType) {
      case NULL:
        biSpanFilter = new NullBiSpanFilter();
        break;
      case EXTERNAL_POSTERIOR:
        biSpanFilter = new ExternalPosteriorBiSpanFilter(externalPosteriorsWordAligner);
        break;
      default:
        throw new RuntimeException("No such filter: " + opts.filterType.toString());
    }
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(0);
    }    
  }

  private void loadData() throws FileNotFoundException, IOException, ClassNotFoundException {
    Logger.startTrack("Preparing Training Data");
    SentencePairReader spReader = getSentencePairReader();
    testPairs = spReader.pairDepotFromSources(testSources, 0, opts.maxTestSentences, getTestFilter(), false);
    trainingPairs = spReader.pairDepotFromSources(trainSources, 0, opts.maxTrainSentences, getTrainingFilter(), false);
    
    Logger.logs("Number of training sentences: " + trainingPairs.size());
    Logger.logs("Max Train Length: " + opts.maxTrainLength);
    Logger.logs("Number of test sentences: " + testPairs.size());
    Logger.logs("Max Test Length: " + opts.maxTestLength);
    Logger.endTrack();
  }

  private SentencePairReader getSentencePairReader() {
    SentencePairReader spReader = new SentencePairReader(opts.lowercase);
    spReader.setEnglishExtension(opts.englishSuffix);
    spReader.setForeignExtension(opts.foreignSuffix);
    spReader.setReverseAndOneIndex(opts.reverseAlignments, opts.oneIndexed);
    return spReader;
  }

  @SuppressWarnings("unchecked")
  private List<Pair<Alignment,Alignment>> test(final WordAligner wordAligner, List<SentencePair> testSentencePairs, final boolean verbose) {
    class TestWorker implements SimpleMapper<SentencePair> {
      Utils.AlignmentEvalCounts evalCounts = new Utils.AlignmentEvalCounts();
      List<Pair<Alignment,Alignment>> aligns = new ArrayList();
      @Override
	public void map(SentencePair sentencePair) {
        Alignment proposedAlignment = wordAligner.alignSentencePair(sentencePair);
        Alignment referenceAlignment = sentencePair.getAlignment();
        aligns.add(Pair.newPair(referenceAlignment,proposedAlignment));

        if (referenceAlignment == null)
          throw new RuntimeException("No reference alignment found for sentenceID " + sentencePair.getSentenceID());

        Utils.updateCounts(evalCounts,referenceAlignment,proposedAlignment);
      }
    }
    List<Pair<Alignment,Alignment>> aligns = new ArrayList();
    List<TestWorker> workers = new ArrayList<TestWorker>();
    for (int t = 0; t < opts.numThreads; t++) {
      workers.add(new TestWorker());
    }
    AsynchronousMapper.doMapping(testSentencePairs,workers);
    Utils.AlignmentEvalCounts evalCounts = new Utils.AlignmentEvalCounts();
    for (TestWorker worker : workers) {
      evalCounts.merge(worker.evalCounts);
      aligns.addAll(worker.aligns);
    }
    if (evalCounts.numCompleted < testSentencePairs.size()) {
      throw new RuntimeException();
    }
    Logger.startTrack(String.format("Eval on %d sentence pairs", testSentencePairs.size()), true);
    if (wordAligner instanceof SupervisedWordAligner) {
      Logger.logss("Weights: " + ((SupervisedWordAligner)wordAligner).weightCounter());
    }
    Logger.logs(evalCounts.toString());
    Logger.endTrack();
    return aligns;
  }

  private Filter<SentencePair> getTestFilter() {
    return new Filter<SentencePair>() {
      @Override
	public boolean accept(SentencePair t) {
        boolean lengthOkay = t.getAlignment() != null &&
            t.getForeignWords().size() <= opts.maxTestLength &&
            t.getEnglishWords().size() <= opts.maxTestLength;
        if (!lengthOkay) return false;
        
        return true;
      }
    };
  }

  private Filter<SentencePair> getUnlabeledFilter() {
    return new Filter<SentencePair>() {
      @Override
	public boolean accept(SentencePair t) {
        return true;
      }
    };
  }

  private Filter<SentencePair> getTrainingFilter() {
    return new Filter<SentencePair>() {
      @Override
	public boolean accept(SentencePair t) {
        boolean lengthOkay = (t.getForeignWords().size() <= opts.maxTrainLength &&
            t.getEnglishWords().size() <= opts.maxTrainLength);
        if (!lengthOkay) return false;
        if (opts.filterTrainNonITG && !Utils.isItg(t.getAlignment(), opts.itgBlockSize)) {
          Logger.logs("Non-itg:\n" + t.getAlignment().toString());
          return false;
        }
        return true;
      }
    };
  }

  public static void main(String[] args) {
	  Experiment.run(args,new ITGMain(),false,false);
  }

}