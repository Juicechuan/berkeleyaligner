package edu.berkeley.nlp.wa.mt;

import edu.berkeley.nlp.fig.basic.*;
import edu.berkeley.nlp.wa.syntax.Tree;

import java.io.PrintWriter;
import java.io.Serializable;
import java.util.*;

// A comment
/**
 * Alignments serve two purposes, both to indicate your system's guessed
 * alignment, and to hold the gold standard alignments. Alignments map index
 * pairs to one of three values, unaligned, possibly aligned, and surely
 * aligned. Your alignment guesses should only contain sure and unaligned pairs,
 * but the gold alignments contain possible pairs as well.
 *
 * To build an alignment, start with an empty one and use
 * addAlignment(i,j,true). To display one, use the render method.
 */
public class Alignment implements Serializable {

	public static class AlignmentNode implements Serializable {
		private static final long serialVersionUID = 1L;

		boolean isWord;
		String word;
		Tree<String> internalNode;

		AlignmentNode(String word) {
			isWord = true;
			this.word = word;
			internalNode = null;
		}

		AlignmentNode(Tree<String> internalNode) {
			isWord = false;
			this.word = internalNode.getLabel();
			this.internalNode = internalNode;
		}

		public boolean isInternalNode() {
			return !isWord;
		}

		public Tree<String> getTreeNode() {
			return internalNode;
		}
	}

	private static final long serialVersionUID = 1L;

	Set<Pair<Integer, Integer>> sureAlignments;
	Set<Pair<Integer, Integer>> possibleAlignments;
	private List<List<Integer>> foreignAlignments, englishAlignments;
	public Map<Pair<Integer, Integer>, Double> strengths; // Strength of alignments
	private List<String> englishSentence, foreignSentence;
	private Tree<String> englishTree, foreignTree;
	private List<AlignmentNode> englishNodes, foreignNodes;

	/////////////////////////////
	// Constructors
	/////////////////////////////

	public Alignment(List<String> englishSentence, List<String> foreignSentence,
			Tree<String> englishTree, Tree<String> foreignTree) {
		this.englishSentence = englishSentence;
		this.foreignSentence = foreignSentence;
		sureAlignments = new HashSet<Pair<Integer, Integer>>();
		possibleAlignments = new HashSet<Pair<Integer, Integer>>();
		strengths = new HashMap<Pair<Integer, Integer>, Double>();

		setTrees(englishTree, foreignTree);
	}

	public void setTrees(Tree<String> englishTree, Tree<String> foreignTree) {
		this.englishTree = englishTree;
		this.foreignTree = foreignTree;
		englishNodes = constructAlignmentNodes(this.englishSentence, englishTree);
		foreignNodes = constructAlignmentNodes(this.foreignSentence, foreignTree);
	}

	public Alignment(List<String> englishSentence, List<String> foreignSentence) {
		this(englishSentence, foreignSentence, null, null);
	}

	public Alignment(SentencePair pair, boolean treeAlignment) {
		this(pair.getEnglishWords(), pair.getForeignWords(), treeAlignment ? pair
				.getEnglishTree() : null, treeAlignment ? pair.getForeignTree() : null);
	}

	public Alignment(Alignment al) {
		this(al.englishSentence, al.foreignSentence, al.englishTree, al.foreignTree);
	}

	private List<AlignmentNode> constructAlignmentNodes(List<String> sentence,
			Tree<String> tree) {
		List<AlignmentNode> nodes = new ArrayList<AlignmentNode>();
		for (String word : sentence) {
			nodes.add(new AlignmentNode(word));
		}
		int preTerminalIndex = 0;
		if (tree != null) {
			for (Tree<String> internalNode : tree.getPreOrderTraversal()) {
				if (!(internalNode.isPreTerminal() || internalNode.isLeaf())) {
					nodes.add(new AlignmentNode(internalNode));
				}
				if (internalNode.isPreTerminal()) {
					nodes.get(preTerminalIndex++).internalNode = internalNode;
				}
			}
		}
		return nodes;
	}

	public static List<Tree<String>> getOrderedNodeList(Tree<String> tree) {
		List<Tree<String>> nodes = new ArrayList<Tree<String>>();
		for (Tree<String> preTerminal : tree.getPreTerminals()) {
			nodes.add(preTerminal);
		}
		for (Tree<String> node : tree.getPreOrderTraversal()) {
			if (!(node.isPreTerminal() || node.isLeaf())) {
				nodes.add(node);
			}
		}
		return nodes;
	}

	/////////////////////////////
	// Getter Methods
	/////////////////////////////

	public int getEnglishLength() {
		return englishNodes.size();
	}

	public List<String> getEnglishSentence() {
		return englishSentence;
	}

	public Tree<String> getEnglishTree() {
		return englishTree;
	}

	public List<AlignmentNode> getEnglishNodes() {
		return englishNodes;
	}

	public int getEnglishPosition(Tree<String> englishTreeNode) {
		for (int i = 0; i < englishNodes.size(); i++) {
			if (englishNodes.get(i).internalNode == englishTreeNode) {
				return i;
			}
		}
		return -1;
	}

	public int getForeignLength() {
		return foreignNodes.size();
	}

	public List<String> getForeignSentence() {
		return foreignSentence;
	}

	public Tree<String> getForeignTree() {
		return foreignTree;
	}

	public List<AlignmentNode> getForeignNodes() {
		return foreignNodes;
	}

	public int getForeignPosition(Tree<String> foreignTreeNode) {
		for (int i = 0; i < foreignNodes.size(); i++) {
			if (foreignNodes.get(i).internalNode == foreignTreeNode) {
				return i;
			}
		}
		return -1;
	}

	public Set<Pair<Integer, Integer>> getPossibleAlignments() {
		return possibleAlignments;
	}

	public Set<Pair<Integer, Integer>> getSureAlignments() {
		return sureAlignments;
	}

	public Map<Pair<Integer, Integer>, Double> getStrengths() {
		return strengths;
	}

	public boolean containsSureAlignment(int englishPosition, int foreignPosition) {
		return sureAlignments.contains(new Pair<Integer, Integer>(englishPosition,
				foreignPosition));
	}

	public boolean containsPossibleAlignment(int englishPosition, int foreignPosition) {
		return possibleAlignments.contains(new Pair<Integer, Integer>(englishPosition,
				foreignPosition));
	}

	public boolean containsEnglishParentAlignment(int englishPosition, int foreignPosition) {
		Tree<String> englishNode = englishNodes.get(englishPosition).internalNode;
		for (int i = englishSentence.size(); i < englishNodes.size(); i++) {
			if (sureAlignments.contains(new Pair<Integer, Integer>(i, foreignPosition))) {
				assert !englishNodes.get(i).isWord;
				Tree<String> root = englishNodes.get(i).internalNode;
				for (Tree<String> node : root.getPreOrderTraversal()) {
					if (node == englishNode) return true;
				}
			}
		}
		return false;
	}

	public boolean containsForeignParentAlignment(int englishPosition, int foreignPosition) {
		Tree<String> foreignNode = foreignNodes.get(foreignPosition).internalNode;
		for (int j = foreignSentence.size(); j < foreignNodes.size(); j++) {
			if (sureAlignments.contains(new Pair<Integer, Integer>(englishPosition, j))) {
				assert !foreignNodes.get(j).isWord;
				Tree<String> root = foreignNodes.get(j).internalNode;
				for (Tree<String> node : root.getPreOrderTraversal()) {
					if (node == foreignNode) return true;
				}
			}
		}
		return false;
	}

	public double getStrength(int i, int j) {
		return MapUtils.get(strengths, new Pair<Integer, Integer>(i, j), 0.0);
	}

  double[][] cachedPosteriors = null;
  double[][] cachedTransposePosteriors = null;

	public double[][] getForeignByEnglishPosteriors() {
		int n = getEnglishLength();
		int m = getForeignLength();
		if (cachedPosteriors == null) {
			cachedPosteriors = new double[m][n];
			for (int j = 0; j < m; j++)
				for (int i = 0; i < n; i++)
					cachedPosteriors[j][i] = getStrength(i, j);
		}
		return cachedPosteriors;
	}

  // Return [English x foreign] posteriors
  public double[][] getEnglishByForeignPosteriors() {
    if (cachedTransposePosteriors == null) {
      cachedTransposePosteriors = NumUtils.transpose(getForeignByEnglishPosteriors());
    }
    return cachedTransposePosteriors;
  }


	public List<Integer> getAlignmentsToEnglish(int englishPos) {
		if (englishAlignments == null) {
			gatherAlignmentArrays();
		}
		if (englishPos >= 0 && englishPos < englishAlignments.size()) {
			return englishAlignments.get(englishPos);
		}
		return new ArrayList<Integer>();
	}

	public List<Integer> getAlignmentsToForeign(int sourcepos) {
		if (englishAlignments == null) {
			gatherAlignmentArrays();
		}
		return foreignAlignments.get(sourcepos);
	}

	private void gatherAlignmentArrays() {
		foreignAlignments = new ArrayList<List<Integer>>(foreignNodes.size());
		for (int i = 0; i < foreignNodes.size(); i++) {
			foreignAlignments.add(new ArrayList<Integer>());
		}
		englishAlignments = new ArrayList<List<Integer>>(englishNodes.size());
		for (int i = 0; i < englishNodes.size(); i++) {
			englishAlignments.add(new ArrayList<Integer>());
		}
		for (Pair<Integer, Integer> alignment : sureAlignments) {
			Integer englishPos = alignment.getFirst();
			Integer foreignPos = alignment.getSecond();
			if (englishPos >= 0 && foreignPos >= 0) {
				foreignAlignments.get(foreignPos).add(englishPos);
				englishAlignments.get(englishPos).add(foreignPos);
			}
		}
	}

	/////////////////////////////
	// Setter Methods
	/////////////////////////////

	public void addAlignment(int englishPos, int foreignPos, boolean sure) {
		Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(englishPos,
				foreignPos);
		if (sure) sureAlignments.add(alignment);
		possibleAlignments.add(alignment);
		englishAlignments = null;
		foreignAlignments = null;
	}

	public void addAlignment(int i, int j) {
		addAlignment(i, j, true);
	}

	public void removeAlignment(int i, int j) {
		Pair<Integer, Integer> al = Pair.newPair(i, j);
		if (sureAlignments.contains(al)) sureAlignments.remove(al);
		if (possibleAlignments.contains(al)) possibleAlignments.remove(al);
	}

	public void setStrength(int i, int j, double strength) {
    if (strength >= 0.01) {
		  strengths.put(new Pair<Integer, Integer>(i, j), strength);
    }
	}

	/////////////////////
	// Thresholding
	/////////////////////

	// Create a new alignment based on thresholding the strengths of the provided alignment.
	public Alignment thresholdAlignmentByStrength(double threshold) {
		Alignment newAlignment = new Alignment(englishSentence, foreignSentence,
				englishTree, foreignTree);
		for (Pair<Integer, Integer> ij : strengths.keySet()) {
			int i = ij.getFirst(), j = ij.getSecond();
			double strength = strengths.get(ij);
			newAlignment.setStrength(i, j, strength);
			if (strength >= threshold) newAlignment.addAlignment(i, j, true);
		}
		return newAlignment;
	}

	/**
	 * Generates an alignment for the same sentence pair according to the supplied posteriors.
	 * Posteriors are stored as strengths and above-threshold are stored as sure alignments.
	 *
	 * @param posteriors
	 * @param threshold
	 * @return
	 */
	public Alignment thresholdPosteriors(double[][] posteriors, double threshold) {
		Alignment newAlignment = new Alignment(this);
		int m = posteriors.length; // Foreign length
		int n = posteriors[0].length; // English length
		assert (m == getForeignLength() && n == getEnglishLength());
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				newAlignment.setStrength(i, j, posteriors[j][i]);
				if (posteriors[j][i] >= threshold) {
					newAlignment.addAlignment(i, j, true);
				}
			}
		}
		return newAlignment;
	}

	// Create new alignments for a whole set of alignments.
	public static Map<Integer, Alignment> thresholdAlignmentsByStrength(
			Map<Integer, Alignment> alignments, double threshold) {
		Map<Integer, Alignment> newAlignments = new HashMap<Integer, Alignment>();
		for (int sid : alignments.keySet()) {
			Alignment alignment = alignments.get(sid);
			alignment = alignment.thresholdAlignmentByStrength(threshold);
			newAlignments.put(sid, alignment);
		}
		return newAlignments;
	}

	///////////////////
	// Combining
	///////////////////

	// Not quite a true intersection -- if an alignment from one source is outside the
	// dimensions of the other, it will be included (if we only have information from one
	// source, no combination is possible).
	public Alignment intersect(Alignment a) {
		Tree<String> englishTree = this.englishTree == null ? a.englishTree
				: this.englishTree;
		Tree<String> foreignTree = this.foreignTree == null ? a.foreignTree
				: this.foreignTree;
		Alignment ia = new Alignment(englishSentence, foreignSentence, englishTree,
				foreignTree);
		for (Pair<Integer, Integer> p : sureAlignments)
			if (a.sureAlignments.contains(p) || p.getFirst() >= a.getEnglishLength()
					|| p.getSecond() >= a.getForeignLength()) ia.sureAlignments.add(p);
		for (Pair<Integer, Integer> p : a.sureAlignments)
			if (p.getFirst() >= getEnglishLength() || p.getSecond() >= getForeignLength())
				ia.sureAlignments.add(p);
		for (Pair<Integer, Integer> p : possibleAlignments)
			if (a.possibleAlignments.contains(p) || p.getFirst() >= a.getEnglishLength()
					|| p.getSecond() >= a.getForeignLength())
				ia.possibleAlignments.add(p);
		for (Pair<Integer, Integer> p : a.possibleAlignments)
			if (p.getFirst() >= getEnglishLength() || p.getSecond() >= getForeignLength())
				ia.possibleAlignments.add(p);
		return ia;
	}

	public Alignment subtract(Alignment a) {
		Alignment ia = new Alignment(this);
		for (Pair<Integer, Integer> p : sureAlignments)
			if (!a.sureAlignments.contains(p)) ia.sureAlignments.add(p);
		for (Pair<Integer, Integer> p : possibleAlignments)
			if (!a.possibleAlignments.contains(p)) ia.possibleAlignments.add(p);
		return ia;
	}

	public Alignment union(Alignment a) {
		Tree<String> englishTree = this.englishTree == null ? a.englishTree
				: this.englishTree;
		Tree<String> foreignTree = this.foreignTree == null ? a.foreignTree
				: this.foreignTree;
		Alignment ua = new Alignment(englishSentence, foreignSentence, englishTree,
				foreignTree);
		for (Pair<Integer, Integer> p : sureAlignments)
			ua.sureAlignments.add(p);
		for (Pair<Integer, Integer> p : a.sureAlignments)
			ua.sureAlignments.add(p);
		for (Pair<Integer, Integer> p : possibleAlignments)
			ua.possibleAlignments.add(p);
		for (Pair<Integer, Integer> p : a.possibleAlignments)
			ua.possibleAlignments.add(p);
		return ua;
	}

	public Alignment reverse() {
		Alignment a2 = new Alignment(foreignSentence, englishSentence, foreignTree,
				englishTree);
		for (Pair<Integer, Integer> p : sureAlignments)
			a2.sureAlignments.add(p.reverse());
		for (Pair<Integer, Integer> p : possibleAlignments)
			a2.possibleAlignments.add(p.reverse());
		if (strengths != null) {
			for (Pair<Integer, Integer> p : strengths.keySet())
				a2.setStrength(p.getFirst(), p.getSecond(), strengths.get(p));
		}
		return a2;
	}

	// TODO: figure out how to make this function accommodate trees -- for now, it
	//       just drops all tree-related information
	public Alignment chop(int i1, int i2, int j1, int j2) {
		Alignment choppedAlignment = new Alignment(englishSentence.subList(i1, i2),
				foreignSentence.subList(j1, j2));
		for (int i = i1; i < i2; i++) {
			for (int j = j1; j < j2; j++) {
				boolean isPossible = containsPossibleAlignment(i, j);
				boolean isSure = containsSureAlignment(i, j);
				if (isPossible) {
					choppedAlignment.addAlignment(i - i1, j - j1, isSure);
				}
				choppedAlignment.setStrength(i - i1, j - j1, getStrength(i, j));
			}
		}
		return choppedAlignment;
	}

	///////////////////////////
	// Rendering and Reading
	///////////////////////////

	public String toString() {
		return render(this, this);
	}

	/**
	 * Renders a proposed alignment relative to a reference alignment.
	 * Strengths are ignored.
	 *
	 * @param reference
	 * @param proposed
	 * @return
	 */
	public static String render(Alignment reference, Alignment proposed) {
		return render(reference, proposed, null);
	}

	/**
	 * Renders a proposed alignment relative to a reference with a gloss of
	 * the foreign sentence.
	 *
	 * @param reference
	 * @param proposed
	 * @param glossDictionary
	 * @return
	 */
	public static String render(Alignment reference, Alignment proposed,
			String2DoubleMap glossDictionary) {
		StringBuilder sb = new StringBuilder();
		List<String> englishWords = reference.englishSentence;
		List<String> foreignWords = reference.foreignSentence;
		for (int sourcePosition = 0; sourcePosition < foreignWords.size(); sourcePosition++) {
			for (int targetPosition = 0; targetPosition < englishWords.size(); targetPosition++) {
				boolean sure = reference.containsSureAlignment(targetPosition,
						sourcePosition);
				boolean possible = reference.containsPossibleAlignment(targetPosition,
						sourcePosition);
				char proposedChar = ' ';
				if (proposed.containsSureAlignment(targetPosition, sourcePosition))
					proposedChar = '#';
				else if (proposed.containsEnglishParentAlignment(targetPosition,
						sourcePosition)) proposedChar = '*';
				if (sure) {
					sb.append('[');
					sb.append(proposedChar);
					sb.append(']');
				} else {
					if (possible) {
						sb.append('(');
						sb.append(proposedChar);
						sb.append(')');
					} else {
						sb.append(' ');
						sb.append(proposedChar);
						sb.append(' ');
					}
				}
			}
			sb.append("| ");
			String fword = foreignWords.get(sourcePosition);
			sb.append(fword);
			// Include gloss if it exists
			if (glossDictionary != null) {
				StringDoubleMap eMap = glossDictionary.getMap(fword, false);

				if (eMap != null) {
					String eWords = eMap.keySet().toString();
					sb.append(" ");
					sb.append(eWords);
				}
			}
			sb.append('\n');
		}
		for (int targetPosition = 0; targetPosition < englishWords.size(); targetPosition++) {
			sb.append("---");
		}
		sb.append("'\n");
		boolean printed = true;
		int index = 0;
		while (printed) {
			printed = false;
			StringBuilder lineSB = new StringBuilder();
			for (int targetPosition = 0; targetPosition < englishWords.size(); targetPosition++) {
				String targetWord = englishWords.get(targetPosition);
				if (targetWord.length() > index) {
					printed = true;
					lineSB.append(' ');
					lineSB.append(targetWord.charAt(index));
					lineSB.append(' ');
				} else {
					lineSB.append("   ");
				}
			}
			index += 1;
			if (printed) {
				sb.append(lineSB);
				sb.append('\n');
			}
		}
		return sb.toString();
	}

	/**
	 * Detects whether this is a soft or hard alignment and writes
	 * the appropriate format (soft if available, otherwise hard).
	 */
	public String output() {
		return dumpModifiedPharaoh(!(strengths == null || strengths.isEmpty()), 0.0);
	}

	/**
	 * Writes the sure and proposed alignments in a modified
	 * version of the Pharaoh format.
	 *
	 * For example, if we have 7 sure alignments and two possibles, we get:
	 *
	 * frPos1-enPos1 frPos2-enPos2 ... frPos8-enPos8-P frPos9-enPos9-P
	 *
	 * here, the -P indicates possible alignments.
	 */
	public String outputHard() {
		return dumpModifiedPharaoh(false, 0.0);
	}

	/**
	 * Writes the posterior alignments in a modified version
	 * of the Pharaoh format.  Each alignment is a triple:
	 *
	 * enPos-frPos-strength
	 */
	public String outputSoft() {
		return dumpModifiedPharaoh(true, 0.0);
	}

	public String outputSoft(double threshold) {
		return dumpModifiedPharaoh(true, threshold);
	}

	private String dumpModifiedPharaoh(boolean soft, double threshold) {
		StringBuffer sbuf = new StringBuffer();
		if (soft) {
			for (Pair<Integer, Integer> pair : strengths.keySet()) {
				double strength = strengths.get(pair);
				if (strength >= threshold) {
					sbuf.append((pair.getSecond()) + "-" + (pair.getFirst()) + "-"
							+ strength);
					sbuf.append(" ");
				}
			}
		} else {
			for (Pair<Integer, Integer> pair : sureAlignments) {
				sbuf.append((pair.getSecond()) + "-" + (pair.getFirst()) + " ");
			}
			for (Pair<Integer, Integer> pair : possibleAlignments) {
				if (!sureAlignments.contains(pair)) {
					sbuf.append((pair.getSecond()) + "-" + (pair.getFirst())
							+ "-P ");
				}
			}
		}
		return sbuf.toString();
	}

	/**
	 * Reads a string of alignments generated by an output function (or Pharaoh)
	 * and adds those alignments.
	 * @param string The alignments to parse.
	 */
	public void parseAlignments(String line) {
		parseAlignments(line, false, false);
	}

	/**
	 * Reads a string of alignments generated by an output function (or Pharaoh)
	 * and adds those alignments.
	 * @param line The alignments to parse.
	 * @param reverse Whether to reverse each link
	 * @param oneIndexed If alignments are off by 1, we need to subtract and make them 0-indexed
	 */
	public void parseAlignments(String line, boolean reverse, boolean oneIndexed) {
		//		String noComment = StrUtils.split(line, "#")[0];
		String noComment = line;
		String[] aligns = StrUtils.split(noComment);
		for (int i = 0; i < aligns.length; i++) {
			String[] els = StrUtils.split(aligns[i], "-");
			
			int fr = Integer.parseInt((reverse ? els[1] : els[0]));
			int en = Integer.parseInt((reverse ? els[0] : els[1]));
						
			en = oneIndexed ? en-1 : en;
			fr = oneIndexed ? fr-1 : fr;
			
			if (els.length == 2) {
				addAlignment(en, fr, true);
			} else if (els[2].equals("P")) {
				addAlignment(en, fr, false);
			} else if (els.length == 3) {
				double strength = Double.parseDouble(els[2]);
				setStrength(en, fr, strength);
			} else if (els.length == 4) {
				double strength = Double.parseDouble(els[2]+"-"+els[3]);
				setStrength(en, fr, strength);
			} else {
				throw new RuntimeException("Alignment.parseAlignments Can't parse alignment "+aligns[i]+" len=="+els.length);
			}
		}
	}

	/////////////////////////
	// Outputting GIZA format
	/////////////////////////

	public List<Integer> getNullAlignedEnglishIndices() {
		List<Integer> nulls = new ArrayList<Integer>();
		boolean[] hasAlignment = new boolean[getEnglishLength()];
		for (Pair<Integer, Integer> al : sureAlignments) {
			hasAlignment[al.getFirst()] = true;
		}

		for (int en = 0; en < getEnglishLength(); en++) {
			if (!hasAlignment[en]) nulls.add(en);
		}

		return nulls;
	}

	public List<Integer> getNullAlignedForeignIndices() {
		List<Integer> nulls = new ArrayList<Integer>();
		boolean[] hasAlignment = new boolean[getForeignLength()];
		for (Pair<Integer, Integer> al : sureAlignments) {
			hasAlignment[al.getSecond()] = true;
		}

		for (int fr = 0; fr < getForeignLength(); fr++) {
			if (!hasAlignment[fr]) nulls.add(fr);
		}

		return nulls;
	}

	private List<Integer> addOne(List<Integer> list) {
		List<Integer> newList = new ArrayList<Integer>();
		for (int x : list)
			newList.add(x + 1);
		return newList;
	}

	public void writeGIZA(PrintWriter out, int idx) {
		out
				.printf(
						"# sentence pair (%d) source length %d target length %d alignment score : 0\n",
						idx, englishSentence.size(), foreignSentence.size());
		out.println(StrUtils.join(foreignSentence));
		out
				.printf("NULL ({ %s })", StrUtils
						.join(addOne(getNullAlignedForeignIndices())));
		for (int i = 0; i < englishSentence.size(); i++) {
			List<Integer> alignments = addOne(getAlignmentsToEnglish(i));
			Collections.sort(alignments);
			out.printf(" %s ({ %s })", englishSentence.get(i), StrUtils.join(alignments));
		}
		out.println("");
	}
}