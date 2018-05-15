package edu.berkeley.nlp.wordAlignment.distortion;

import java.util.ArrayList;
import java.util.Currency;
import java.util.List;

import javax.print.attribute.Size2DSyntax;

import edu.berkeley.nlp.wa.mt.SentencePair;
import edu.berkeley.nlp.wa.syntax.DepTree;
import edu.berkeley.nlp.wa.syntax.Node;
import edu.berkeley.nlp.wa.syntax.Tree;
import edu.berkeley.nlp.wordAlignment.TrainingCache;

/**
 * The baseline tree-based distortion model assigns a position to each
 * word in the English tree according to its order in a depth-first 
 * left-branching traversal of the tree.  Distances between words are
 * measured according to these positions.  
 * 
 * A standard window-based HMM distortion model is fitted to these positions.
 */
public class DepTreeDistanceModel extends BucketModel {

	private class DepTreeDistanceParameters implements DistortionParameters {

		private static final long serialVersionUID = 1L;
//		private int[] treePositions;
		private int[][] treeDistance;

		public DepTreeDistanceParameters(SentencePair pair) {
			// Create map of string positions to tree order positions
			DepTree t = pair.getEnglishDepTree();
			if (t == null) {
				throw new RuntimeException("No dependency tree is available for the English sentence.");
			}
			
			createTreeDistances(t);
//			treePositions = new ArrayList<Integer>();
//			treePositions.add(-1);
//			totalNodes = createTreePositions(t.root, 0, treePositions);
//			treePositions.add(totalNodes);
			
//			int N = t.nodes.size();
//			treePositions = new int[N+1];
//			treePositions[0] = -1;
//			totalNodes = createTreePositions(t.root, 0, treePositions);
//			treePositions[N] = totalNodes;

		}

		private void createTreeDistances(DepTree tree) {
			int length = tree.nodes.size();
//			System.out.println(String.valueOf(length));
//			System.out.println(tree.toDependencyString());
			treeDistance = new int[length+1][length+1];
			
			for (int i = 0; i < treeDistance.length-1; i++){
				for (int j = i; j< treeDistance.length; j++) {
//					if (i == j) treeDistance[i][j] = 0;
					if (i == 0) {  // initial
						if (j == tree.root.idx+1) treeDistance[i][j] = 0;
						else treeDistance[i][j] = -1; //tree.root.getPathLength(tree.findNode(j-1));
					} else if (j == length) {  // final
						if (tree.findNode(i-1).isLeaf()) treeDistance[i][j] = 0;
						else treeDistance[i][j] = -1;
					} else {
						Node curr = tree.findNode(i-1);
						Node next = tree.findNode(j-1);

						treeDistance[i][j] = curr.getPathLength(next);
						
					}
					
//					System.out.print(String.valueOf(treeDistance[i][j])+" ");
				}
//				System.out.println();
			}
		}

//		private int createTreePositions(Node root, int rootpos,
//				List<Integer> treePositions) {
//			if (root.isLeaf()) { 
//				treePositions.add(rootpos);
//				return rootpos + 1;
//			} else {
//				int childpos = rootpos + 1;
//				for (Node child : root.getChildren()) {
//					childpos = createTreePositions(child, childpos, treePositions);
//				}
//				return childpos;
//			}
//		}
		
//		private int createTreePositions(Node root, int rootpos,
//				int[] treePositions) {
//			if (root.isLeaf()) { 
//				treePositions[root.idx] = rootpos;
//				return rootpos + 1;
//			} else {
//				treePositions[root.idx] = rootpos;
//				int childpos = rootpos + 1;
//				for (Node child : root.getChildren()) {
//					childpos = createTreePositions(child, childpos, treePositions);
//				}
//				return childpos;
//			}
//		}


		//		int pos = treePhositions.get(i + 1);
		//		int mind = Math.max(0 - pos, -windowSize); // lower distortion bound
		//		int maxd = Math.min(totalNodes - pos, windowSize); // upper distortion bound
		//		assert mind <= maxd : String.format("mind=%d,maxd=%d", mind, maxd);
		//		// Return probs[state][mind] + ... + probs[state][maxd]
		//		return sums[state][maxd + windowSize + 1] - sums[state][mind + windowSize];
		// P(a_j = i | a_{j-1} = h) (Assume nulls are taken care of elsewhere)
		// Current position in English is i
		// Previous position in English is h
		// Independent of current position in French j
		// I = length of English sentence
		// If we're in the fringe buckets, split the probability uniformly
		public double get(int state, int h, int i, int I) {
			int d = getDistance(i, h);
			// div = Number of positions i out of [0, I]
			// that share this bucket d for this given h
			// (so we need to split the probability among these h)
			int div = 0;
			if (d <= -windowSize) {
				d = -windowSize;
				div = getSubWindowCount(h);
			} else if (d >= windowSize) {
				d = windowSize;
				div = getSuperWindowCount(h, I);
			} else {
				div = getWindowCount(h, d);
			}
			double norm = 1;
			if (norm == 0) {
				return 0;
			}
			if (div == 0) // nothing shares with current distance
				div = 1;
			assert div > 0 : String.format(
					"getDDiv: state=%d, d=%d, h=%d, i=%d, I=%d: div=%d, norm=%f/%f", state, d, h,
					i, I, div, norm, sums[state][2 * windowSize + 1]);
			double prob = probs[state][d + windowSize] / div / norm;
			return prob;
		}

		private int getSubWindowCount(int h) {
//			int max = treePositions[h + 1] - windowSize;
			int i = 0;
			int count = 0;
			while (i < treeDistance.length) {
				if (getDistance(h, count-1) >= windowSize)
					count++;
				i++;
			}
			return count;
		}
		
		private int getWindowCount(int h, int d) {
			int i = 0;
			int count = 0;
			while (i < treeDistance.length) {
				if (getDistance(count-1, h) == d || getDistance(count-1, h) == -d)
					count++;
				i++;
			}
			return count;
		}

		private int getSuperWindowCount(int h, int I) {
//			int min = treePositions[h + 1] + windowSize;
			int i = 0;
			int count = 0;
			while (i < treeDistance.length) {
				if (getDistance(I - count, h) >= windowSize)
					count++;
				i++;
			}
			return count;
		}

		public void add(int state, int h, int i, int I, double count) {
			// dbg("DistortProbTable.add state=%d, %d -> %d: %f", state, h, i, count);
			int d = getDistance(i, h);
			if (d < -windowSize)
				d = -windowSize;
			else if (d > windowSize) d = windowSize;
			// Even if we are using a scaled version of the parameter probs[state][...]
			// by the number of divisions, the maximum likelihood estimate of that
			// parameter does not scale the count.
			probs[state][d + windowSize] += count; // /d_div.second;
		}

		int getDistance(int i, int h) {
//			return treePositions[i + 1] - treePositions[h + 1];

			if (i > h)	{
				return treeDistance[h+1][i+1];
			} else {
				return treeDistance[i+1][h+1];
			}	
		}

	}

	public DistortionParameters getDistortionParameters(SentencePair pair) {
		return new DepTreeDistanceParameters(pair);
	}

	static final long serialVersionUID = 42;

	private int totalNodes;

	// sums[state][k] = probs[state][0] + ... + probs[state][k-1]

	public DepTreeDistanceModel(StateMapper stateMapper) {
		super(stateMapper);
	}

	public BucketModel copy() {
		BucketModel model = new DepTreeDistanceModel(stateMapper);
		model.set(this);
		return model;
	}

	public TrainingCache getTrainingCache() {
		return new HMMTrainingCache(-1);
	}

}
