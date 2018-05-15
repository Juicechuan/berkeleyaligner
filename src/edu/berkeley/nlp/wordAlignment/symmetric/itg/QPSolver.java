package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.math.DoubleArrays;

/*****
 * 
 * @author Koby Crammer
 *
 *****/
public class QPSolver {

    static final int MAX_ITER = 10000;
    static final double EPS = 1e-8;
    static final double ZERO = 0.0000000000000001;

    public static double[] hildreth(double[][] a, double[] b) {
	
	int i;
	
	//System.out.println("In hildreth");

	double[] alpha = new double[b.length];
	
	double[] F = new double[b.length];
	double[] kkt = new double[b.length];
	double max_kkt = Double.NEGATIVE_INFINITY;
	
	int K = a.length;
	
	double[][] A = new double[K][K];
	boolean[] is_computed = new boolean[K];
	for(i = 0; i < K; i++) {
	    A[i][i] = DoubleArrays.innerProduct(a[i], a[i]);
	    is_computed[i] = false;
	}
	
	int max_kkt_i = -1;
	
	
	for(i = 0; i < F.length; i++) {
	    F[i] = b[i];
	    kkt[i] = F[i];
	    if(kkt[i] > max_kkt) { max_kkt = kkt[i]; max_kkt_i = i; }
	}
	
	int iter = 0;
	double diff_alpha;
	double try_alpha;
	double add_alpha;
	
	while (max_kkt >= EPS && iter < MAX_ITER) {
	    //System.out.println("max_kkt_i = " + max_kkt_i + " max_kkt = " + max_kkt + " iter = " + iter);
	    
	    diff_alpha = A[max_kkt_i][max_kkt_i] <= ZERO ? 0.0 : F[max_kkt_i]/A[max_kkt_i][max_kkt_i];
	    try_alpha = alpha[max_kkt_i] + diff_alpha;
	    add_alpha = 0.0;
	    
	    if(try_alpha < 0.0)
		add_alpha = -1.0 * alpha[max_kkt_i];
	    else
		add_alpha = diff_alpha;
	    
	    alpha[max_kkt_i] = alpha[max_kkt_i] + add_alpha;
	    
	    if (!is_computed[max_kkt_i]) {
		for(i = 0; i < K; i++) {
		    A[i][max_kkt_i] = DoubleArrays.innerProduct(a[i], a[max_kkt_i]);
		    is_computed[max_kkt_i] = true;
		}
	    }
	    
	    for(i = 0; i < F.length; i++) {
		F[i] -= add_alpha * A[i][max_kkt_i];
		kkt[i] = F[i];
		if(alpha[i] > ZERO)
		    kkt[i] = Math.abs(F[i]);
	    }
	    
	    max_kkt = Double.NEGATIVE_INFINITY;
	    max_kkt_i = -1;
	    for(i = 0; i < F.length; i++)
		if(kkt[i] > max_kkt) { max_kkt = kkt[i]; max_kkt_i = i; }
	    
	    iter++;
	}
	
	return alpha;
    }




    public static double[] hildreth(double[][] a, double[] b, double C) {
	
	int i;
	
	double[] alpha = new double[b.length];
	
	double[] F = new double[b.length];
	double[] kkt = new double[b.length];
	double max_kkt = Double.NEGATIVE_INFINITY;
	
	int K = a.length;
	
	double[][] A = new double[K][K];
	boolean[] is_computed = new boolean[K];
	for(i = 0; i < K; i++) {
	    A[i][i] = DoubleArrays.innerProduct(a[i], a[i]);
	    is_computed[i] = false;
	}
	
	int max_kkt_i = -1;
	
	
	for(i = 0; i < F.length; i++) {
	    F[i] = b[i];
	    kkt[i] = F[i];
	    if(kkt[i] > max_kkt) { max_kkt = kkt[i]; max_kkt_i = i; }
	}
	
	int iter = 0;
	double diff_alpha;
	double try_alpha;
	double add_alpha;
	
	while (max_kkt >= EPS && iter < MAX_ITER) {
	    
	    diff_alpha = A[max_kkt_i][max_kkt_i] <= ZERO ? 0.0 : F[max_kkt_i]/A[max_kkt_i][max_kkt_i];
	    try_alpha = alpha[max_kkt_i] + diff_alpha;
	    add_alpha = 0.0;
	    
	    if (try_alpha < 0.0)
		add_alpha = -1.0 * alpha[max_kkt_i];
	    else if (try_alpha > C)
		add_alpha = C - alpha[max_kkt_i];
	    else
		add_alpha = diff_alpha;
	    
	    alpha[max_kkt_i] = alpha[max_kkt_i] + add_alpha;
	    
	    if (!is_computed[max_kkt_i]) {
		for(i = 0; i < K; i++) {
		    A[i][max_kkt_i] = DoubleArrays.innerProduct(a[i],a[max_kkt_i]);
		    is_computed[max_kkt_i] = true;
		}
	    }
	    
	    for(i = 0; i < F.length; i++) {
		F[i] -= add_alpha * A[i][max_kkt_i];
		kkt[i] = F[i];
		if (alpha[i] > C - ZERO)
		    kkt[i]=-kkt[i];
		else if (alpha[i] > ZERO)
		    kkt[i] = Math.abs(F[i]);
	    }
	    
	    max_kkt = Double.NEGATIVE_INFINITY;
	    max_kkt_i = -1;
	    for(i = 0; i < F.length; i++)
		if(kkt[i] > max_kkt) { max_kkt = kkt[i]; max_kkt_i = i; }
	    
	    iter++;
	}
	//System.out.println("max_kkt_i = " + max_kkt_i + " max_kkt = " + max_kkt + " iter = " + iter);
	//if (iter >= MAX_ITER)
	//    System.out.println("QPSolver : iter >= MAX_ITER");
	return alpha;
    }    
}
