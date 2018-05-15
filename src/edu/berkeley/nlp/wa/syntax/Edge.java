/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package edu.berkeley.nlp.wa.syntax;

import java.io.Serializable;

/**
 *
 * @author aa496
 */
public class Edge implements Serializable{

    public Node source;
    public Node target;
    public String label;
    public int sourceIndex;
    public int targetIndex;
    public boolean visible = false;
    public int height;
    public int errorType = 0;

    @Override
    public String toString() {
	return label+"["+sourceIndex+"->" + targetIndex+"]";
    }


}
