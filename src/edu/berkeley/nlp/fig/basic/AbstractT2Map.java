package edu.berkeley.nlp.fig.basic;

import edu.berkeley.nlp.fig.basic.*;
import static edu.berkeley.nlp.fig.basic.LogInfo.*;

import java.io.*;
import java.util.*;

/**
 * Just a dummy template right now.
 * TODO: move functionality in here.
 */
public abstract class AbstractT2Map<S extends Comparable<S>, T extends Comparable<T>> {
  public abstract void switchToSortedList();
  public abstract void lock();
  public abstract int size();

  protected boolean locked;
  protected AbstractTMap.Functionality<T> keyFunc;
}
