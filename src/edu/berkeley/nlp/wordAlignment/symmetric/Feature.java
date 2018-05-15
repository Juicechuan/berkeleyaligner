package edu.berkeley.nlp.wordAlignment.symmetric;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Oct 10, 2008
 * Time: 2:00:34 PM
 */
public class Feature {
  public Object feature;
  public int index;

  public Feature() {

  }
  public Feature(Object f) {
    this.feature = f;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result;
    result = prime * result
        + ((feature == null) ? 0 : feature.hashCode());
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    Feature other = (Feature) obj;
    if (feature == null) {
      if (other.feature != null)
        return false;
    } else if (!feature.equals(other.feature))
      return false;
    return true;
  }

  public String toString() {
    return "feat(" + feature + ")";
  }

}

