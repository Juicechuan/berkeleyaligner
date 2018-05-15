package edu.berkeley.nlp.wordAlignment.symmetric.itg;

/**
 * User: aria42
 */
public class State {

  public final boolean isNormal ;
  public int index;
  public final String id;
  public final boolean isTerminal;

  public State(boolean isNormal,
                boolean isTerminal,
                String id)
  {
    this.isNormal = isNormal;
    this.id = id;
    this.isTerminal = isTerminal;
  }

  @Override
  public String toString() {
    return "State(" + id + ")";
  }

  public void setIndex(int index) {
    this.index = index;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    State state = (State) o;

    if (id != null ? !id.equals(state.id) : state.id != null) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = (isNormal ? 1 : 0);
    result = 31 * result + index;
    result = 31 * result + (id != null ? id.hashCode() : 0);
    result = 31 * result + (isTerminal ? 1 : 0);
    return result;
  }
}
