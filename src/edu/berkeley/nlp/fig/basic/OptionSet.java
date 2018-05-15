package edu.berkeley.nlp.fig.basic;

import java.lang.annotation.*;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface OptionSet {
  String name();
}

