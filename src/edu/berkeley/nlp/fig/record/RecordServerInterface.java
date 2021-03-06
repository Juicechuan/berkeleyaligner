package edu.berkeley.nlp.fig.record;

import java.util.*;
import java.rmi.*;

import edu.berkeley.nlp.fig.basic.*;

public interface RecordServerInterface extends Remote {
  public ResultReceiver processCommand(String line, ReceiverInterface receiver) throws RemoteException;
  public String getPrompt() throws RemoteException;
}
