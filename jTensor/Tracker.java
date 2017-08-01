package jTensor;

import java.util.ArrayList;

public class Tracker{
	ArrayList<Node> nodes; // length N (number of nodes being tracked)
	ArrayList<Tensor[]> data; // length t, (Tensor[N])

	public Tracker(){
		nodes = new ArrayList<Node>();
		data = new ArrayList<Tensor[]>();
	}

	public void trackNode(Graph graph, int nodeId){

	}
}
