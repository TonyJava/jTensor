package jTensor;

import java.util.ArrayList;

public abstract class Node{

	private Tensor tensor = null;
	private int[] dimensions;
	private int id;

	public Node(int id, int[] dimensions){
		this.id = id;
		this.dimensions = dimensions;
	}

	public Tensor getTensor(){
		return tensor;
	}

	public void setTensor(Tensor tensor){
		this.tensor = tensor;
	}

	public int getId(){
		return id;
	}

	public int[] getDimensions(){
		return dimensions;
	}

	public abstract boolean runNode();

}