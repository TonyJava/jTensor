package jTensor;

public class VariableNode extends Node{

	String name;
	Initializer init;
	
	public VariableNode(int id, int[] dimensions, String name, Initializer init){
		super(id, dimensions);
		this.name = name;
		this.init = init;
	}

	public void initialize(){
		init.initialize(this);
	}

	public boolean runNode(){
		return getTensor() != null;
	}

}