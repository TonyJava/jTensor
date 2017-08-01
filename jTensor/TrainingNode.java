package jTensor;

import java.util.ArrayList;

public abstract class TrainingNode extends OpNode{

	protected ArrayList<VariableNode> updateTargets;
	protected ArrayList<Node> gradientInputs;

	public TrainingNode(){
		super(-1, null, null);
		updateTargets = new ArrayList<VariableNode>();
		gradientInputs = new ArrayList<Node>();
		setInputs(gradientInputs);
	}

	// Allows users to directly instantiate a traningNode
	public void setId(int id){
		this.id = id;
	}

	public void addInputUpdateTarget(Node inputNode, VariableNode updateTarget){
		updateTargets.add(updateTarget);
		gradientInputs.add(inputNode);
	}

	public ArrayList<VariableNode> getUpdateTargets(){
		return updateTargets;
	}

	public ArrayList<Node> getGradientInputs(){
		return gradientInputs;
	}
	
}
