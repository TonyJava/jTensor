import java.util.ArrayList;

public abstract class TrainingNode extends OpNode{

	protected ArrayList<VariableNode> updateTargets;
	protected ArrayList<Node> gradientInputs;

	public TrainingNode(int id){
		super(id, null, null);
		updateTargets = new ArrayList<VariableNode>();
		gradientInputs = new ArrayList<Node>();
		setInputs(gradientInputs);
	}

	public void addInputUpdateTarget(Node inputNode, VariableNode updateTarget){
		updateTargets.add(updateTarget);
		gradientInputs.add(inputNode);
	}	
	
}