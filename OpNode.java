import java.util.ArrayList;

public class OpNode extends Node{
	private ArrayList<Node> inputs;
	private TensorOperation operation = null; // null for nonOp nodes
	private boolean finished = false;

	public OpNode(int id, int[] dimensions, TensorOperation operation){
		super(id, dimensions);
		this.operation = operation;
		inputs = new ArrayList<Node>();
	}

	public ArrayList<Node> getInputs(){
		return inputs;
	}

	public void addInput(Node... nodes){
		for(Node node: nodes){
			inputs.add(node);
		}
	}

	public void reset(){
		finished = false;
	}

	public boolean runNode(){
		if(!finished){
			Tensor[] inputTensors = new Tensor[inputs.size()];
			for(int j = 0; j < inputs.size(); j++){
				if(!inputs.get(j).runNode()){
					return false;
				}
				inputTensors[j] = inputs.get(j).getTensor();
			}
			setTensor(operation.execute(inputTensors));
			finished = true;
		}
		return true;
	}
}