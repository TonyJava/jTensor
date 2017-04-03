package jTensor;

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

	public TensorOperation getOperation(){
		return operation;
	}

	public ArrayList<Node> getInputs(){
		return inputs;
	}

	public void setInputs(ArrayList<Node> inputs){
		this.inputs = inputs;
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
			int[][] inputDimensions = new int[inputs.size()][];
			for(int j = 0; j < inputs.size(); j++){
				if(!inputs.get(j).runNode()){
					return false;
				}
				inputTensors[j] = inputs.get(j).getTensor();
				inputDimensions[j] = inputTensors[j].getDimensions();
				// System.out.println("OpNode " + getId() + ", input " + j + ": ");
				// inputTensors[j].printTensor();
			}
		
			int[] outputDimensions = operation.getOutputDimensions(inputDimensions);

			Tensor currentTensor = getTensor();
			boolean sameSize = currentTensor != null;
			if(currentTensor != null){
				int[] currentDimensions = currentTensor.getDimensions();
				sameSize = (outputDimensions.length == currentDimensions.length);
				for(int j = 0; sameSize && j < outputDimensions.length; j++){
					if(outputDimensions[j] != currentDimensions[j]){
						sameSize = false;
					}
				}
			}
			if(!sameSize){
				setTensor(new Tensor(outputDimensions));
			}
			operation.execute(getTensor(), inputTensors);

			if(Graph.DEBUG_ON){
				System.out.println("OpNode " + getId() + ":\n\tAvg Value: " + getTensor().getAverage() + "\n\tAvg Mag: " + getTensor().getAverageMagnitude());
				if(Graph.DEBUG_VERBOSE){
					getTensor().printTensor();
				}
			}
			
			finished = true;
		}
		return true;
	}
}