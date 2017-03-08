public class TrainingNode extends Node{

	VariableNode updateTarget;
	Node inputNode;

	public TrainingNode(int id, int[] dimensions){
		super(id, dimensions);
	}

	public void setUpdateTarget(VariableNode updateTarget){
		this.updateTarget = updateTarget;
	}

	public void setInput(Node inputNode){
		this.inputNode = inputNode;
	}

	public boolean runNode(){
		Tensor[] inputTensors = new Tensor[1];
		if(!inputNode.runNode()){
			return false;
		}
		inputTensors[0] = inputNode.getTensor();
		
		Tensor tVar = updateTarget.getTensor();
		Tensor tUpdate = getTensor();
		Index index = new Index(tVar.getOrder());
		do{
			tVar.setValue(index, tVar.getValue(index) + tUpdate.getValue(index));
		}while(index.increment(tVar));
		return true;
	}
	
}