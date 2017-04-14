package jTensor;

public class PlaceHolderNode extends Node{
	private boolean initialized = false;
	private Node input = null;

	public void setInput(Node n){
		input = n;
	}

	public boolean isInputSet(){
		return input != null;
	}

	public void copyFromInput(){
		if(input != null){
			if(input.tensor == null){
				System.out.println("PROOF");
			}
			if(tensor == null){
				System.out.println("PROOF2");
			}
			input.tensor.copyTo(tensor, CopyOp.identity);
		}
	}

	public PlaceHolderNode(int id, int[] dimensions){
		super(id, dimensions);
	}

	public boolean isInitialized(){
		return initialized;
	}

	public void setPlaceHolder(Tensor tensor){
		setTensor(tensor);
		initialized = true;
	}

	public boolean runNode(){
		return initialized;
	}
}