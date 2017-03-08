public class PlaceHolderNode extends Node{
	private boolean initialized = false;

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