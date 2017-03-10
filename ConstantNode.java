public class ConstantNode extends Node{

	public ConstantNode(int id, Tensor tensor){
		super(id, tensor.getDimensions());
		super.setTensor(tensor);
	}

	public boolean runNode(){
		return true;
	}

	@Override
	public void setTensor(Tensor tensor){
		System.out.println("Cannot set tensor of ConstantNode");
	} 
}