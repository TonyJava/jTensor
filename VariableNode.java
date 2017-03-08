public class VariableNode extends Node{
	
	public VariableNode(int id, int[] dimensions){
		super(id, dimensions);
	}

	public void initializeUniformRange(final double min, final double max){
		final double range = max - min;
		Tensor newTensor = new Tensor(getDimensions(), new InitOp(){
			public double execute(int[] dimensions, int[] index){
				return Math.random() * range + min;
			}
		});
		setTensor(newTensor);
	}

	public boolean runNode(){
		return getTensor() != null;
	}

}