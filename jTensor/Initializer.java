package jTensor;

public abstract class Initializer{
	
	public abstract void initialize(VariableNode node);

	public static class UniformInitializer extends Initializer{
		private double range;
		private double center;

		public UniformInitializer(double range, double center){
			this.range = range;
			this.center = center;
		}

		public void initialize(VariableNode node){
			final double min = center - (range/2);
			Tensor newTensor = new Tensor(node.getDimensions(), new InitOp(){
				public double execute(int[] dimensions, Index index){
					return Math.random() * range + min;
				}
			});
			node.setTensor(newTensor);
		}
	}

}