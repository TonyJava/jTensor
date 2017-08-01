package jTensor;

public class GradientDescentNode extends TrainingNode{

	private double learningRate;

	public GradientDescentNode(double learningRate){
		super();
		this.learningRate = learningRate;
	}

	public boolean runNode(){
		for(int j = 0; j < gradientInputs.size(); j++){
			if(!gradientInputs.get(j).runNode()){
				return false;
			}
			final Tensor tGradients = gradientInputs.get(j).getTensor();
			updateTargets.get(j).getTensor().operate(new CopyOp(){
				public double execute(double value, Index index){
					double gradient = tGradients.getValue(index);
					return value + (-1 * learningRate * gradient);
				}
			});
		}
		return true;

	}
}
