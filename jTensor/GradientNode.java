package jTensor;

public class GradientNode extends TrainingNode{

	public GradientNode(int id){
		super(id);
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
					return value + gradient;
				}
			});
		}
		return true;
	}
}