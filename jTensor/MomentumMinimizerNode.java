package jTensor;

import java.util.ArrayList;

public class MomentumMinimizerNode extends TrainingNode{

	private double learningRate;
	private double momentumRate;
	private ArrayList<Tensor> lastUpdate;
	public MomentumMinimizerNode(double learningRate, double momentumRate){
		super();
		this.learningRate = learningRate;
		this.momentumRate = momentumRate;
		lastUpdate = new ArrayList<Tensor>();
	}

	@Override
	public void addInputUpdateTarget(Node inputNode, VariableNode updateTarget){
		super.addInputUpdateTarget(inputNode, updateTarget);
		lastUpdate.add(null);
	}

	public boolean runNode(){
		for(int j = 0; j < gradientInputs.size(); j++){
			if(!gradientInputs.get(j).runNode()){
				return false;
			}
			final Tensor tGradients = gradientInputs.get(j).getTensor();
			Tensor tempLastUpdateTensor = lastUpdate.get(j);
			if(tempLastUpdateTensor == null){
				tempLastUpdateTensor = new Tensor(tGradients.getDimensions());
				lastUpdate.set(j, tempLastUpdateTensor);
			}
			final Tensor lastUpdateTensor = tempLastUpdateTensor;
			updateTargets.get(j).getTensor().operate(new CopyOp(){
				public double execute(double value, Index index){
					double gradient = (tGradients.getValue(index) * -1 * learningRate) + (momentumRate * lastUpdateTensor.getValue(index));
					lastUpdateTensor.setValue(index, gradient);
					return value + gradient;
				}
			});
		}
		return true;

	}
}
