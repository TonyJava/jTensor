package jTensor;

import java.util.ArrayList;

public class AdamMinimizerNode extends TrainingNode{

	private final double epsilon = 1e-8;

	private double learningRate;
	private double meanDecay, varianceDecay;
	private double timeStep;
	private ArrayList<Tensor> meanMoment, varianceMoment;

	public AdamMinimizerNode(double learningRate, double meanDecay, double varianceDecay){
		super();
		this.learningRate = learningRate;
		this.meanDecay = meanDecay;
		this.varianceDecay = varianceDecay;
		meanMoment = new ArrayList<Tensor>();
		varianceMoment = new ArrayList<Tensor>();
		timeStep = 0;
	}

	public AdamMinimizerNode(){
		super();
		this.learningRate = .001;
		this.meanDecay = .9;
		this.varianceDecay = .999;
		meanMoment = new ArrayList<Tensor>();
		varianceMoment = new ArrayList<Tensor>();
		timeStep = 0;
	}

	public AdamMinimizerNode(double learningRate){
		super();
		this.learningRate = learningRate;
		this.meanDecay = .9;
		this.varianceDecay = .999;
		meanMoment = new ArrayList<Tensor>();
		varianceMoment = new ArrayList<Tensor>();
		timeStep = 0;
	}

	@Override
	public void addInputUpdateTarget(Node inputNode, VariableNode updateTarget){
		super.addInputUpdateTarget(inputNode, updateTarget);
		meanMoment.add(null);
		varianceMoment.add(null);
	}

	public boolean runNode(){
		timeStep += 1;
		for(int j = 0; j < gradientInputs.size(); j++){
			if(!gradientInputs.get(j).runNode()){
				return false;
			}
			final Tensor tGradients = gradientInputs.get(j).getTensor();


			// Node target = updateTargets.get(j);

			// String gradDims = "[";
			// int[] gDimensions = tGradients.getDimensions();
			// for(int d : gDimensions){
			// 	gradDims = gradDims + d + ", ";
			// }
			// gradDims = gradDims.substring(0, gradDims.length() - 2) + "]";

			// String targetDims = "[";
			// int[] tDims = target.getDimensions();
			// for(int d : tDims){
			// 	targetDims = targetDims + d + ", ";
			// }
			// targetDims = targetDims.substring(0, targetDims.length() - 2) + "]";


			// System.out.println("grads: " + gradDims + " ::: targets: " + targetDims);



			Tensor tempMeanTensor = meanMoment.get(j);
			Tensor tempVarianceTensor = varianceMoment.get(j);
			if(tempMeanTensor == null){
				tempMeanTensor = new Tensor(tGradients.getDimensions());
				meanMoment.set(j, tempMeanTensor);
				tempVarianceTensor = new Tensor(tGradients.getDimensions());
				varianceMoment.set(j, tempVarianceTensor);
			}
			final Tensor meanTensor = tempMeanTensor;
			final Tensor varianceTensor = tempVarianceTensor;
			updateTargets.get(j).getTensor().operate(new CopyOp(){
				public double execute(double value, Index index){
					double gradient = tGradients.getValue(index);
					double m0 = meanTensor.getValue(index);
					double m1 = meanDecay * m0 + (1  - meanDecay) * gradient;
					double v0 = varianceTensor.getValue(index);
					double v1 = varianceDecay * v0 + (1  - varianceDecay) * Math.pow(gradient, 2);
					double effectiveStep = learningRate * Math.sqrt(1 - Math.pow(varianceDecay, timeStep))/(1 - Math.pow(meanDecay, timeStep));
					double update = -1 * effectiveStep * m1 / (Math.sqrt(v1) + epsilon);
					meanTensor.setValue(index, m1);
					varianceTensor.setValue(index, v1);
					return value + update;
				}
			});
		}
		return true;
	}
}
