package jTensor;

public abstract class InputData{

	protected static class TrainingData{
		public Tensor input;
		public Tensor output;
		public TrainingData(){}
	}

	protected TrainingData[] trainData;
	protected TrainingData[] testData;

	public abstract int getOutputClasses();

	public int[] getBatchInputDimensions(int batchSize){
		int[] inputDimensions = trainData[0].input.getDimensions();
		int[] batchDimensions = new int[inputDimensions.length + 1];
		for(int j = 0; j < inputDimensions.length; j++){
			batchDimensions[j + 1] = inputDimensions[j];
		}
		batchDimensions[0] = batchSize;
		return batchDimensions;
	}

	public int[] getInputDimensions(){
		int[] inputDimensions = trainData[0].input.getDimensions();
		return inputDimensions;
	}

	public int[] getBatchOutputDimensions(int batchSize){
		int[] inputDimensions = trainData[0].output.getDimensions();
		int[] batchDimensions = new int[inputDimensions.length + 1];
		for(int j = 0; j < inputDimensions.length; j++){
			batchDimensions[j + 1] = inputDimensions[j];
		}
		batchDimensions[0] = batchSize;
		return batchDimensions;
	}

	public Tensor[] getBatch(int batchSize, boolean testSet){
		TrainingData[] data = testSet ? testData : trainData;
		Tensor[] trainingSamples = new Tensor[batchSize];
		Tensor[] trainingLabels = new Tensor[batchSize];
		for(int j = 0; j < batchSize; j++){
			TrainingData training = data[(int)(Math.random()*(data.length))];
			trainingSamples[j] = training.input;
			trainingLabels[j] = training.output;
		}
		Tensor[] retVals = {Tensor.combineTensors(trainingSamples), Tensor.combineTensors(trainingLabels)};
		return retVals;
	}
}
