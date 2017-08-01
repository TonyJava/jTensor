import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

import jTensor.*;

public class ConvMNISTTrainer{
	
	public static void main(String[] args){
		new ConvMNISTTrainer();
	}

	public ConvMNISTTrainer(){

		InputData inputData = new MNISTInputData(false, true);

		int batchSize = 8;
		int epochs = 1000;
		String filename = "conv_mnist_model";
		
		int[] filters = {32, 32};
		int[] pooling = {1, 1};
		int fcSize = 512;
		int[] filterWidth = {5, 5};
		int[] stride = {2, 2};

		HashMap<String, Object> params = new HashMap<String, Object>();
		params.put("filters", filters);
		params.put("pooling", pooling);
		params.put("fcSize", fcSize);
		params.put("filterWidth", filterWidth);
		params.put("stride", stride);
		params.put("inputWidth", 28);
		params.put("inputDepth", 1);
		params.put("inputSamples", batchSize);
		params.put("classes", 10);
		params.put("l2", false);

		Model convModel = new ConvModel(filename, params);
		int[][] convInputs = convModel.getPlaceHolderIds();
		int pInput = convInputs[0][0];
		int pLabels = convInputs[0][1];

		long startTime = System.currentTimeMillis();

		for(int j = 0; j < epochs; j++){
			Tensor[] trainingBatch = inputData.getBatch(batchSize, false);

			HashMap<Integer, Tensor> trainDict = new HashMap<Integer, Tensor>();
			trainDict.put(pInput, trainingBatch[0]);
			trainDict.put(pLabels, trainingBatch[1]);

			double totalLoss = convModel.fit(trainDict);

			double avgTime = (double)(System.currentTimeMillis() - startTime)/j;
			double minsLeft = ((double)(epochs-j)*avgTime) / (1000 * 60);

			System.out.println("Training Loss: " + totalLoss + ", Round: " + j + ", Est Duration: " + minsLeft);
		}

		Tensor[] testBatch = inputData.getBatch(batchSize, true);	

		HashMap<Integer, Tensor> testDict = new HashMap<Integer, Tensor>();
		testDict.put(pInput, testBatch[0]);
		testDict.put(pLabels, testBatch[1]);

		double accuracy = convModel.eval(testDict);
		System.out.println("Test Accuracy: " + accuracy);
			
	}

}
