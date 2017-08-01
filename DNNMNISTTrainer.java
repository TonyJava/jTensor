import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

import jTensor.*;

public class DNNMNISTTrainer{
	
	public static void main(String[] args){
		new DNNMNISTTrainer();
	}

	public DNNMNISTTrainer(){

		InputData inputData = new MNISTInputData(true, true);

		int batchSize = 100;
		int epochs = 1000;
		String filename = "dnn_mnist_model";
		
		int[] layerSize = {512, 256};

		HashMap<String, Object> params = new HashMap<String, Object>();
		params.put("layerSize", layerSize);
		params.put("inputSize", 784);
		params.put("inputSamples", batchSize);
		params.put("classes", 10);
		params.put("activation", "RELU");
		params.put("l2", false);

		Model dnnModel = new DNNModel(filename, params);
		int[][] dnnInputs = dnnModel.getPlaceHolderIds();
		int pInput = dnnInputs[0][0];
		int pLabels = dnnInputs[0][1];

		long startTime = System.currentTimeMillis();

		for(int j = 0; j < epochs; j++){
			Tensor[] trainingBatch = inputData.getBatch(batchSize, false);

			HashMap<Integer, Tensor> trainDict = new HashMap<Integer, Tensor>();
			trainDict.put(pInput, trainingBatch[0]);
			trainDict.put(pLabels, trainingBatch[1]);

			double totalLoss = dnnModel.fit(trainDict);

			double avgTime = (double)(System.currentTimeMillis() - startTime)/j;
			double minsLeft = ((double)(epochs-j)*avgTime) / (1000 * 60);

			System.out.println("Training Loss: " + totalLoss + ", Round: " + j + ", Est Duration: " + minsLeft);
		}

		Tensor[] testBatch = inputData.getBatch(batchSize, true);	

		HashMap<Integer, Tensor> testDict = new HashMap<Integer, Tensor>();
		testDict.put(pInput, testBatch[0]);
		testDict.put(pLabels, testBatch[1]);

		double accuracy = dnnModel.eval(testDict);
		System.out.println("Test Accuracy: " + accuracy);
			
	}

}