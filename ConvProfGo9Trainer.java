import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.text.DecimalFormat;

import jTensor.*;

public class ConvProfGo9Trainer{
	
	public static void main(String[] args){
		new ConvProfGo9Trainer();
	}

	public ConvProfGo9Trainer(){

		InputData inputData = new ProfGo9InputData(false, true, 20);

		int batchSize = 32;
		int rounds = 10000;
		int testFrequency = 50;
		String filename = "conv_go9_model";
		
		int[] filters = {48, 32};
		int[] pooling = {1, 1};
		int fcSize = 512;
		int[] filterWidth = {3, 3};
		int[] stride = {1, 1};

		HashMap<String, Object> params = new HashMap<String, Object>();
		params.put("filters", filters);
		params.put("pooling", pooling);
		params.put("fcSize", fcSize);
		params.put("filterWidth", filterWidth);
		params.put("stride", stride);
		// params.put("learningRate", 0.0001);
		params.put("inputWidth", 9);
		params.put("inputDepth", 12);
		params.put("inputSamples", batchSize);
		params.put("classes", 82);
		params.put("l2", false);

		Model convModel = new ConvModel(filename, params);
		int[][] convInputs = convModel.getPlaceHolderIds();
		int pInput = convInputs[0][0];
		int pLabels = convInputs[0][1];

		double lastAccuracy = 0;

		long startTime = System.currentTimeMillis();

		DecimalFormat format = new DecimalFormat("#0.0000");

		for(int j = 0; j < rounds; j++){
			Tensor[] trainingBatch = inputData.getBatch(batchSize, false);
			HashMap<Integer, Tensor> trainDict = new HashMap<Integer, Tensor>();
			trainDict.put(pInput, trainingBatch[0]);
			trainDict.put(pLabels, trainingBatch[1]);

			double totalLoss = convModel.fit(trainDict);

			if(j % testFrequency == 0){
				Tensor[] testBatch = inputData.getBatch(batchSize, true);
				HashMap<Integer, Tensor> testDict = new HashMap<Integer, Tensor>();
				testDict.put(pInput, testBatch[0]);
				testDict.put(pLabels, testBatch[1]);
				lastAccuracy = convModel.eval(testDict);
				convModel.saveState();
			}
			double avgTime = (double)(System.currentTimeMillis() - startTime)/j;
			double minsLeft = ((double)(rounds-j)*avgTime) / (1000 * 60);
			int hoursLeft = (int)(minsLeft/60);
			double minsLeftMod = minsLeft % 60;

			System.out.println("Training Loss: " + format.format(totalLoss) + ", Last Accuracy: " + format.format(lastAccuracy) + ", Round: " + j + ", Est Duration: " + hoursLeft + "h:" + format.format(minsLeftMod) + "m");

		}

		Tensor[] testBatch = inputData.getBatch(batchSize, true);	

		HashMap<Integer, Tensor> testDict = new HashMap<Integer, Tensor>();
		testDict.put(pInput, testBatch[0]);
		testDict.put(pLabels, testBatch[1]);

		double accuracy = convModel.eval(testDict);
		System.out.println("Test Accuracy: " + accuracy);
		
	}

}
