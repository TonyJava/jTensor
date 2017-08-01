import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

import jTensor.*;

public class DNNProfGo9Trainer{
	
	public static void main(String[] args){
		new DNNProfGo9Trainer();
	}

	public DNNProfGo9Trainer(){

		InputData inputData = new ProfGo9InputData(true, true);

		int batchSize = 32;
		int epochs = 20000;
		int testFrequency = 20;
		String filename = "dnn_go9_model";
		
		int[] layerSize = {512};

		HashMap<String, Object> params = new HashMap<String, Object>();
		params.put("layerSize", layerSize);
		params.put("inputSize", 9*9*12);
		params.put("inputSamples", batchSize);
		params.put("classes", 82);
		params.put("activation", "RELU");
		params.put("l2", false);

		Model dnnModel = new DNNModel(filename, params);
		int[][] dnnInputs = dnnModel.getPlaceHolderIds();
		int pInput = dnnInputs[0][0];
		int pLabels = dnnInputs[0][1];

		double lastAccuracy = 0;

		for(int j = 0; j < epochs; j++){
			Tensor[] trainingBatch = inputData.getBatch(batchSize, false);
			HashMap<Integer, Tensor> trainDict = new HashMap<Integer, Tensor>();
			trainDict.put(pInput, trainingBatch[0]);
			trainDict.put(pLabels, trainingBatch[1]);

			double totalLoss = dnnModel.fit(trainDict);

			if(j % testFrequency == 0){
				Tensor[] testBatch = inputData.getBatch(batchSize, true);
				HashMap<Integer, Tensor> testDict = new HashMap<Integer, Tensor>();
				testDict.put(pInput, testBatch[0]);
				testDict.put(pLabels, testBatch[1]);
				lastAccuracy = dnnModel.eval(testDict);
			}

			System.out.println("Training Loss: " + totalLoss + ", Last Accuracy: " + lastAccuracy + " - " + j);
		}

		Tensor[] testBatch = inputData.getBatch(batchSize, true);	

		HashMap<Integer, Tensor> testDict = new HashMap<Integer, Tensor>();
		testDict.put(pInput, testBatch[0]);
		testDict.put(pLabels, testBatch[1]);

		double accuracy = dnnModel.eval(testDict);
		System.out.println("Test Accuracy: " + accuracy);
		
		dnnModel.saveState();
	}

}
