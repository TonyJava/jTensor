
import jTensor.*;

import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.text.DecimalFormat;

public class RunModel{

	/*
		Command line args: --argName (type / default)
			--filename (String / "modelfile"+randInt(10000))
			--model (["CONV", "DNN"] / "DNN")
			--dataset (["MNIST", "GO9"] / "MNIST")
			--batchSize (int / 32)
			--rounds (int / 1000)
			--testFrequency (int / 20)
			--params (Comma delimited argName:typeId=value pairs String / "") (eg: --params "inputSize:i=9,layerSize:a=20-10")
			typeIds:
				i - int
				a - int[] (- delimited)
				d - double
				b - boolean
				s - String
	*/

	public static HashMap<String, Object> parseModelParams(String paramsString){
		HashMap<String, Object> modelParams = new HashMap<>();
		int currentChar = -1;
		while(++currentChar < paramsString.length()){
			int startingChar = currentChar;
			while(paramsString.charAt(++currentChar) != ':');
			String argName = paramsString.substring(startingChar, currentChar);
			currentChar += 1;
			Object argValue = null;
			char typeId = paramsString.charAt(currentChar);
			currentChar += 2;
			int valueStartChar = currentChar;
			switch(typeId){
				case 'i':{
					do{currentChar++;}while(currentChar < paramsString.length() && paramsString.charAt(currentChar) != ',');
					argValue = Integer.parseInt(paramsString.substring(valueStartChar, currentChar));
				}break;
				case 'a':{
					ArrayList<Integer> list = new ArrayList<>();
					while(currentChar < paramsString.length() && paramsString.charAt(currentChar) != ','){
						int numStart = currentChar;
						do{currentChar++;}while(currentChar < paramsString.length() && paramsString.charAt(currentChar) != '-' && paramsString.charAt(currentChar) != ',');
						list.add(Integer.parseInt(paramsString.substring(numStart, currentChar)));
						if(currentChar < paramsString.length() && paramsString.charAt(currentChar) == '-'){
							currentChar++;
						}
					}
					int[] intArray = new int[list.size()];
					for(int j = 0; j < list.size(); j++){
						intArray[j] = list.get(j);
					}
					argValue = intArray;
				}break;
				case 'd':{
					do{currentChar++;}while(currentChar < paramsString.length() && paramsString.charAt(currentChar) != ',');
					argValue = Double.parseDouble(paramsString.substring(valueStartChar, currentChar));
				}break;
				case 'b':{
					do{currentChar++;}while(currentChar < paramsString.length() && paramsString.charAt(currentChar) != ',');
					argValue = Boolean.parseBoolean(paramsString.substring(valueStartChar, currentChar));
				}break;
				case 's':{
					do{currentChar++;}while(currentChar < paramsString.length() && paramsString.charAt(currentChar) != ',');
					argValue = paramsString.substring(valueStartChar, currentChar);
				}break;
			}
			modelParams.put(argName, argValue);
		}
		return modelParams;
	}

	public static void main(String[] args){

		// Parse args

		String paramsString = "";
		HashMap<String, Object> argsMap = new HashMap<>();
		for(int j = 0; j < args.length; j += 2){
			String argName = args[j];
			Object value = null;
			if(argName.equals("--batchSize") || argName.equals("--rounds") || argName.equals("--testFrequency")){
				value = Integer.parseInt(args[j+1]);
			}else if(argName.equals("--filename") || argName.equals("--model") || argName.equals("--dataset")){
				value = args[j+1];
			}else if(argName.equals("--params")){
				paramsString = args[j+1];
				continue;
			}
			argsMap.put(argName, value);
		}

		HashMap<String, Object> modelParams = parseModelParams(paramsString);

		// Initialize with default values and check argsMap
		int batchSize = 32;
		if(argsMap.containsKey("--batchSize")){
			batchSize = (Integer)(argsMap.get("--batchSize"));
		}

		int rounds = 10000;
		if(argsMap.containsKey("--rounds")){
			rounds = (Integer)(argsMap.get("--rounds"));
		}

		int testFrequency = 20;
		if(argsMap.containsKey("--testFrequency")){
			testFrequency = (Integer)(argsMap.get("--testFrequency"));
		}

		boolean saveFile = false;
		String filename = "modelfile" + (int)(Math.random()*10000);
		if(argsMap.containsKey("--filename")){
			saveFile = true;
			filename = (String)(argsMap.get("--filename"));
		}

		String dataset = "MNIST";
		if(argsMap.containsKey("--dataset")){
			dataset = (String)(argsMap.get("--dataset"));
		}

		String modelArg = "DNN";
		if(argsMap.containsKey("--model")){
			modelArg = (String)(argsMap.get("--model"));
		}

		boolean flat = !modelArg.equals("CONV");

		InputData inputData;
		if(dataset.equals("GO9")){
			inputData = new ProfGo9InputData(flat, false);
		}else{
			inputData = new MNISTInputData(flat, false);
		}

		int[] inputDimensions = inputData.getInputDimensions();

		modelParams.put("batchSize", batchSize);
		modelParams.put("classes", inputData.getOutputClasses());

		Model model;
		if(modelArg.equals("CONV")){
			modelParams.put("inputWidth", inputDimensions[0]);
			modelParams.put("inputDepth", inputDimensions[2]);
			model = new ConvModel(filename, modelParams);
		}else{
			modelParams.put("inputSize", inputDimensions[0]);
			model = new DNNModel(filename, modelParams);
		}

		int[][] modelInputs = model.getPlaceHolderIds();
		int pInput = modelInputs[0][0];
		int pLabels = modelInputs[0][1];

		double lastAccuracy = 0;

		long startTime = System.currentTimeMillis();

		DecimalFormat format = new DecimalFormat("#0.0000");

		System.out.println("BatchSize: " + batchSize);

		for(int j = 0; j < rounds; j++){
			Tensor[] trainingBatch = inputData.getBatch(batchSize, false);
			HashMap<Integer, Tensor> trainDict = new HashMap<Integer, Tensor>();
			trainDict.put(pInput, trainingBatch[0]);
			trainDict.put(pLabels, trainingBatch[1]);

			double totalLoss = model.fit(trainDict);

			if(j % testFrequency == 0){
				Tensor[] testBatch = inputData.getBatch(batchSize, true);
				HashMap<Integer, Tensor> testDict = new HashMap<Integer, Tensor>();
				testDict.put(pInput, testBatch[0]);
				testDict.put(pLabels, testBatch[1]);
				lastAccuracy = model.eval(testDict);
				if(saveFile){
					model.saveState();
				}
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

		double accuracy = model.eval(testDict);
		System.out.println("Test Accuracy: " + accuracy);
		
	}
}