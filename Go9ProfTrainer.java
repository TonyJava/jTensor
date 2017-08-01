import jTensor.*;
import jGame.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.StringTokenizer;

public class Go9ProfTrainer{
	
	public static void main(String[] args){
		new Go9ProfTrainer();
	}

	int testSize = 1000;

	// int h1Nodes = 128;
	int filters = 64;
	int filterWidth = 3;
	int stride = 1;
	int batchSize = 128;
	
	int inputDepth = 12;

	int outputNodes = 82;
	int[] inputDimensions = {batchSize, 9, 9, inputDepth};
	int[] labelDimensions = {batchSize};

	double hitPercentage = 0;
	double testGoal = 60;

	String filename = "conv_supervised_go_9";

	int pInput;
	int pLabels;
	int[] runRequests;
	int[] trainRequests;
	int[] inputSamples;

	public Go9ProfTrainer(){

		int gameFiles = 150;
		TrainingData[][] dataFiles = new TrainingData[gameFiles][];
		int totalLength = 0;
		int discardFirst = 0;
		for(int j = 0; j < gameFiles; j++){
			dataFiles[j] = loadMoves("./9x9sgfs/game" + String.format("%03d", j + 1) + ".sgf");
			totalLength += dataFiles[j].length - discardFirst;
		}
		TrainingData[] data = new TrainingData[totalLength];
		int c = 0;
		for(int j = 0; j < gameFiles; j++){
			for(int i = discardFirst; i < dataFiles[j].length; i++){
				data[c++] = dataFiles[j][i - discardFirst];
			}
		}

		System.out.println("Positions: " + c);

		
		int filterOutWidth = (9-(filterWidth-stride))/stride;		
		int hiddenNodes = filterOutWidth*filterOutWidth*filters;
		int[] tMult1Size = {batchSize, hiddenNodes};
		int[] vW1Size = {filterWidth, filterWidth, inputDepth, filters};
		int[] vB1Size = {hiddenNodes};
		int[] vW2Size = {hiddenNodes, outputNodes};
		int[] vB2Size = {outputNodes};
		
		Graph graph = new Graph();
		
		pInput = graph.createPlaceholder(inputDimensions);
		pLabels = graph.createPlaceholder(labelDimensions);
		int vWeights1 = graph.createVariable("w1", vW1Size);
		int vWeights2 = graph.createVariable("w2", vW2Size);
		int vBias1 = graph.createVariable("b1", vB1Size);
		int vBias2 = graph.createVariable("b2", vB2Size);

		int tConv1 = graph.addOp(new Operations.Conv2d(stride), pInput, vWeights1);
		int tMult1 = graph.addOp(new Operations.TensorReshape(tMult1Size), tConv1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int tOut1 = graph.addOp(new Operations.TensorReLU(), tNet1);
		int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		int logits = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		int loss = graph.addOp(new Operations.SparseCrossEntropySoftmax(), logits, pLabels);
		int train = graph.trainMinimizer(loss, new AdamMinimizerNode());  

		runRequests = new int[2];
		runRequests[0] = logits;
		runRequests[1] = loss;

		trainRequests = new int[3];
		trainRequests[0] = logits; 
		trainRequests[1] = loss;
		trainRequests[2] = train;

		graph.initializeVariables();

		graph.loadVariableState(VariableState.readFromFile(filename));

		int rounds = 0;
		while(true){
			rounds++;

			double totalError = 0;

			int trainingStart = 0;
			int trainingCount = data.length - testSize;

			Tensor[] batch = getBatch(data, trainingStart, trainingCount);

			double[] results = trainBatch(data, graph, true, batch[0], batch[1]);

			hitPercentage = (int)(results[1]*100);

			System.out.println("Avg error (" + trainingCount + "): " + results[0]/batchSize);
			System.out.println("Hit % (" + batchSize + "): " + hitPercentage);
			System.out.println("");

			if(rounds % 100 == 0){
				int testStart = data.length - testSize;
				Tensor[] testBatch = getBatch(data, testStart, testSize);

				double[] testResults = trainBatch(data, graph, false, testBatch[0], testBatch[1]);

				hitPercentage = (int)(testResults[1]*100);

				System.out.println("Test Avg error (" + testSize + "): " + testResults[0]/batchSize);
				System.out.println("Test Hit % (" + batchSize + "): " + hitPercentage);
				System.out.println("----------");

				if(hitPercentage >= testGoal){
					break;
				}
			}
			
		}

		graph.saveVariableState().writeToFile(filename);

	}

	public Tensor[] getBatch(TrainingData[] data, int trainingStart, int trainingCount){
		Tensor inputTensor = new Tensor(inputDimensions);
		double[][][][] inputArray = (double[][][][])(inputTensor.getObject());
		Tensor outputTensor = new Tensor(labelDimensions);
		double[] outputArray = ((double[])(outputTensor.getObject()));
		inputSamples = new int[batchSize];
		for(int j = 0; j < batchSize; j++){
			int randomSample = (int)(Math.random()*(trainingCount) + trainingStart);
			TrainingData training = data[randomSample];
			inputSamples[j] = randomSample;
			inputArray[j] = training.input;
			outputArray[j] = training.output;
		}
		Tensor[] retvals = {inputTensor, outputTensor};
		return retvals;
	}

	public double[] trainBatch(TrainingData[] data, Graph graph, boolean train, Tensor inputTensor, Tensor outputTensor){
		HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
		dict.put(pInput, inputTensor);
		dict.put(pLabels, outputTensor);
		Tensor[] graphOut;
		if(!train){
			graphOut = graph.runGraph(runRequests, dict);	
		}else{
			graphOut = graph.runGraph(trainRequests, dict);	
		}
		int hits = 0;

		for(int j = 0; j < batchSize; j++){

			TrainingData training = data[inputSamples[j]];

			double[] networkOutput = ((double[][])graphOut[0].getObject())[j];

			double maxValue = networkOutput[0];
			boolean valueFound = false;
			for(int i = 0; i < networkOutput.length; i++){
				if(training.legalMoves.contains(i) && (!valueFound || networkOutput[i] > maxValue)){
					valueFound = true;
					maxValue = networkOutput[i];
				}
			}

			double softmaxNorm = 0;
			for(int i = 0; i < networkOutput.length; i++){
				if(training.legalMoves.contains(i)){
					double softVal = Math.exp(networkOutput[i] - maxValue);
					networkOutput[i] = softVal;
					softmaxNorm += softVal;
				}
			}

			double randomChoice = Math.random();
			int action;
			for(action = 0; ; action++){
				if(training.legalMoves.contains(action)){
					randomChoice -= networkOutput[action]/softmaxNorm;
					if(randomChoice < 0){
						break;
					}
				}	
			}

			if(training.output == action){
				hits++;
			}
		}

		double error = graphOut[1].getSum();		
		double hitRate = (double)hits/batchSize;
		
		double[] retVals = {error, hitRate};
		return retVals;
	}

	public static class TrainingData{
		double[][][] input;
		//double[] output;
		int output; // classification index
		ArrayList<Integer> legalMoves;
	}


	// aa is top left corner is 1
	// ab is to the right and is 2
	public TrainingData[] loadMoves(String file){
		BufferedReader br = null;
		ArrayList<double[][][]> games = new ArrayList<double[][][]>();
		ArrayList<Integer> correctMoves = new ArrayList<Integer>();
		ArrayList<ArrayList<Integer>> legalMoveLists = new ArrayList<ArrayList<Integer>>();


		// System.out.println("Loading " + file);

		try {

			br = new BufferedReader(new FileReader(file));

			String line = "";

			while ((line = br.readLine()) != null) {
				if(line.length() < 3 || !line.substring(0, 3).equals(";B[")){
					continue;
				}
				StringTokenizer st = new StringTokenizer(line, ";BW[]");
				while(st.hasMoreTokens()){
					String coordinates = st.nextToken();
					int x = coordinates.charAt(1) - 'a';
					int y = coordinates.charAt(0) - 'a';
					int move = y*9 + x + 1;
					correctMoves.add(move);
					// System.out.println("MOVE: " + move + ", " + x + ":" + y);
				}
			}
		} catch (Exception e) {
			System.out.println("Error: " + e);
		}

		Go goGame = new Go(9);
		goGame.setTrackMoves(true);
		Go.GoGameState state = (Go.GoGameState)goGame.newGame();

		for(int i = 0; i < correctMoves.size(); i++){
			

			int correctMove = correctMoves.get(i);

			// int turn = rawMoves[rawMoves.length - 1];

			// turn = turn + 1;

			ArrayList<Integer> legals = goGame.legalMoves(state);
			

			legalMoveLists.add(legals);
			double[][][] gameRep = state.getVolumeRepresentation();
			// double[][] gameUnrolled = new double[1][gameRep.length * gameRep[0].length * gameRep[0][0].length];
			// for(int x = 0; x < gameRep.length; x++){
			// 	for(int y = 0; y < gameRep[0].length; y++){
			// 		for(int z = 0; z < gameRep[0][0].length; z++){
			// 			gameUnrolled[0][x*gameRep[0].length*gameRep[0][0].length+y*gameRep[0][0].length+z] = gameRep[x][y][z];
			// 		}
			// 	}
			// }
			games.add(gameRep);

			int result = goGame.simMove(correctMove, state);
			if(result != 0){
				System.out.println("Game over " + result);
			}

		}
	

		TrainingData[] data = new TrainingData[games.size()];
		for(int j = 0; j < games.size(); j++){
			data[j] = new TrainingData();

			data[j].input = games.get(j);
			data[j].output = correctMoves.get(j);
			data[j].legalMoves = legalMoveLists.get(j);
		}


		return data;
	}



}
