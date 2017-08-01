import jTensor.*;
import jGame.*;

import java.util.*;

public class Go9Bot extends Player{
 
	int pInput;
	int logits;
	Graph graph;
	int[] inputDimensions;
 
	public Go9Bot(Game game, VariableState params){
		super(game);
		int h1Nodes = 128;

		int outputNodes = 81;
		int[] t_inputDimensions = {1, 9, 9, 3};
		inputDimensions = t_inputDimensions;
			
		int filters = 64;
		int filterWidth = 3;
		int stride = 1;
		int filterOutWidth = (9-(filterWidth-stride))/stride;		
		int hiddenNodes = filterOutWidth*filterOutWidth*filters;
		int[] tMult1Size = {1, hiddenNodes};
		int[] vW1Size = {filterWidth, filterWidth, 3, filters};
		int[] vB1Size = {hiddenNodes};
		int[] vW2Size = {hiddenNodes, outputNodes};
		int[] vB2Size = {outputNodes};
		
		graph = new Graph();
		
		pInput = graph.createPlaceholder(inputDimensions);
		int vWeights1 = graph.createVariable("w1", vW1Size);
		int vWeights2 = graph.createVariable("w2", vW2Size);
		int vBias1 = graph.createVariable("b1", vB1Size);
		int vBias2 = graph.createVariable("b2", vB2Size);

		int tConv1 = graph.addOp(new Operations.Conv2d(stride), pInput, vWeights1);
		int tMult1 = graph.addOp(new Operations.TensorReshape(tMult1Size), tConv1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int tOut1 = graph.addOp(new Operations.TensorReLU(), tNet1);
		int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		logits = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		graph.initializeVariables();
		graph.loadVariableState(params); 
	}

	public double[] getMoveDistribution(GameState gameState){
		double[][][][] gameRep = {gameState.getVolumeRepresentation()};
		//double[][] gameUnrolled = new double[1][gameRep.length * gameRep[0].length * gameRep[0][0].length];
		//for(int x = 0; x < gameRep.length; x++){
		//	for(int y = 0; y < gameRep[0].length; y++){
		//		for(int z = 0; z < gameRep[0][0].length; z++){
		//			gameUnrolled[0][x*gameRep[0].length*gameRep[0][0].length+y*gameRep[0][0].length+z] = gameRep[x][y][z];
		//		}
		//	}
		//}
		ArrayList<Integer> legalMoves = game.legalMoves(gameState);

		

		int[] logitRequest = {logits};
		HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
		dict.put(pInput, new Tensor(gameRep, inputDimensions));
		Tensor[] graphOut = graph.runGraph(logitRequest, dict);
		double[] networkOutput = ((double[][])graphOut[0].getObject())[0];
		int maxChoice = -1;

		double maxValue = networkOutput[0];
		boolean valueFound = false;
		for(int i = 0; i < networkOutput.length; i++){
			if(legalMoves.contains(i) && (!valueFound || networkOutput[i] > maxValue)){
				valueFound = true;
				maxValue = networkOutput[i];
				maxChoice = i;
			}
		}

		double softmaxNorm = 0;
		for(int i = 0; i < networkOutput.length; i++){
			if(legalMoves.contains(i)){
				double softVal = Math.exp(networkOutput[i] - maxValue);
				networkOutput[i] = softVal;
				softmaxNorm += softVal;
			}
		}

		for(int i = 0; i < networkOutput.length; i++){
			if(legalMoves.contains(i)){
				networkOutput[i] /= softmaxNorm;
			}else{
				networkOutput[i] = 0;
			}
		}
		return networkOutput;
	}

	public int getMove(GameState gameState){

		ArrayList<Integer> legalMoves = game.legalMoves(gameState);

		if(legalMoves.size() == 1){
			// System.out.println(gameState);
			return legalMoves.get(0);
		}
		
		double[] networkOutput = getMoveDistribution(gameState);

		double randomChoice = Math.random();
		int action = 0;
		for(action = 0; action < networkOutput.length; action++){
			if(legalMoves.contains(action)){
				randomChoice -= networkOutput[action];
				if(randomChoice <= 0){
					break;
				}
			}	
		}
		if(action == networkOutput.length){
			System.out.println("Bad");
		}
		return action;
	}

}
