import jTensor.*;
import jGame.*;

import java.util.*;

public class Go9ModelBot extends Player{
 
 	Model convModel;
	int pInput;
	int[] inputDimensions = {1, 9, 9, 12};

	int batchSize = 1;
	
	int[] filters = {48};
	int[] pooling = {1};
	int fcSize = 512;
	int[] filterWidth = {3};
	int[] stride = {1};
 
	public Go9ModelBot(Game game, String loadFile){
		super(game);

		HashMap<String, Object> params = new HashMap<String, Object>();
		params.put("filters", filters);
		params.put("pooling", pooling);
		params.put("fcSize", fcSize);
		params.put("filterWidth", filterWidth);
		params.put("stride", stride);
		params.put("learningRate", 0.0001);
		params.put("inputWidth", 9);
		params.put("inputDepth", 12);
		params.put("inputSamples", batchSize);
		params.put("classes", 82);
		params.put("l2", false);

		convModel = new ConvModel(loadFile, params);
		int[][] convInputs = convModel.getPlaceHolderIds();
		pInput = convInputs[0][0];
	}

	public double[] getMoveDistribution(GameState gameState){
		double[][][][] gameRep = {gameState.getVolumeRepresentation()};
		ArrayList<Integer> legalMoves = game.legalMoves(gameState);

		HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
		dict.put(pInput, new Tensor(gameRep, inputDimensions));
		Tensor[] graphOut = convModel.predict(dict);
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
