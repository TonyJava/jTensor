import jTensor.*;
import jREC.*;
import jGame.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class PolicyAgent{

	final static String loadFile = "conv_go_9";
	final static String saveFile = "reinforce_go_9";
	
	public static void main(String[] args){
		// Game go = new Go(9);
		Game game = new TicTacToe();
		// new PolicyAgent(new GameEnvironment(go, new Go9Bot(go, VariableState.readFromFile(loadFile))), 128);
		new PolicyAgent(new GameEnvironment(game, new RandomPlayer(game)), 100);
		// new PolicyAgent(new Rabbit(), 16);
	}

	public PolicyAgent(Environment env, int hiddenNodesCount){

		// Hyper Parameters
		final int episodesPerEpoch = 10000;
		final int batchSize = 1000;
		final double expSampleRate = 10;
		final double performanceGoal = 1000;
		final double rewardDiscount = .99;
		final double e_greedy_start = .5;
		final int e_greedy_decay = 1; // reciprocal of decay


		// Create the policy network

		int inputSize = env.getObservationSpace().dimensions[0];
		int outputSize = env.getActionSpace().dimensions[0];
		System.out.println("Obs Space: " + inputSize);
		System.out.println("Act Space: " + outputSize);
		int[] inputDimensions = {batchSize, inputSize};
		int[] labelDimensions = {batchSize}; // posision of 1 in one-hot vector for 1 training example
		// int[] labelDimensions = {batchSize, outputSize}; // index of the one in the one hot encoded action
		int[] rewardDimensions = {batchSize}; // The reward for the action
		int[] hiddenNodes = {hiddenNodesCount};
		int[] vW1Size = {inputDimensions[1], hiddenNodes[0]};
		int[] vB1Size = {vW1Size[1]};
		int[] vW2Size = {vW1Size[1], outputSize};
		int[] vB2Size = {vW2Size[1]};

		Graph graph = new Graph();
		int pInput = graph.createPlaceholder(inputDimensions);
		int pLabels = graph.createPlaceholder(labelDimensions);
		int pRewards = graph.createPlaceholder(rewardDimensions);
		int vWeights1 = graph.createVariable("w1", vW1Size);
		int vBias1 = graph.createVariable("b1", vB1Size);
		int vWeights2 = graph.createVariable("w2", vW2Size);
		int vBias2 = graph.createVariable("b2", vB2Size);
		int tMult1 = graph.addOp(new Operations.MatMult(), pInput, vWeights1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int tOut1 = graph.addOp(new Operations.TensorReLU(), tNet1);
		int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		int logits = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		int actionProbs = graph.addOp(new Operations.MatSoftmax(), logits);
		
		int xEntropy = graph.addOp(new Operations.SparseCrossEntropySoftmax(), logits, pLabels);
		int scaledEntropy = graph.addOp(new Operations.TensorScale(), xEntropy, pRewards);
		int reinforceLoss = graph.addOp(new Operations.VecAvg(), scaledEntropy);

		int train = graph.trainMinimizer(reinforceLoss, new AdamMinimizerNode(0.0001));
		
		int[] runRequests = {actionProbs};
		int[] trainRequests = {train};

		graph.initializeVariables();
		if(loadFile != null){
			graph.loadVariableState(VariableState.readFromFile(loadFile));
		}

		// Set up for policy gradient
		int epoch = 0;

		double performance = 0;
		double runningPerformance = 0;
		while(true){

			final boolean renderScreen = runningPerformance >= performanceGoal;
			if(renderScreen){
				graph.saveVariableState().writeToFile(saveFile);
			}

			epoch += 1;

			ArrayList<Info> observations = new ArrayList<Info>();
			ArrayList<Integer> actions = new ArrayList<Integer>();
			ArrayList<Double> discountedRewards = new ArrayList<Double>();

			// Perform policy rollouts
			double e_greedy = e_greedy_start*1.0/(epoch/e_greedy_decay + 1.0);
			// double e_greedy = 0;
			// if(epoch < 2){
			// 	e_greedy = 1;
			// }
			// System.out.println("Starting rollouts, e_greedy: " + e_greedy);
			performance = 0;
			// int[] actionsSelected = new int[outputSize];
			for(int ep = 0; ep < episodesPerEpoch; ep++){
				ArrayList<Double> rewards = new ArrayList<Double>();
				boolean finished = false;
				
				Info obs = env.reset();
				double episodeReward = 0;
				while(!finished){

					if(renderScreen){
						env.render();
						try{Thread.sleep(500);}catch(Exception e){}
					}



					// Get action
					int action = 0;
					

					if(env instanceof GameEnvironment){

						if(Math.random() < e_greedy){
							// Take random action
							ArrayList<Integer> legalMoves = ((GameEnvironment)env).game.legalMoves(((GameEnvironment.State)obs).gameState);
							action = legalMoves.get((int)(Math.random()*legalMoves.size()));
						}else{

							ArrayList<Integer> legalMoves = ((GameEnvironment)env).game.legalMoves(((GameEnvironment.State)obs).gameState);

							int[] logitRequest = {logits};
							HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
							
							double[] rawInput = obs.getDouble1();
							double[][] gameUnrolled = new double[1][];
							gameUnrolled[0] = rawInput;
							dict.put(pInput, new Tensor(gameUnrolled, inputDimensions));
							Tensor[] graphOut = graph.runGraph(logitRequest, dict);
							double[] networkOutput = ((double[][])graphOut[0].getObject())[0];
							int maxChoice = -1;

							double maxValue = 0;
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

							double randomChoice = Math.random();
							for(action = 0; action < networkOutput.length; action++){
								if(legalMoves.contains(action)){
									randomChoice -= networkOutput[action];
									if(randomChoice <= 0){
										break;
									}
								}	
							}
						}
					}else{
						if(Math.random() < e_greedy){
							// Take random action
							action = (int)(Math.random() * outputSize);
						}else{
							// Calculate action
							final double[] rawInput = obs.getDouble1();
							HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
							dict.put(pInput, new Tensor(inputDimensions, new InitOp(){
								public double execute(int[] dimensions, Index index){
									return rawInput[index.getValues()[1]];
								}
							}));
							Tensor[] graphOutput = graph.runGraph(runRequests, dict);
							double[][] rawOutput = (double[][])(graphOutput[0].getObject());
							// action = argmax(rawOutput[0]);
							double randChoice = Math.random();
							for(action = -1; randChoice > 0;){
								randChoice -= rawOutput[0][++action];
							}
						}	
					}

					

					// actionsSelected[action]++;

					// Take action, get reward
					Info a = env.createAction();
					a.getInt1()[0] = action;
					ROF rof = env.step(a);
					finished = rof.finished;

					observations.add(obs);
					actions.add(action);
					rewards.add(rof.reward);

					episodeReward += rof.reward;

					//System.out.println(obs.getDouble1()[0] + ":" + obs.getDouble1()[1] + ":" + obs.getDouble1()[2] + ":" + obs.getDouble1()[3]);
					// Update with next experience
					obs = (Info)(rof.observation);
				}

				performance += episodeReward;

				// Get discounted rewards
				double currentReward = 0;
				for(int j = rewards.size() - 1; j >= 0; j--){
					currentReward = (currentReward * rewardDiscount) + rewards.get(j);
					rewards.set(j, currentReward);
				}

				normalize(rewards);

				for(double r: rewards){
					discountedRewards.add(r);
					//discountedRewards.add(episodeReward);
				}
			}

			// System.out.println(actionsSelected[0] - actionsSelected[1]);

			performance /= (double)episodesPerEpoch;
			if(epoch == 1){
				runningPerformance = performance;
			}else if(epoch < 10){
				runningPerformance = runningPerformance * 0.9 + performance * 0.1;
			}else{
				runningPerformance = runningPerformance * 0.95 + performance * 0.05;
			}
			System.out.println("Steps: " + observations.size() + ", e_greedy: " + (int)(e_greedy*100) + ", Avg Performance(" + episodesPerEpoch + "): " + performance + ", Running Performance: " + runningPerformance);

			// normalize(discountedRewards);

			Tensor pInputTensor = new Tensor(inputDimensions);
			Tensor pLabelTensor = new Tensor(labelDimensions);
			Tensor pRewardTensor = new Tensor(rewardDimensions);

			// Iterate through saved experiences
			for(int j = 0; j < (observations.size() / batchSize) * expSampleRate; j++){
				HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();

				double[][] rawInputs = (double[][])(pInputTensor.getObject());
				double[] rawActions = (double[])(pLabelTensor.getObject());
				double[] rawRewards = (double[])(pRewardTensor.getObject());

				for(int i = 0; i < batchSize; i++){
					int sample = (int)(Math.random() * observations.size());
					rawInputs[i] = observations.get(sample).getDouble1();
					rawActions[i] = actions.get(sample);
					rawRewards[i] = discountedRewards.get(sample);
				}

				dict.put(pInput, pInputTensor);
				dict.put(pLabels, pLabelTensor);
				dict.put(pRewards, pRewardTensor);

				graph.runGraph(trainRequests, dict);
			}
		}
	}

	public void normalize(ArrayList<Double> nums){
		double sum = 0;
		for(int j = 0; j < nums.size(); j++){
			sum += nums.get(j);
		}
		double mean = sum/nums.size();
		sum = 0;
		for(int j = 0; j < nums.size(); j++){
			sum += Math.pow(nums.get(j) - mean, 2);
		}
		double stddev = Math.sqrt(sum/nums.size());
		if(stddev == 0){
			stddev = 1;
		}
		for(int j = 0; j < nums.size(); j++){
			nums.set(j, (nums.get(j) - mean) / stddev);
		}
		// ArrayList<Double> numsCopy = (ArrayList<Double>)(nums.clone());
		// Collections.sort(numsCopy);
		// System.out.println(numsCopy.get(0) + " : " +  numsCopy.get(numsCopy.size() - 1));
		// int c = 0;
		// for(Double d: numsCopy){
		// 	System.out.println(c++ + ": " + d);
		// }
	}

	public int argmax(double[] nums){
		double high = nums[0];
		int max = 0;
		for(int j = 1; j < nums.length; j++){
			if(nums[j] > high){
				max = j;
				high = nums[j];
			}
		}
		return max;
	}

}
