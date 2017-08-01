import jTensor.*;
import jREC.*;
import jGame.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class QAgentTable{

	final static String loadFile = "conv_go_9";
	final static String saveFile = "reinforce_go_9";
	
	public static void main(String[] args){
		// Game go = new Go(9);
		Game game = new TicTacToe();
		// new QAgentTable(new GameEnvironment(go, new Go9Bot(go, VariableState.readFromFile(loadFile))), 128);
		new QAgentTable(new GameEnvironment(game, new RandomPlayer(game)), 100);
		// new QAgentTable(new Rabbit(), 16);
	}

	private double getStateValue(double[] qValues, double[] state){
		int placeValue = 1;
		int index = 0;
		double[] tempRow = new double[3];
		for(int j = 0; j < state.length; j += 3){
			for(int i = 0; i < 3; i++){
				tempRow[i] = state[j + i];
			}
			index += placeValue * argmax(tempRow);
			placeValue *= 3;
		}
		return qValues[index];
	}

	public QAgentTable(Environment env, int hiddenNodesCount){

		// Hyper Parameters
		final int episodesPerEpoch = 500;
		final int batchSize = 100;
		final double expSampleRate = 1;
		final double performanceGoal = 1000;
		final double rewardDiscount = .99;
		final double e_greedy_start = 1;
		final int e_greedy_decay = 20; // reciprocal of decay


		// Create the policy network

		int inputSize = env.getObservationSpace().dimensions[0];
		int outputSize = env.getActionSpace().dimensions[0];
		System.out.println("Obs Space: " + inputSize);
		System.out.println("Act Space: " + outputSize);
		
		double[] qValues = new double[19683];

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

			Tensor pState0Tensor = new Tensor(inputDimensions);
			Tensor pState1Tensor = new Tensor(inputDimensions);

			// Iterate through saved experiences
			for(int j = 0; j < (observations.size() / batchSize) * expSampleRate; j++){
				double[][] raw0Obs = (double[][])(pState0Tensor.getObject());
				double[][] raw1Obs = (double[][])(pState1Tensor.getObject());
				int[] rawActions = new int[batchSize];
				double[] rawRewards = new double[batchSize];

				for(int i = 0; i < batchSize; i++){
					int sample = (int)(Math.random() * (observations.size() - 1));
					raw0Obs[i] = observations.get(sample).getDouble1();
					raw1Obs[i] = observations.get(sample + 1).getDouble1();
					rawActions[i] = actions.get(sample);
					rawRewards[i] = discountedRewards.get(sample);
				}

				// Get s0 q-values
				HashMap<Integer, Tensor> dict0 = new HashMap<Integer, Tensor>();
				dict0.put(pInput, pState0Tensor);
				Tensor[] state0Output = graph.runGraph(runRequests, dict0);
				double[][] state0Q = (double[][])((state0Output[1]).getObject());

				//  Get s1 q-values
				HashMap<Integer, Tensor> dict1 = new HashMap<Integer, Tensor>();
				dict1.put(pInput, pState1Tensor);
				Tensor[] state1Output = graph.runGraph(runRequests, dict1);
				double[][] state1Q = (double[][])((state1Output[1]).getObject());

				// Correct s0 q-value for action we took
				for(int i = 0; i < batchSize; i++){
					state0Q[i][rawActions[i]] = rawRewards[i] + rewardDiscount * max(state1Q[i]);
				}

				// Train
				dict0 = new HashMap<Integer, Tensor>();
				dict0.put(pInput, pState0Tensor);
				dict0.put(pQValues, state0Output[1]);
				graph.runGraph(trainRequests, dict0);
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

	public double max(double[] nums){
		double high = nums[0];
		for(int j = 1; j < nums.length; j++){
			if(nums[j] > high){
				high = nums[j];
			}
		}
		return high;
	}

}
