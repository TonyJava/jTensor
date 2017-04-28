import jTensor.*;
import jREC.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class PolicyCartPole{
	
	public static void main(String[] args){
		new PolicyCartPole();
	}

	public PolicyCartPole(){

		// Params

		final int episodesPerBatch = 50000;
		final double performanceGoal = 50;
		final double rewardDiscount = .99;
		final double learningRate = .1;//.001;
		final int e_greedy_decay = 50; // reciprocal of decay


		// Create the policy network

		int trainingExamplesPerRun = 1;
		int outputSize = 2;
		int[] inputDimensions = {trainingExamplesPerRun, 4};
		int[] labelDimensions = {trainingExamplesPerRun}; // posision of 1 in one-hot vector for 1 training example
		// int[] labelDimensions = {trainingExamplesPerRun, outputSize}; // index of the one in the one hot encoded action
		int[] rewardDimensions = {trainingExamplesPerRun}; // The reward for the action
		int[] hiddenNodes = {3};
		int[] vW1Size = {inputDimensions[1], hiddenNodes[0]};
		int[] vB1Size = {vW1Size[1]};
		int[] vW2Size = {vW1Size[1], outputSize};
		int[] vB2Size = {vW2Size[1]};

		Graph graph = new Graph();
		int pInput = graph.createPlaceholder(inputDimensions);
		int pLabels = graph.createPlaceholder(labelDimensions);
		int pRewards = graph.createPlaceholder(rewardDimensions);
		int vWeights1 = graph.createVariable(vW1Size);
		int vBias1 = graph.createVariable(vB1Size);
		int vWeights2 = graph.createVariable(vW2Size);
		int vBias2 = graph.createVariable(vB2Size);
		int tMult1 = graph.addOp(new Operations.MatMult(), pInput, vWeights1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int tOut1 = graph.addOp(new Operations.TensorReLU(), tNet1);
		int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		int tNet2 = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		int actionProbs = graph.addOp(new Operations.MatSoftmax(), tNet2);
		// int logProbs = graph.addOp(new Operations.TensorLn(), actionProbs);
		// int selectedActionProb = graph.addOp(new Operations.TensorScale(), logProbs, pLabels);
		int xEntropy = graph.addOp(new Operations.SparseCrossEntropySoftmax(), tNet2, pLabels);
		int reinforceGrad = graph.addOp(new Operations.TensorScale(), xEntropy, pRewards);
		// int loss = graph.addOp(new Operations.TensorScale(), xEntropy, pRewards);

		int[][] train = graph.trainRawGradients(reinforceGrad);
		// int train = graph.trainGradientDescent(learningRate, loss);

		int[] gradientNodeIds = train[1];
		int[] gradientPlaceHolderIds = train[2];

		int[] runRequests = {actionProbs};
		// int[] trainRequests = {train};
		int[] trainRequests = {train[0][0]};

		graph.initializeVariablesUniformRange(-0.1, 0.1);

		// Set up for policy gradient

		CartPole env = new CartPole();
		int epoch = 0;

		double performance = 0;
		double runningPerformance = 0;
		while(true){

			final boolean renderScreen = performance >= performanceGoal;

			epoch += 1;

			ArrayList<CartPole.CartPoleObservation> observations = new ArrayList<CartPole.CartPoleObservation>();
			ArrayList<Integer> actions = new ArrayList<Integer>();
			ArrayList<Double> discountedRewards = new ArrayList<Double>();

			// Perform policy rollouts
			double e_greedy = 1.0/(epoch/e_greedy_decay + 1.0);
			// double e_greedy = 0;
			// if(epoch < 2){
			// 	e_greedy = 1;
			// }
			// System.out.println("Starting rollouts, e_greedy: " + e_greedy);
			performance = 0;
			int[] actionsSelected = new int[outputSize];
			for(int ep = 0; ep < episodesPerBatch; ep++){
				ArrayList<Double> rewards = new ArrayList<Double>();
				boolean finished = false;
				
				CartPole.CartPoleObservation obs = (CartPole.CartPoleObservation)env.reset();
				double episodeReward = 0;
				while(!finished){

					if(renderScreen){
						env.render();
						try{Thread.sleep(500);}catch(Exception e){}
					}

					// Get action
					int action = 0;
					if(Math.random() < e_greedy){
						// Take random action
						action = (int)(Math.random() * 2);
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

					actionsSelected[action]++;

					// Take action, get reward
					CartPole.Action a = new CartPole.Action();
					a.action = action;
					ROF rof = env.step(a);
					finished = rof.finished;

					observations.add(obs);
					actions.add(action);
					rewards.add(rof.reward);

					episodeReward += rof.reward;
				}

				performance += episodeReward;

				// Get discounted rewards
				double currentReward = 0;
				for(int j = rewards.size() - 1; j >= 0; j--){
					currentReward = (currentReward * rewardDiscount) + rewards.get(j);
					rewards.set(j, currentReward);
				}

				for(double r: rewards){
					discountedRewards.add(r);
					// discountedRewards.add(episodeReward);
				}
			}

			System.out.println(actionsSelected[0] - actionsSelected[1]);

			performance /= episodesPerBatch;
			if(epoch == 1){
				runningPerformance = performance;
			}else if(epoch < 10){
				runningPerformance = runningPerformance * 0.9 + performance * 0.1;
			}else{
				runningPerformance = runningPerformance * 0.95 + performance * 0.05;
			}
			System.out.println("Steps: " + observations.size() + ", e_greedy: " + (int)(e_greedy*100) + ", Performance: " + performance + ", Average Performance: " + runningPerformance);

			normalize(discountedRewards);

			Tensor pInputTensor = new Tensor(inputDimensions);
			Tensor pLabelTensor = new Tensor(labelDimensions);
			Tensor pRewardTensor = new Tensor(rewardDimensions);

			// Tensors to accumulate gradients
			Tensor[] gradientTensors = new Tensor[gradientNodeIds.length];
			for(int j = 0; j < gradientTensors.length; j++){
				gradientTensors[j] = new Tensor(graph.getNodeDimensions(gradientNodeIds[j]));
			}

			// Iterate through all saved experiences
			// Accumulate gradients
			for(int j = 0; j < observations.size(); j++){
				HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();

				final double[] rawInput = observations.get(j).getDouble1();
				pInputTensor.operate(new CopyOp(){
					public double execute(double value, Index index){
						return rawInput[index.getValues()[1]];
					}
				});
				dict.put(pInput, pInputTensor);

				final int action = actions.get(j);
				pLabelTensor.operate(new CopyOp(){
					public double execute(double value, Index index){
						return action;// == index.getValues()[1] ? 1 : 0;
					}
				});
				dict.put(pLabels, pLabelTensor);

				final double reward = discountedRewards.get(j);
				pRewardTensor.operate(new CopyOp(){
					public double execute(double value, Index index){
						return reward;
					}
				});
				dict.put(pRewards, pRewardTensor);

				final Tensor[] graphOutput = graph.runGraph(gradientNodeIds, dict);
				for(int i = 0; i < gradientTensors.length; i++){
					final Tensor graphTensor = graphOutput[i];
					gradientTensors[i].operate(new CopyOp(){
						public double execute(double value, Index index){
							return value + graphTensor.getValue(index);
						}
					});
				}
			}

			final double modifier = learningRate / observations.size();
			for(int i = 0; i < gradientTensors.length; i++){
				gradientTensors[i].operate(new CopyOp(){
					public double execute(double value, Index index){
						return value * modifier;
					}
				});
				// System.out.println(i + ": " + gradientTensors[i].getAverage() + ", " + gradientTensors[i].getAverageMagnitude());
			}


			// Apply graidents
			HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
			for(int j = 0; j < gradientPlaceHolderIds.length; j++){
				dict.put(gradientPlaceHolderIds[j], gradientTensors[j]);
			}
			graph.runGraph(trainRequests, dict);


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