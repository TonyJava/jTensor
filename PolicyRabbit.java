import jTensor.*;
import jREC.*;

import java.util.ArrayList;
import java.util.HashMap;

public class PolicyRabbit{
	
	public static void main(String[] args){
		new PolicyRabbit();
	}

	public PolicyRabbit(){

		// Params

		final int episodesPerBatch = 10000;
		final double performanceGoal = 1000;
		final double rewardDiscount = .99;
		final double learningRate = .00001;


		// Create the policy network

		int trainingExamplesPerRun = 1;
		int outputSize = 3;
		int[] inputDimensions = {trainingExamplesPerRun, 4};
		// int[] labelDimensions = {trainingExamplesPerRun}; // posision of 1 in one-hot vector for 1 training example
		int[] labelDimensions = {trainingExamplesPerRun}; // index of the one in the one hot encoded action
		int[] rewardDimensions = {trainingExamplesPerRun}; // The reward for the action
		int[] hiddenNodes = {16};
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
		int y = graph.addOp(new Operations.MatSoftmax(), tNet2);

		int xEntropy = graph.addOp(new Operations.SparseCrossEntropySoftmax(), tNet2, pLabels);
		int loss = graph.addOp(new Operations.TensorScale(), xEntropy, pRewards);


		int[] runRequests = {y};
		int[] trainRequests = {loss};

		graph.initializeVariablesUniformRange(-2, 2);

		// Set up for policy gradient

		Rabbit env = new Rabbit();
		int epoch = 0;

		double performance = 0;
		while(performance < performanceGoal){

			epoch += 1;

			ArrayList<Rabbit.RabbitObservation> observations = new ArrayList<Rabbit.RabbitObservation>();
			ArrayList<Integer> actions = new ArrayList<Integer>();
			ArrayList<Double> discountedRewards = new ArrayList<Double>();

			// Perform policy rollouts
			double e_greedy = 1.0/(epoch/4 + 1.0);
			System.out.println("Starting rollouts, e_greedy: " + e_greedy);
			performance = 0;
			for(int ep = 0; ep < episodesPerBatch; ep++){
				ArrayList<Double> rewards = new ArrayList<Double>();
				boolean finished = false;
				
				Rabbit.RabbitObservation obs = (Rabbit.RabbitObservation)env.reset();
				double episodeReward = 0;
				while(!finished){

					// Get action
					int action = 0;
					if(Math.random() < e_greedy){
						// Take random action
						action = (int)(Math.random() * 3);
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
						action = -1;
						for(; randChoice > 0;){
							randChoice -= rawOutput[0][++action];
						}
					}

					// Take action, get reward
					Rabbit.Action a = new Rabbit.Action();
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
					discountedRewards.add(currentReward);
					// discountedRewards.add(episodeReward);
				}
			}

			performance /= episodesPerBatch;
			System.out.println("Steps: " + observations.size() + ", Performance: " + performance);

			normalize(discountedRewards);

			Tensor pInputTensor = new Tensor(inputDimensions);
			Tensor pLabelTensor = new Tensor(labelDimensions);
			Tensor pRewardTensor = new Tensor(rewardDimensions);

			// Iterate through all saved experiences
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
						return action; // == index.getValues()[1] ? 1 : 0;
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
			stddev = .0001;
		}
		for(int j = 0; j < nums.size(); j++){
			nums.set(j, (nums.get(j) - mean) / stddev);
		}
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