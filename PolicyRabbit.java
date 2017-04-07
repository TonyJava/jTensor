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

		final int episodesPerBatch = 1;
		final double performanceGoal = 1000;
		final double rewardDiscount = .99;
		final double learningRate = .1;


		// Create the policy network

		int trainingExamplesPerRun = 1;
		int[] inputDimensions = {trainingExamplesPerRun, 2};
		// int[] labelDimensions = {trainingExamplesPerRun}; // posision of 1 in one-hot vector for 1 training example
		int[] labelDimensions = {trainingExamplesPerRun, 3};
		int[] hiddenNodes = {16};
		int[] vW1Size = {inputDimensions[1], hiddenNodes[0]};
		int[] vB1Size = {vW1Size[1]};
		int[] vW2Size = {vW1Size[1], labelDimensions[0]};
		int[] vB2Size = {vW2Size[1]};

		Graph graph = new Graph();
		int pInput = graph.createPlaceholder(inputDimensions);
		int pLabels = graph.createPlaceholder(labelDimensions);
		int vWeights1 = graph.createVariable(vW1Size);
		int vBias1 = graph.createVariable(vB1Size);
		int vWeights2 = graph.createVariable(vW2Size);
		int vBias2 = graph.createVariable(vB2Size);
		int tMult1 = graph.addOp(new Operations.MatMult(), pInput, vWeights1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int tOut1 = graph.addOp(new Operations.TensorReLU(), tNet1);
		int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		int tNet2 = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		int soft = graph.addOp(new Operations.TensorSigmoid(), tNet2);
		int y = graph.addOp(new Operations.MatSoftmax(), soft);

		// Loss = ln(y * sm)
		int probMult = graph.addOp(new Operations.TensorScale(), y, pLabels);
		int loss = graph.addOp(new Operations.TensorLn(), probMult);

		// int xEntropyError = graph.addOp(new Operations.SparseCrossEntropySoftmax(), y, pLabels);
		int[][] gradientInfo = graph.trainRawGradients(loss);

		graph.initializeVariablesUniformRange(-0.1, 0.1);
		int[] runRequests = {y};
		int[] gradientRequests = new int[gradientInfo[1].length];
		int[] trainRequests = {gradientInfo[0][0]};
		 
		int startIndex = 0;
		for(int j = startIndex; j < gradientRequests.length; j++){
			gradientRequests[j] = gradientInfo[1][j - startIndex];
		}

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
			double e_greedy = 1.0/(epoch + 1);
			System.out.println("Starting rollouts, e_greedy: " + e_greedy);
			performance = 0;
			for(int ep = 0; ep < episodesPerBatch; ep++){
				ArrayList<Double> rewards = new ArrayList<Double>();
				boolean finished = false;
				
				Rabbit.RabbitObservation obs = (Rabbit.RabbitObservation)env.reset();

				while(!finished){
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
					int action = argmax(rawOutput[0]);
					if(Math.random() < e_greedy){
						action = (int)(Math.random() * 3);
					}

					// Take action, get reward
					Rabbit.Action a = new Rabbit.Action();
					a.action = action;
					ROF rof = env.step(a);
					finished = rof.finished;

					observations.add(obs);
					actions.add(action);
					rewards.add(rof.reward);

					performance += rof.reward;
				}

				// Get discounted rewards
				double currentReward = 0;
				for(int j = rewards.size() - 1; j >= 0; j--){
					currentReward = (currentReward * rewardDiscount) + rewards.get(j);
					discountedRewards.add(currentReward);
				}
			}

			performance /= episodesPerBatch;
			System.out.println("Steps: " + observations.size() + ", Performance: " + performance);

			normalize(discountedRewards);

			// Calculate scaled gradients
			System.out.println("Calculating Gradients");


			Tensor[] gradients = new Tensor[gradientInfo[1].length];
			for(int j = 0; j < gradients.length; j++){
				gradients[j] = new Tensor(graph.getNodeDimensions(gradientInfo[1][j]));
			}

			Tensor pInputTensor = new Tensor(inputDimensions);
			Tensor pLabelTensor = new Tensor(labelDimensions);

			// Iterate through all saved experiences
			for(int j = 0; j < observations.size(); j++){
				final double[] rawInput = observations.get(j).getDouble1();
				HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
				pInputTensor.operate(new CopyOp(){
					public double execute(double value, Index index){
						return rawInput[index.getValues()[1]];
					}
				});
				dict.put(pInput, pInputTensor);
				final int action = actions.get(j);
				pLabelTensor.operate(new CopyOp(){
					public double execute(double value, Index index){
						return action == index.getValues()[1] ? 1 : 0;
					}
				});
				dict.put(pLabels, pLabelTensor);
				final Tensor[] graphOutput = graph.runGraph(gradientRequests, dict);
				double advantage = discountedRewards.get(j);
				final double modifier = -1 * learningRate * advantage;
				for(int i = 0; i < graphOutput.length; i++){
					final Tensor graphOutI = graphOutput[i];
					gradients[i].operate(new CopyOp(){
						public double execute(double input, Index index){
							return input + (graphOutI.getValue(index) * modifier);
						}
					});
				}

			}

			// Scale Gradients by 1/batch_size
			final double scale = observations.size();
			for(int i = 0; i < gradients.length; i++){
				gradients[i].operate(new CopyOp(){
					public double execute(double input, Index index){
						return input / scale;
					}
				});
			}

			// Apply updates
			System.out.println("Applying gradients");

			HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();

			for(int j = 0; j < gradientInfo[2].length; j++){
				System.out.println("Gradient " + j + ": Avg=" + gradients[j].getAverage() + ", Mag=" + gradients[j].getAverageMagnitude());
				dict.put(gradientInfo[2][j], gradients[j]);
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