import jTensor.*;
import jREC.*;

import java.util.ArrayList;
import java.util.*;

public class EvolutionRabbit{
	
	public static void main(String[] args){
		new EvolutionRabbit();
	}

	// Params
	final double performanceGoal = 1000;

	// Policy network params
	final int trainingExamplesPerRun = 1;
	final int outputSize = 3;
	final int[] inputDimensions = {trainingExamplesPerRun, 4};
	final int[] hiddenNodes = {4};
	final int[] hiddenNodesDimensions = {trainingExamplesPerRun, hiddenNodes[0]};
	final int[] vW1Size = {inputDimensions[1], hiddenNodes[0]};
	final int[] vB1Size = {vW1Size[1]};
	final int[] vWHSize = {hiddenNodes[0], hiddenNodes[0]};
	final int[] vW2Size = {vW1Size[1], outputSize};
	final int[] vB2Size = {vW2Size[1]};

	// Evolution params
	final double initMean = 0;
	final double initStdDev = 4.0;
	final double mutateMean = 0;
	final double mutateStdDev = 1.0;
	final double mutateChance = .1;
	final int rollouts = 10;
	final int initPopSize = 10000;
	final int reproductionFactor = 4;
	final int randomOffspring = 500;

	// Add all variable sizes to this for candidate class to use to calculate total params
	// final int[][] varSizes = {vW1Size, vWHSize, vB1Size, vW2Size, vB2Size};
	final int[][] varSizes = {vW1Size, vB1Size, vW2Size, vB2Size};

	Candidate bestCandidate = null;

	double getUniformRandom(){
		return ((Math.random() - .5) * 2);
	}

	class Candidate{

		Tensor[] tensors = null;
		double score = 0;

		// Creates random individual
		public Candidate(int[][] varSizes){
			tensors = new Tensor[varSizes.length];
			for(int j = 0; j < tensors.length; j++){
				tensors[j] = new Tensor(varSizes[j], new InitOp(){
					public double execute(int[] dimensions, Index index){
						return (getUniformRandom() + initMean) * initStdDev;
					}
				});
			}
		}

		private Candidate(){
			// Intentionally blank
		}

		// creates mutated child
		public Candidate createOffspring(){
			Candidate retCandidate = new Candidate();
			retCandidate.tensors = new Tensor[tensors.length];
			for(int j = 0; j < tensors.length; j++){
				final Tensor currentTensor = tensors[j];
				retCandidate.tensors[j] = new Tensor(currentTensor.getDimensions(), new InitOp(){
					public double execute(int[] dimensions, Index index){
						return currentTensor.getValue(index) + (Math.random() < mutateChance ? ((getUniformRandom() + mutateMean) * mutateStdDev) : 0);
						// return currentTensor.getValue(index) + ((getUniformRandom() + mutateMean) * mutateStdDev);
					}
				});
			}
			return retCandidate;
		}

		// creates mutated child
		public Candidate createCrossoverOffspring(Candidate parent){
			Candidate retCandidate = new Candidate();
			retCandidate.tensors = new Tensor[tensors.length];
			for(int j = 0; j < tensors.length; j++){
				final Tensor currentTensor = tensors[j];
				final Tensor parentTensor = parent.tensors[j];
				retCandidate.tensors[j] = new Tensor(currentTensor.getDimensions(), new InitOp(){
					public double execute(int[] dimensions, Index index){
						return (Math.random() < .5 ? currentTensor.getValue(index) : parentTensor.getValue(index)) + (Math.random() < mutateChance ? ((getUniformRandom() + mutateMean) * mutateStdDev) : 0);
						// return currentTensor.getValue(index) + ((getUniformRandom() + mutateMean) * mutateStdDev);
					}
				});
			}
			return retCandidate;
		}

	}

	public EvolutionRabbit(){

		Graph graph = new Graph();
		int pInput = graph.createPlaceholder(inputDimensions);
		int vWeights1 = graph.createPlaceholder(vW1Size);
		int vBias1 = graph.createPlaceholder(vB1Size);
		int vWeights2 = graph.createPlaceholder(vW2Size);
		int vBias2 = graph.createPlaceholder(vB2Size);
		int tMult1 = graph.addOp(new Operations.MatMult(), pInput, vWeights1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int tOut1 = graph.addOp(new Operations.TensorSigmoid(), tNet1);
		int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		int tNet2 = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);

		// int pInput = graph.createPlaceholder(inputDimensions);
		// int pLastHidden = graph.createPlaceholder(hiddenNodesDimensions);
		// int vWeights1 = graph.createPlaceholder(vW1Size);
		// int vWeightsH = graph.createPlaceholder(vWHSize);
		// int vBias1 = graph.createPlaceholder(vB1Size);
		// int vWeights2 = graph.createPlaceholder(vW2Size);
		// int vBias2 = graph.createPlaceholder(vB2Size);
		// int tMult1A = graph.addOp(new Operations.MatMult(), pInput, vWeights1);
		// int tMult1H = graph.addOp(new Operations.MatMult(), pLastHidden, vWeightsH);
		// int tMult1 = graph.addOp(new Operations.TensorAdd(), tMult1A, tMult1H);
		// int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		// int tOut1 = graph.addOp(new Operations.TensorSigmoid(), tNet1);
		// int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		// int tNet2 = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		int y = graph.addOp(new Operations.MatSoftmax(), tNet2);
		// graph.setPlaceHolderInput(pLastHidden, tOut1);

		int[] runRequests = {y};

		// Set up evolution population
		ArrayList<Candidate> population = new ArrayList<Candidate>();

		for(int j = 0; j < initPopSize; j++){
			population.add(new Candidate(varSizes));
		}

		// Set up for policy gradient

		Rabbit env = new Rabbit();

		double performance = 0;
		while(true){

			boolean goal = bestCandidate != null && bestCandidate.score >= performanceGoal;

			// Perform policy rollouts
			performance = 0;
			for(int j = 0; j < population.size(); j++){

				Candidate candidate = population.get(j);

				if(goal){
					candidate = bestCandidate;
				}

				HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();

				dict.put(vWeights1, candidate.tensors[0]);
				dict.put(vBias1, candidate.tensors[1]);
				dict.put(vWeights2, candidate.tensors[2]);
				dict.put(vBias2, candidate.tensors[3]);

				// Rollouts per individual
				double individualScore = 0;	
				for(int i = 0; i < rollouts; i++){

					// Test individual
					boolean finished = false;
					Rabbit.RabbitObservation obs = (Rabbit.RabbitObservation)env.reset();
					// boolean firstStep = true;

					while(!finished){

						if(goal){
							env.render();
							try{Thread.sleep(50);}catch(Exception e){}
						}

						// Calculate action
						final double[] rawInput = obs.getDouble1();
						dict.put(pInput, new Tensor(inputDimensions, new InitOp(){
							public double execute(int[] dimensions, Index index){
								return rawInput[index.getValues()[1]];
							}
						}));
						// Set placeholders
						// dict.put(vWeights1, candidate.tensors[0]);
						// dict.put(vWeightsH, candidate.tensors[1]);
						// dict.put(vBias1, candidate.tensors[2]);
						// dict.put(vWeights2, candidate.tensors[3]);
						// dict.put(vBias2, candidate.tensors[4]);

						
						// if(firstStep){
						// 	firstStep = false;
						// 	dict.put(pLastHidden, new Tensor(hiddenNodesDimensions));
						// }

						Tensor[] graphOutput = graph.runGraph(runRequests, dict);
						double[][] rawOutput = (double[][])(graphOutput[0].getObject());
						double randChoice = Math.random();
						int action = -1;
						for(; randChoice > 0;){
							randChoice -= rawOutput[0][++action];
						}
						// System.out.println(rawOutput[0][action]);

						// Take action, get reward
						Rabbit.Action a = new Rabbit.Action();
						// a.action = action;
						a.getInt1()[0] = action;
						ROF rof = env.step(a);
						finished = rof.finished;

						individualScore += rof.reward;
					}
				}

				candidate.score = individualScore / rollouts;
				performance += candidate.score;
			}

			performance /= population.size();

			// Do evolution stuff (update population with more fit solutions)

			Collections.sort(population, new Comparator<Candidate>() {
			    public int compare(Candidate obj1, Candidate obj2) {
			        return ((Double)obj1.score).compareTo(obj2.score);
			    }
			});

			double topTenPerformance  = 0;
			for(int j = 0; j < 10; j++){
				topTenPerformance += population.get(population.size() - 1 - j).score;
			}
			topTenPerformance /= 10;

			Candidate top = population.get(population.size() - 1);
			if(bestCandidate == null || top.score > bestCandidate.score){
				bestCandidate = top;
			}

			System.out.println("\nPopulation size: " + population.size());
			System.out.println("Average Performance: " + performance);
			System.out.println("Top 10 Performance: " + topTenPerformance);
			System.out.println("Max Performance: " + top.score);

			double totalScore = 0;
			for(int j = 0; j < population.size(); j++){
				totalScore += population.get(j).score;
			}

			ArrayList<Candidate> nextPopulation = new ArrayList<Candidate>();

			int totalChosen = 0;
			for(int j = 0; j < (population.size() - randomOffspring) / (reproductionFactor + 2); j++){
				double randScore = Math.random() * totalScore / 10;
				int i;
				for(i = population.size() - 1; i > 0 && randScore > 0; i--){
					randScore -= population.get(i).score;
				}
				// totalChosen += i;
				Candidate parent1 = population.get(i);
				randScore = Math.random() * totalScore / 10;
				for(i = population.size() - 1; i > 0 && randScore > 0; i--){
					randScore -= population.get(i).score;
				}
				// totalChosen += i;
				Candidate parent2 = population.get(i);
				nextPopulation.add(parent1.createOffspring());
				nextPopulation.add(parent2.createOffspring());
				for(i = 0; i < reproductionFactor; i++){
					nextPopulation.add(parent1.createCrossoverOffspring(parent2));
				}

				// The next population is entirely children of the fittest
				// nextPopulation.add(population.get(population.size() - 1).createOffspring());
			}
			for(int j = 0; j < randomOffspring; j++){
				nextPopulation.add(new Candidate(varSizes));
			}
			System.out.println(totalChosen/nextPopulation.size());


			// for(int j = 0; j < 100; j++){
			// 	nextPopulation.add(population.get(population.size() - 1 - j));
			// }

			population = nextPopulation;

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