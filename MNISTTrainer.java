import java.io.*;
// import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

import jTensor.*;

public class MNISTTrainer{
	
	// final double INPUT_ON = .999;
	// final double INPUT_OFF = .001;

	public static void main(String[] args){
		new MNISTTrainer();
	}

	int labelNum = 50000;

	public MNISTTrainer(){

		TrainingData[] data = loadImages("./train-images.idx3-ubyte");
		final int[] answers = loadLabels("./train-labels.idx1-ubyte");

		int[] outputSize = {1};
		for(int j = 0; j < data.length; j++){
			final int answer = answers[j];
			data[j].output = new Tensor(outputSize, new InitOp(){
				public double execute(int[] dimensions, Index index){
					return answer;
					// return index.values[1] == answer ? 1 : 0;
				}
			});
			// if(j==0)data[j].output.printTensor();
		}

		
		int[] inputDimensions = {1, 28 * 28};
		int[] labelDimensions = {1};

		int[] hiddenNodes = {64};

		int[] vW1Size = {inputDimensions[1], 10};//hiddenNodes[0]};
		int[] vW2Size = {hiddenNodes[0], 10};
		int[] vB1Size = {vW1Size[1]};
		int[] vB2Size = {vW2Size[1]};

		Graph graph = new Graph();

		int pInput = graph.createPlaceholder(inputDimensions);
		int pLabels = graph.createPlaceholder(labelDimensions);
		int vWeights1 = graph.createVariable(vW1Size);
		int vWeights2 = graph.createVariable(vW2Size);
		int vBias1 = graph.createVariable(vB1Size);
		int vBias2 = graph.createVariable(vB2Size);
		int[] variables = {vWeights1, vWeights2};

		int tMult1 = graph.addOp(new Operations.MatMult(), pInput, vWeights1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int y = graph.addOp(new Operations.TensorSigmoid(), tNet1);

		// int tMult2 = graph.addOp(new Operations.MatMult(), h1, vWeights2);
		// int tNet2 = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		// int y = graph.addOp(new Operations.TensorSigmoid(), tNet2);

		// int err = graph.addOp(new Operations.MatSub(), y, pLabels);
		// int err_sq = graph.addOp(new Operations.TensorSquare(), err);
		// int xEntropyError = graph.addOp(new Operations.MatSumCols(), err_sq);
		
		int xEntropyError = graph.addOp(new Operations.SparseCrossEntropySoftmax(), y, pLabels);

		// int sumNode = -1;
		// for(int varId: variables){
		// 	int squares = graph.addOp(new Operations.TensorSquare(), varId);
		// 	int sumCols = graph.addOp(new Operations.MatSumCols(), squares);
		// 	int varSum = graph.addOp(new Operations.VecSum(), sumCols);
		// 	if(sumNode == -1){
		// 		sumNode = varSum;
		// 	}else{
		// 		sumNode = graph.addOp(new Operations.TensorAdd(), sumNode, varSum);
		// 	}
		// }

		// int errorNode = graph.addOp(new Operations.TensorAdd(), sumNode, xEntropyError);

		// int train = graph.trainMomentumMinimizer(0.1, 0.1, xEntropyError);
		int train = graph.trainGradientDescent(0.0005, xEntropyError);

		graph.printIdNames();

		graph.initializeVariablesUniformRange(-0.01, 0.01);

		int trainBatch = 1000;
		int trainingExamples = data.length - 10000;

		int[] idRequests = {y, xEntropyError, train};


		final double hitTarget = 95;
		double hitPercent = 0;
		while(true){
			double totalError = 0;
			int hits = 0;
			int trainStart = hitPercent < hitTarget ? 0 : trainingExamples;
			int trainEnd = hitPercent < hitTarget ? trainingExamples : data.length;
			int trainCount = trainEnd - trainStart;
			int batchSize = trainBatch;
			for(int j = 0; j < batchSize; j++){
				int randomSample = (int)(Math.random() * trainCount) + trainStart;
				
				HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
				dict.put(pInput, data[randomSample].input);
				dict.put(pLabels, data[randomSample].output);

				Tensor[] graphOutput = graph.runGraph(idRequests, dict);

				double currentError = ((double[])(graphOutput[1].getObject()))[0];
				int onIndex = answers[randomSample];

				double[][] networkOutput = (double[][])(graphOutput[0].getObject());


				int index = 0;
				double high = networkOutput[0][0];
				for(int i = 1; i < networkOutput[0].length; i++){
					if(networkOutput[0][i] > high){
						high = networkOutput[0][i];
						index = i;
					}
				}

				hits += index == onIndex ? 1 : 0;

				totalError += currentError;
				// System.out.println("j = " + j + ", error: " + currentError);

			}

			boolean exit = hitPercent >= hitTarget;

			double avgError = totalError/batchSize;
			hitPercent = (double)hits*100/batchSize;
			System.out.println("Avg error (" + batchSize + "): " + avgError);
			System.out.println("Hit % (" + batchSize + "): " + hitPercent);
			System.out.println("");

			if(exit){
				break;
			}

			
		}
	}

	public static class TrainingData{
		Tensor input;
		Tensor output;
	}

	public int[] loadLabels(String file){
		BufferedInputStream br = null;
		ArrayList<Integer> labels = new ArrayList<Integer>();

		try {

			br = new BufferedInputStream(new FileInputStream(file));

			byte[] buffer = new byte[4];

			// while(true){
				// System.out.println("DATA: " + br.read());
				// if(false){
					// break;
				// }
			// }

			buffer[0] = (byte)br.read(); // magic number
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();
			int intRead = 0;

			buffer[0] = (byte)br.read(); // number of labels
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();
			for(int j = 0; j < 4; j++){
				// System.out.println("byte[" + j + "]: " + buffer[j] + ":" + intRead);
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}

			// byte bb = (byte)-22 | 0;
			// System.out.println(bb);

			int labelCount = labelNum;
			// System.out.println("Labelcount: " + intRead);
			for(int j = 0; j < labelCount; j++){
				int b = br.read();
				// System.out.println("Read: " + b);
				if(b != -1){
					labels.add((int)(b));
				}else{
					System.out.println("Error2!" + b);
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("Error20: " + e);
		}

		int[] data = new int[labels.size()];

		for(int j = 0; j < data.length; j++){
			data[j] = labels.get(j);
		}

		System.out.println("Labels: " + data.length);

		return data;
	}

	// leave outputs blank
	public TrainingData[] loadImages(String file){
		BufferedInputStream br = null;
		ArrayList<double[][]> images = new ArrayList<double[][]>();

		try {

			br = new BufferedInputStream(new FileInputStream(file));


			byte[] buffer = new byte[4];

			buffer[0] = (byte)br.read(); // magic number
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();		


			buffer[0] = (byte)br.read(); // number of images
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();		
			int intRead = 0;
			for(int j = 0; j < 4; j++){
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}
			int labelCount = labelNum;


			buffer[0] = (byte)br.read(); // rows
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();				
			for(int j = 0; j < 4; j++){
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}
			int rows = intRead;

			buffer[0] = (byte)br.read(); // cols
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();
			for(int j = 0; j < 4; j++){
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}
			int cols = intRead;

			out_Label:
			for(int j = 0; j < labelCount; j++){
				// Runtime runtime = Runtime.getRuntime();
				// if(j%1000 == 0)System.out.println("Free Memory(" + j + "): " + runtime.freeMemory() / (1024*1024));

				double[][] image = new double[1][28 * 28];

				for(int i = 0; i < 28*28; i++){
					int b = br.read();
					if(b != -1){
						image[0][i] = (((double)b)/255)*.9 + .05;
						// System.out.println("IM: "+ image[i/28][i%28][0]);
					}else{
						System.out.println("Error1!");
						break out_Label;
					}
				}
				images.add(image);
			}
		} catch (Exception e) {
			System.out.println("Error10: " + e);
			e.printStackTrace();
		}

		TrainingData[] data = new TrainingData[images.size()];

		int[] inputDimensions = {1, 28*28};
		for(int j = 0; j < data.length; j++){
			data[j] = new TrainingData();
			data[j].input = new Tensor(images.get(j), inputDimensions);
		}

		System.out.println("Images: " + data.length);

		return data;
	}


}