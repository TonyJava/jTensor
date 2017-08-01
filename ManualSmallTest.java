import jTensor.*;

import java.util.HashMap;

public class ManualSmallTest{
	public static void main(String[] args){
		new ManualSmallTest();
	}

	public ManualSmallTest(){
		Graph graph = new Graph();
		int[] inputSize = {2, 4};
		int[] labelsSize = {2, 2};
		
		int[] vW1Size = {4, 2};
		int[] vW2Size = {3, 2};
		int[] vB1Size = {2};
		int[] vB2Size = {2};

		int pInput = graph.createPlaceholder(inputSize);
		int pLabels = graph.createPlaceholder(labelsSize);
		int vWeights1 = graph.createVariable(vW1Size);
		int vWeights2 = graph.createVariable(vW2Size);
		int vBias1 = graph.createVariable(vB1Size);
		int vBias2 = graph.createVariable(vB2Size);

		int tMult1 = graph.addOp(new Operations.MatMult(), pInput, vWeights1);
		int y = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		// int y = graph.addOp(new Operations.TensorSigmoid(), tNet1);

		// int tMult2 = graph.addOp(new Operations.MatMult(), h1, vWeights2);
		// int tNet2 = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		// int y = graph.addOp(new Operations.TensorSigmoid(), tNet2);

		int err = graph.addOp(new Operations.MatSub(), y, pLabels);
		int err_sq = graph.addOp(new Operations.TensorSquare(), err);
		int err_sum = graph.addOp(new Operations.MatSumCols(), err_sq);

		int out = y;
		int errSums = err_sum;

		int train = graph.trainMinimizer(errSums, new GradientDescentNode(.1));

		// double[][] weights1 = new double[4][2];
		// weights1[0][0] = .1;
		// weights1[0][1] = .2;
		// // weights1[0][2] = .2;
		// weights1[1][0] = .3;
		// weights1[1][1] = .4;
		// // weights1[1][1] = .4;
		// weights1[2][0] = .5;
		// weights1[2][1] = .6;
		// // weights1[2][1] = .6;
		// weights1[3][0] = .7;
		// weights1[3][1] = .8;
		// // weights1[3][1] = .8;
		// Tensor pWeight1Tensor = new Tensor(weights1, vW1Size);

		// double[] bias1 = new double[2];
		// bias1[0] = .1;
		// bias1[1] = .2;
		// Tensor pBias1Tensor = new Tensor(bias1, vB1Size);


		// double[][] weights2 = new double[2][1];
		// weights2[0][0] = .4;
		// weights2[1][0] = .2;
		// Tensor pWeight2Tensor = new Tensor(weights2, vW2Size);

		// double[] bias2 = new double[1];
		// bias2[0] = .1;
		// Tensor pBias2Tensor = new Tensor(bias2, vB2Size);

		graph.initializeVariables();
		// graph.setVariable(vWeights1, pWeight1Tensor);
		// graph.setVariable(vWeights2, pWeight2Tensor);
		// graph.setVariable(vBias1, pBias1Tensor);
		// graph.setVariable(vBias2, pBias2Tensor);



		HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();

		double[][] inputs = new double[2][4];
		inputs[0][0] = -.5;
		inputs[0][1] = 1;
		inputs[0][2] = -.9;
		inputs[0][3] = .7;
		inputs[1][0] = .3;
		inputs[1][1] = .2;
		inputs[1][2] = .7;
		inputs[1][3] = .1;
		Tensor pInputTensor = new Tensor(inputs, inputSize);

		double[][] labels = new double[2][2];
		labels[0][0] = 1;
		labels[0][1] = -1;
		labels[1][0] = 2.5;
		labels[1][1] = .3;
		Tensor pLabelTensor = new Tensor(labels, labelsSize);
		
		dict.put(pInput, pInputTensor);
		dict.put(pLabels, pLabelTensor);

		for(int epoch = 0; epoch < 10; epoch++){

			int[] idRequests = {out, errSums, train};
			

			Tensor[] results = graph.runGraph(idRequests, dict);
			// results[0].printTensor();
			double totalError = 0;
			for(int j = 0; j < 1; j++){
				totalError += ((double[])(results[1].getObject()))[j];
			}
			// System.out.println("vBias: ");
			// results[3].printTensor();
			// System.out.println("vWeights: ");
			// results[4].printTensor();
			System.out.println("Avg Error: " + (totalError/1));
		}
	}

	
}