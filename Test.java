import java.util.HashMap;

public class Test{
	public static void main(String[] args){
		new Test();
	}

	public Test(){
		Graph graph = new Graph();
		int[] inputSize = {10, 5};
		int[] labelsSize = {10, 2};
		int[] vW1Size = {5, 3};
		int[] vW2Size = {3, 2};
		int[] vB1Size = {3};
		int[] vB2Size = {2};

		int pInput = graph.createPlaceholder(inputSize);
		int pLabels = graph.createPlaceholder(labelsSize);
		int vWeights1 = graph.createVariable(vW1Size);
		int vWeights2 = graph.createVariable(vW2Size);
		int vBias1 = graph.createVariable(vB1Size);
		int vBias2 = graph.createVariable(vB2Size);

		int tMult1 = graph.opMatMult(pInput, vWeights1);
		int tNet1 = graph.opMatAddVec(tMult1, vBias1);
		int h1 = graph.opTensorSigmoid(tNet1);

		int tMult2 = graph.opMatMult(h1, vWeights2);
		int tNet2 = graph.opMatAddVec(tMult2, vBias2);
		int y = graph.opTensorSigmoid(tNet2);

		int err = graph.opMatSub(y, pLabels);
		int err_sq = graph.opTensorSquare(err);
		int err_sum = graph.opMatSumCols(err_sq);

		int out = y;
		int errSums = err_sum;

		int train = graph.trainGradientDescent(0.1, errSums);

		graph.initializeVariablesUniformRange(-1, 1);

		HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
		Tensor pInputTensor = new Tensor(inputSize, new InitOp(){
			public double execute(int[] dimensions, Index index){
				return Math.random();
			}
		});
		Tensor pLabelTensor = new Tensor(labelsSize, new InitOp(){
			public double execute(int[] dimensions, Index index){
				return Math.random();
			}
		});

		dict.put(pInput, pInputTensor);
		dict.put(pLabels, pLabelTensor);

		for(int epoch = 0; epoch < 100000; epoch++){

			int[] idRequests = {out, errSums, train, vBias1};
			

			

			Tensor[] results = graph.runGraph(idRequests, dict);
			// results[0].printTensor();
			double totalError = 0;
			for(int j = 0; j < 10; j++){
				totalError += ((double[])(results[1].getObject()))[j];
			}
			// System.out.println("vBias: ");
			// results[3].printTensor();
			System.out.println("Avg Error: " + (totalError/10.0));
		}
	}

	
}