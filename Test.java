import java.util.HashMap;

public class Test{
	public static void main(String[] args){
		new Test();
	}

	public Test(){
		Graph graph = new Graph();
		int[] inputSize = {10, 3};
		int[] labelsSize = {10, 1};
		int[] vWeightsSize = {3, 1};
		int[] vBiasSize = {1};

		int pInput = graph.createPlaceholder(inputSize);
		int pLabels = graph.createPlaceholder(labelsSize);
		int vWeights = graph.createVariable(vWeightsSize);
		int vBias = graph.createVariable(vBiasSize);
		int tMult = graph.opMatMult(pInput, vWeights);
		int tAdjusted = graph.opMatAddVec(tMult, vBias);
		int y = graph.opTensorSigmoid(tAdjusted);

		int err = graph.opMatSub(pLabels, y);
		int err_sq = graph.opTensorSquare(err);
		int err_sum = graph.opMatSumCols(err_sq);

		int out = y;
		int errSums = err_sum;

		graph.initializeVariablesUniformRange(-1, 1);

		int[] idRequests = {out, errSums};
		HashMap<Integer, Tensor> dict = new HashMap<Integer, Tensor>();
		Tensor pInputTensor = new Tensor(inputSize, new InitOp(){
			public double execute(int[] dimensions, int[] index){
				return (index[0]%2 == 0 ? -1 : 1) * 2.0 / (index[1] + Math.random());
			}
		});
		Tensor pLabelTensor = new Tensor(labelsSize, new InitOp(){
			public double execute(int[] dimensions, int[] index){
				return (index[0]%2 == 0 ? 0 : 1);
			}
		});

		dict.put(pInput, pInputTensor);
		dict.put(pLabels, pLabelTensor);

		Tensor[] results = graph.runGraph(idRequests, dict);
		// results[0].printTensor();
		double totalError = 0;
		for(int j = 0; j < 10; j++){
			totalError += ((double[])(results[1].getObject()))[j];
		}
		System.out.println("Avg Error: " + (totalError/10));
	}

	
}