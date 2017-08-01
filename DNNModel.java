import jTensor.*;

import java.util.HashMap;

public class DNNModel extends Model{

	public DNNModel(String modelDir, HashMap<String, Object> params){
		super(modelDir, params);
	}

	/*
	Params: "argname": type (default)
		"layerSize": int[] ({256})
		"inputSize": int (required)
		"batchSize": int (required)
		"classes": int (required)
		"activation": String, one of: ["SIGMOID", "RELU"] ("RELU")
		"l2": boolean (false)
	*/

	public Model.ModelOps createModel(HashMap<String, Object> params){
		int[] layerSize = {256};
		if(params.containsKey("layerSize")){
			layerSize = (int[])(params.get("layerSize"));
		}
		int inputSize = -1;
		if(params.containsKey("inputSize")){
			inputSize = (int)(params.get("inputSize"));
		}
		int batchSize = -1;
		if(params.containsKey("batchSize")){
			batchSize = (int)(params.get("batchSize"));
		}
		int outputNodes = -1;
		if(params.containsKey("classes")){
			outputNodes = (int)(params.get("classes"));
		}
		String activation = "RELU";
		if(params.containsKey("activation")){
			activation = (String)(params.get("activation"));
		}
		boolean useL2 = false;
		if(params.containsKey("l2")){
			useL2 = (boolean)(params.get("l2"));
		}
		
		Graph graph = new Graph();
		
		int[] inputDimensions = {batchSize, inputSize};
		int[] labelDimensions = {batchSize};

		int pInput = graph.createPlaceholder(inputDimensions);
		int pLabels = graph.createPlaceholder(labelDimensions);

		int currentLayer = pInput;
		int lastNodeSize = inputDimensions[1];
		for(int j = 0; j < layerSize.length + 1; j++){
			int outputSize = j == layerSize.length ? outputNodes : layerSize[j];
			int[] weightDimensions = {lastNodeSize, outputSize};
			int[] biasDimensions = {outputSize};
			int vWeights = graph.createVariable("fcw"+j, weightDimensions);
			int vBias = graph.createVariable("fcb"+j, biasDimensions);
			int tMult = graph.addOp(new Operations.MatMult(), currentLayer, vWeights);
			currentLayer = graph.addOp(new Operations.TensorAdd(), tMult, vBias);
			if(j < layerSize.length){
				if(activation.toLowerCase().equals("sigmoid")){
					currentLayer = graph.addOp(new Operations.TensorSigmoid(), currentLayer);					
				}else if(activation.toLowerCase().equals("relu")){
					currentLayer = graph.addOp(new Operations.TensorReLU(), currentLayer);					
				}else{
					System.out.println("Unknown activation!");
					currentLayer = -1; // sabotage model for using bad param
				}
			}
			lastNodeSize = outputSize;
		}

		int logits = currentLayer;
		
		int prediction = graph.addOp(new Operations.MatArgmax(), logits);
		int hits = graph.addOp(new Operations.TensorEquals(), prediction, pLabels);
		int accuracy = graph.addOp(new Operations.VecAvg(), hits);
		int xEntropy = graph.addOp(new Operations.SparseCrossEntropySoftmax(), logits, pLabels);
		int loss = graph.addOp(new Operations.VecAvg(), xEntropy);

		graph.initializeVariables();
				
		if(useL2){
			int[] l2BetaSize = {1};
			int paramCount = graph.getParameterCount();
			double[] l2BetaValue = {10.0/paramCount};
			System.out.println("L2 Beta: " + l2BetaValue[0]);
			int l2Beta = graph.createConstantNode(new Tensor(l2BetaValue, l2BetaSize));

			int[] varIds = graph.getVariableIds();
			int totalVarSum = -1;
			for(int j = 0; j < varIds.length; j++){
				int varSquare = graph.addOp(new Operations.TensorSquare(), varIds[j]);
				int varSum = graph.addOp(new Operations.TensorSum(), varSquare);
				if(totalVarSum == -1){
					totalVarSum = varSum;
				}else{
					totalVarSum = graph.addOp(new Operations.TensorAdd(), totalVarSum, varSum);
				}
			}
			totalVarSum = graph.addOp(new Operations.TensorScale(), totalVarSum, l2Beta);
			loss = graph.addOp(new Operations.TensorAdd(), loss, totalVarSum);
		}

		int train = graph.trainMinimizer(loss, new AdamMinimizerNode());

		int[][] inputs = {{pInput, pLabels}, {pInput, pLabels}, {pInput}};
		
		int[] predictions = {prediction};


		return new Model.ModelOps(graph, inputs, loss, train, accuracy, predictions);
	}
}
