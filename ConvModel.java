import jTensor.*;

import java.util.HashMap;

public class ConvModel extends Model{

	public ConvModel(String modelDir, HashMap<String, Object> params){
		super(modelDir, params);
	}

	/*
	Params: "argname": type (Default)
		"filters": int[] ({32})
		"pooling": int[] ({1, 1, ...})
		"fcSize": int ({256})
		"filterWidth": int[] ({5})
		"stride": int[] ({1, 1, ...})
		"learningRate": double (0.001)
		"inputWidth": int
		"inputDepth": int
		"batchSize": int
		"classes": int
		"l2": boolean (false)
	*/

	public Model.ModelOps createModel(HashMap<String, Object> params){
		int[] filters = {32};
		if(params.containsKey("filters")){
			filters = (int[])(params.get("filters"));
		}

		int[] pooling = new int[filters.length];
		for(int j = 0; j < pooling.length; j++){
			pooling[j] = 1;
		}
		if(params.containsKey("pooling")){
			pooling = (int[])(params.get("pooling"));
		}

		int fcSize = 256;
		if(params.containsKey("fcSize")){
			fcSize = (int)(params.get("fcSize"));
		}

		int[] filterWidth = {5};
		if(params.containsKey("filterWidth")){
			filterWidth = (int[])(params.get("filterWidth"));
		}

		int[] stride = new int[filters.length];
		for(int j = 0; j < stride.length; j++){
			stride[j] = 1;
		}
		if(params.containsKey("stride")){
			stride = (int[])(params.get("stride"));
		}

		double learningRate = 0.001;
		if(params.containsKey("learningRate")){
			learningRate = (double)(params.get("learningRate"));
		}

		int inputWidth = (int)(params.get("inputWidth"));
		int inputDepth = (int)(params.get("inputDepth"));
		int batchSize = (int)(params.get("batchSize"));
		int outputNodes = (int)(params.get("classes"));

		boolean useL2 = false;
		if(params.containsKey("l2")){
			useL2 = (boolean)(params.get("l2"));
		}

		int[] prePoolWidth = new int[filters.length];
		int[] filterOutWidth = new int[filters.length];
		int lastWidth = inputWidth;
		for(int j = 0; j < filters.length; j++){
			prePoolWidth[j] = (lastWidth-(filterWidth[j]-stride[j]))/stride[j];
			filterOutWidth[j] = prePoolWidth[j]/pooling[j];
			lastWidth = filterOutWidth[j];
		}
		int lastFilter = filters.length - 1;
		int hiddenNodes = filterOutWidth[lastFilter]*filterOutWidth[lastFilter]*filters[lastFilter];
		int[] reshapeSize = {batchSize, hiddenNodes};
		int[][] vFiltersSize = new int[filters.length][];
		int[][] vFiltersBiasSize = new int[filters.length][];
		int lastDepth = inputDepth;
		for(int j = 0; j < filters.length; j++){
			int[] currentFilterSize = {filterWidth[j], filterWidth[j], lastDepth, filters[j]};
			int[] currentFilterBiasSize = {prePoolWidth[j], prePoolWidth[j], filters[j]};
			lastDepth = filters[j];
			vFiltersSize[j] = currentFilterSize;
			vFiltersBiasSize[j] = currentFilterBiasSize;
		}
		int[] vFC1Size = {hiddenNodes, fcSize};
		int[] vFC1BiasSize = {fcSize};

		int[] vFC2Size = {fcSize, outputNodes};
		int[] vFC2BiasSize = {outputNodes};
		
		Graph graph = new Graph();
		
		int[] inputDimensions = {batchSize, inputWidth, inputWidth, inputDepth};
		int[] labelDimensions = {batchSize};

		int pInput = graph.createPlaceholder(inputDimensions);
		int pLabels = graph.createPlaceholder(labelDimensions);

		int currentVolume = pInput;

		Initializer initBias = new Initializer.UniformInitializer(0.05, 0.1);

		for(int j = 0; j < filters.length; j++){
			int vWeights = graph.createVariable("fw"+j, vFiltersSize[j]);
			int vBias = graph.createVariable("fb"+j, vFiltersBiasSize[j], initBias);
			int tConv = graph.addOp(new Operations.Conv2d(stride[j]), currentVolume, vWeights);
			int tNet = graph.addOp(new Operations.TensorAdd(), tConv, vBias);
			if(pooling[j] > 1){
				tNet = graph.addOp(new Operations.MaxPool2d(pooling[j], pooling[j]), tConv);
			}
			currentVolume = graph.addOp(new Operations.TensorReLU(), tNet);
		}

		int vWeights1 = graph.createVariable("fcw1", vFC1Size);
		int vBias1 = graph.createVariable("fcb1", vFC1BiasSize, initBias);
		int tUnroll1 = graph.addOp(new Operations.TensorReshape(reshapeSize), currentVolume);
		int tMult1 = graph.addOp(new Operations.MatMult(), tUnroll1, vWeights1);
		int tNet1 = graph.addOp(new Operations.MatAddVec(), tMult1, vBias1);
		int tOut1 = graph.addOp(new Operations.TensorReLU(), tNet1);

		int vWeights2 = graph.createVariable("fcw2", vFC2Size);
		int vBias2 = graph.createVariable("fcb2", vFC2BiasSize, initBias);
		int tMult2 = graph.addOp(new Operations.MatMult(), tOut1, vWeights2);
		int logits = graph.addOp(new Operations.MatAddVec(), tMult2, vBias2);
		
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

		int train = graph.trainMinimizer(loss, new AdamMinimizerNode(learningRate, .9, .999));

		int[][] inputs = {{pInput, pLabels}, {pInput, pLabels}, {pInput}};
		
		int[] predictions = {logits, prediction};


		return new Model.ModelOps(graph, inputs, loss, train, accuracy, predictions);
	}
}
