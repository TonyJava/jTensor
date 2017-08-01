package jTensor;

import java.util.HashMap;

public abstract class Model{
	
	private String modelDir;
	private ModelOps modelOps;

	public abstract ModelOps createModel(HashMap<String, Object> params);

	public Model(String modelDir, HashMap<String, Object> params){
		this.modelDir = modelDir;
		modelOps = createModel(params);
		loadState();
	}

	// Returns placeholder Ids for fit, eval, and predict.
	public int[][] getPlaceHolderIds(){
		return modelOps.inputs;
	}

	// Returns average loss
	public double fit(HashMap<Integer, Tensor> inputDict){
		int[] requests = {modelOps.loss, modelOps.train};
		Tensor[] graphOut = modelOps.graph.runGraph(requests, inputDict);
		return ((double[])(graphOut[0].getObject()))[0];
	}


	public double eval(HashMap<Integer, Tensor> inputDict){
		int[] requests = {modelOps.accuracy};
		Tensor[] graphOut = modelOps.graph.runGraph(requests, inputDict);
		return ((double[])(graphOut[0].getObject()))[0];
	}

	public Tensor[] predict(HashMap<Integer, Tensor> inputDict){
		return modelOps.graph.runGraph(modelOps.predictions, inputDict);
	}

	public void loadState(){
		System.out.println("Loading state from file: " + modelDir);
		modelOps.graph.loadVariableState(VariableState.readFromFile(modelDir));	
	}

	public void saveState(){
		System.out.println("Saving state to file: " + modelDir);
		modelOps.graph.saveVariableState().writeToFile(modelDir);
	}

	public enum Mode{
		FIT, EVAL, PREDICT
	}

	public static class ModelOps{
		Graph graph;
		int[][] inputs;	
		int loss;
		int train;
		int accuracy;
		int[] predictions;

		public ModelOps(Graph graph, int[][] inputs, int loss, int train, int accuracy, int[] predictions){
			this.graph = graph;
			this.inputs = inputs;
			this.loss = loss;
			this.train = train;
			this.accuracy = accuracy;
			this.predictions = predictions;
		}
	}

}
