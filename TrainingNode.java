import java.util.ArrayList;

public class TrainingNode extends OpNode{

	private ArrayList<VariableNode> updateTargets;
	private ArrayList<Node> gradientInputs;

	public TrainingNode(int id){
		super(id, null, null);
		updateTargets = new ArrayList<VariableNode>();
		gradientInputs = new ArrayList<Node>();
		setInputs(gradientInputs);
	}

	public void addInputUpdateTarget(Node inputNode, VariableNode updateTarget){
		updateTargets.add(updateTarget);
		gradientInputs.add(inputNode);
	}

	public boolean runNode(){
		for(int j = 0; j < gradientInputs.size(); j++){
			if(!gradientInputs.get(j).runNode()){
				return false;
			}
			Tensor tVar = updateTargets.get(j).getTensor();
			final Tensor tGradients = gradientInputs.get(j).getTensor();
			Tensor tUpdatedVar = new Tensor(tVar, new CopyOp(){
				public double execute(double value, Index index){
					double sum = value;
					Index tGradIndex = new Index(tGradients.getOrder());
					for(int i = 1; i < tGradIndex.values.length; i++){
						tGradIndex.values[i] = index.values[i - 1];
					}
					for(tGradIndex.values[0] = 0; tGradIndex.values[0] < tGradients.getDimensions()[0]; tGradIndex.values[0]++){
						sum += tGradients.getValue(tGradIndex);
					}
					return sum;
				}
			});
			updateTargets.get(j).setTensor(tUpdatedVar);
		}
		return true;
	}
	
}