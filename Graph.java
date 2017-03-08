import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Stack;

public class Graph{
	private static int idCounter = 0;
	private HashMap<Integer, Node> idNodeTable;
	private ArrayList<VariableNode> variableNodes;

	private int getNextId(){
		int nextId = idCounter;
		idCounter += 1;
		return nextId;
	}

	public Graph(){
		variableNodes = new ArrayList<VariableNode>();
		idNodeTable = new HashMap<Integer, Node>();
	}

	public Tensor[] runGraph(int[] idRequests, HashMap<Integer, Tensor> placeHolderValues){

		HashSet<OpNode> neededNodes = new HashSet<OpNode>();

		// find and fill placeholders
		for(int id: idRequests){
			Node node = idNodeTable.get(id);
			if(node instanceof OpNode){
				Stack<OpNode> nodeStack = new Stack<OpNode>();
				nodeStack.push((OpNode)node);
				while(!nodeStack.isEmpty()){
					OpNode currentNode = nodeStack.pop();
					for(Node child: currentNode.getInputs()){
						if(child instanceof OpNode){
							nodeStack.push((OpNode)child);
						}else if(child instanceof PlaceHolderNode){
							Tensor t = placeHolderValues.get(child.getId());
							((PlaceHolderNode)child).setPlaceHolder(t);
						}
					}
					neededNodes.add((OpNode)node);
				}
			}
		}

		// reset opNodes
		// Iterating over hashset = Bad
		for(OpNode node: neededNodes){
			node.reset();
		}

		// run endNodes
		for(int id: idRequests){
			Node node = idNodeTable.get(id);
			if(node instanceof OpNode){
				OpNode opNode = (OpNode)node;
				if(!opNode.runNode()){
					System.out.println("Error run graph id: " + id);
				}
			}
		}

		// return tensors
		Tensor[] tensors = new Tensor[idRequests.length];
		for(int j = 0; j < tensors.length; j++){
			tensors[j] = idNodeTable.get(idRequests[j]).getTensor();
		}

		return tensors;

	}

	public void initializeVariablesUniformRange(int min, int max){
		for(VariableNode node: variableNodes){
			node.initializeUniformRange(min, max);
		}
	}

	// -1 as a dimension size for any amount
	public int createPlaceholder(int[] dimensions){
		int id = getNextId();
		PlaceHolderNode placeHolder = new PlaceHolderNode(id, dimensions);
		idNodeTable.put(id, placeHolder);
		return id;
	}

	public int createVariable(int[] dimensions){
		int id = getNextId();
		VariableNode variableNode = new VariableNode(id, dimensions);
		idNodeTable.put(id, variableNode);
		variableNodes.add(variableNode);
		return id;
	}

	public int opMatMult(int t1, int t2){
		Node node1 = idNodeTable.get(t1);
		Node node2 = idNodeTable.get(t2);
		int[] node1Dimensions = node1.getDimensions();
		int[] node2Dimensions = node2.getDimensions();

		if(node1Dimensions.length != 2 || node2Dimensions.length != 2 || node1Dimensions[1] != node2Dimensions[0]){
			System.out.println("OpError: MatMult input sizes");
		}

		int[] nodeOutDimensions = {node1Dimensions[0], node2Dimensions[1]};
		int id = getNextId();
		OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatMul());
		outputNode.addInput(node1, node2);
		idNodeTable.put(id, outputNode);

		return id;
	}

	public int opMatAddVec(int t1, int t2){
		Node node1 = idNodeTable.get(t1);
		Node node2 = idNodeTable.get(t2);
		int[] node1Dimensions = node1.getDimensions();
		int[] node2Dimensions = node2.getDimensions();

		if(node1Dimensions.length != 2 || node2Dimensions.length != 1 || node1Dimensions[1] != node2Dimensions[0]){
			System.out.println("OpError: MatAddVec input sizes");
		}

		int[] nodeOutDimensions = {node1Dimensions[0], node1Dimensions[1]};
		int id = getNextId();
		OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatAddVec());
		outputNode.addInput(node1, node2);
		idNodeTable.put(id, outputNode);
		
		return id;
	}

	public int opMatSub(int t1, int t2){
		Node node1 = idNodeTable.get(t1);
		Node node2 = idNodeTable.get(t2);
		int[] node1Dimensions = node1.getDimensions();
		int[] node2Dimensions = node2.getDimensions();

		if(node1Dimensions.length != 2 || node2Dimensions.length != 2 || node1Dimensions[0] != node2Dimensions[0] || node1Dimensions[1] != node2Dimensions[1]){
			System.out.println("OpError: MatSub input sizes");
		}

		int[] nodeOutDimensions = {node1Dimensions[0], node1Dimensions[1]};
		int id = getNextId();
		OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatSub());
		outputNode.addInput(node1, node2);
		idNodeTable.put(id, outputNode);
		
		return id;
	}

	public int opTensorSigmoid(int t1){
		Node node1 = idNodeTable.get(t1);
		int[] node1Dimensions = node1.getDimensions();
		int[] nodeOutDimensions = node1Dimensions;
		int id = getNextId();
		OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.TensorSigmoid());
		outputNode.addInput(node1);
		idNodeTable.put(id, outputNode);
		
		return id;
	}

	public int opTensorSquare(int t1){
		Node node1 = idNodeTable.get(t1);
		int[] node1Dimensions = node1.getDimensions();
		int[] nodeOutDimensions = node1Dimensions;
		int id = getNextId();
		OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.TensorSquare());
		outputNode.addInput(node1);
		idNodeTable.put(id, outputNode);
		
		return id;
	}

	public int opMatSumCols(int t1){
		Node node1 = idNodeTable.get(t1);
		int[] node1Dimensions = node1.getDimensions();

		if(node1Dimensions.length != 2){
			System.out.println("OpError: MatAddVec input sizes");
		}

		int[] nodeOutDimensions = {node1Dimensions[0]};
		int id = getNextId();
		OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatSumCols());
		outputNode.addInput(node1);
		idNodeTable.put(id, outputNode);
		
		return id;
	}



}