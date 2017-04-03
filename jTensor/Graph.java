package jTensor;

import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Stack;

public class Graph{

	public static boolean DEBUG_ON = false;
	public static boolean DEBUG_VERBOSE = false;

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

	public void printIdNames(){
		for(int j = 0; j < idCounter; j++){
			Node n = idNodeTable.get(j);
			String opName = n instanceof OpNode && ((OpNode)n).getOperation() != null ? ((OpNode)n).getOperation().getClass().getSimpleName() : "";
			System.out.println("Node " + j + ": " + n.getClass().getSimpleName() + " " + opName);
		}
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
					neededNodes.add((OpNode)currentNode);
				}
			}
		}

		// System.out.println("neededNodes: " + neededNodes.size());

		// reset opNodes
		// Iterating over hashset = Bad
		for(OpNode node: neededNodes){
			node.reset();
		}

		// run endNodes
		for(int id: idRequests){
			Node node = idNodeTable.get(id);
			// if(node instanceof OpNode){
				// OpNode opNode = (OpNode)node;
			if(!node.runNode()){
				System.out.println("Error run graph id: " + id);
			}
			// }
		}

		// return tensors
		Tensor[] tensors = new Tensor[idRequests.length];
		for(int j = 0; j < tensors.length; j++){
			tensors[j] = idNodeTable.get(idRequests[j]).getTensor();
		}

		return tensors;

	}

	public void initializeVariablesUniformRange(double min, double max){
		for(VariableNode node: variableNodes){
			node.initializeUniformRange(min, max);
		}
	}

	public void setVariable(int id, Tensor tensor){
		Node node = idNodeTable.get(id);
		node.setTensor(tensor);
	}

	public int trainGradientDescent(double learningRate, int target){
		int trainId = getNextId();
		TrainingNode trainingNode = new GradientDescentNode(trainId, learningRate);
		idNodeTable.put(trainId, trainingNode);

		initTraining(target, trainingNode);

		return trainId;
	}

	public int trainMomentumMinimizer(double learningRate, double momentumRate, int target){
		int trainId = getNextId();
		TrainingNode trainingNode = new MomentumMinimizerNode(trainId, learningRate, momentumRate);
		idNodeTable.put(trainId, trainingNode);

		initTraining(target, trainingNode);

		return trainId;
	}

	public void initTraining(int target, TrainingNode trainingNode){

		

		OpNode targetNode = (OpNode)idNodeTable.get(target);

		// Find path from target to all variables

		// Create constant 1
		Tensor tConst = new Tensor(targetNode.getDimensions(), new InitOp(){
			public double execute(int[] dimensions, Index index){
				return 1;
			}
		});

		int varId = getNextId();
		ConstantNode constantNode = new ConstantNode(varId, tConst);
		idNodeTable.put(varId, constantNode);

		// Create gradient opNodes
		Stack<OpNode> branchNodes = new Stack<OpNode>();
		Stack<Node> branchGradientNodes = new Stack<Node>();
		branchNodes.push(targetNode);
		branchGradientNodes.push(constantNode);
		while(!branchNodes.isEmpty()){
			OpNode currentNode = branchNodes.pop();
			Node lastGradientNode = branchGradientNodes.pop();

			ArrayList<Node> currentNodeInputs = currentNode.getInputs();
			for(int j = 0; j < currentNodeInputs.size(); j++){
				Node currentInput = currentNodeInputs.get(j);


				if(currentInput instanceof PlaceHolderNode){
					continue;
				}



				TensorOperation.TensorDerivativeInfo derivOpInfo = currentNode.getOperation().getDerivative(j);

				ArrayList<Node> inputsToNewNode = new ArrayList<Node>();
				inputsToNewNode.add(lastGradientNode); // add gradients as first input always
				for(Integer inputNum: derivOpInfo.inputsNeeded){
					inputsToNewNode.add(currentNodeInputs.get(inputNum));
				}

				int[][] inputSizes = new int[inputsToNewNode.size()][];
				for(int i = 0; i < inputsToNewNode.size(); i++){
					Node currentNewNodeInput = inputsToNewNode.get(i);
					inputSizes[i] = currentNewNodeInput.getDimensions();
				}

				int[] newNodeDimensions = derivOpInfo.op.getOutputDimensions(inputSizes);

				int newId = getNextId();
				OpNode newNode = new OpNode(newId, newNodeDimensions, derivOpInfo.op);
				idNodeTable.put(newId, newNode);

				for(Node ni: inputsToNewNode){
					newNode.addInput(ni);
				}

				if(currentInput instanceof OpNode){
					branchNodes.push((OpNode)currentInput);
					branchGradientNodes.push(newNode);
				}else if(currentInput instanceof VariableNode){
					// Add Link
					trainingNode.addInputUpdateTarget(newNode, (VariableNode)currentInput);
				}

			}

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

	public int addOp(TensorOperation op, int... inputIds){
		Node[] inputNodes = new Node[inputIds.length];
		int[][] inputDimensions = new int[inputIds.length][];
		for(int j = 0; j < inputIds.length; j++){
			inputNodes[j] = idNodeTable.get(inputIds[j]);
			inputDimensions[j] = inputNodes[j].getDimensions();
		}
		int[] outputDimensions = op.getOutputDimensions(inputDimensions);
		if(outputDimensions == null){
			System.out.println("Error: Incorrect Tensor Size. Input Ids: ");
			for(int id: inputIds){System.out.print(id + ", ");}
		}
		int id = getNextId();
		OpNode outputNode = new OpNode(id, outputDimensions, op);
		outputNode.addInput(inputNodes);
		idNodeTable.put(id, outputNode);

		return id;
	}

	// public int opMatMult(int t1, int t2){
	// 	Node node1 = idNodeTable.get(t1);
	// 	Node node2 = idNodeTable.get(t2);
	// 	int[] node1Dimensions = node1.getDimensions();
	// 	int[] node2Dimensions = node2.getDimensions();

	// 	if(node1Dimensions.length != 2 || node2Dimensions.length != 2 || node1Dimensions[1] != node2Dimensions[0]){
	// 		System.out.println("OpError: MatMult input sizes");
	// 	}

	// 	int[] nodeOutDimensions = {node1Dimensions[0], node2Dimensions[1]};
	// 	int id = getNextId();
	// 	OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatMul());
	// 	outputNode.addInput(node1, node2);
	// 	idNodeTable.put(id, outputNode);

	// 	return id;
	// }



	// public int opMatAddVec(int t1, int t2){
	// 	Node node1 = idNodeTable.get(t1);
	// 	Node node2 = idNodeTable.get(t2);
	// 	int[] node1Dimensions = node1.getDimensions();
	// 	int[] node2Dimensions = node2.getDimensions();

	// 	if(node1Dimensions.length != 2 || node2Dimensions.length != 1 || node1Dimensions[1] != node2Dimensions[0]){
	// 		System.out.println("OpError: MatAddVec input sizes");
	// 	}

	// 	int[] nodeOutDimensions = {node1Dimensions[0], node1Dimensions[1]};
	// 	int id = getNextId();
	// 	OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatAddVec());
	// 	outputNode.addInput(node1, node2);
	// 	idNodeTable.put(id, outputNode);
		
	// 	return id;
	// }

	// public int opMatSub(int t1, int t2){
	// 	Node node1 = idNodeTable.get(t1);
	// 	Node node2 = idNodeTable.get(t2);
	// 	int[] node1Dimensions = node1.getDimensions();
	// 	int[] node2Dimensions = node2.getDimensions();

	// 	if(node1Dimensions.length != 2 || node2Dimensions.length != 2 || node1Dimensions[0] != node2Dimensions[0] || node1Dimensions[1] != node2Dimensions[1]){
	// 		System.out.println("OpError: MatSub input sizes");
	// 	}

	// 	int[] nodeOutDimensions = {node1Dimensions[0], node1Dimensions[1]};
	// 	int id = getNextId();
	// 	OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatSub());
	// 	outputNode.addInput(node1, node2);
	// 	idNodeTable.put(id, outputNode);
		
	// 	return id;
	// }

	// public int opTensorSigmoid(int t1){
	// 	Node node1 = idNodeTable.get(t1);
	// 	int[] node1Dimensions = node1.getDimensions();
	// 	int[] nodeOutDimensions = node1Dimensions;
	// 	int id = getNextId();
	// 	OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.TensorSigmoid());
	// 	outputNode.addInput(node1);
	// 	idNodeTable.put(id, outputNode);
		
	// 	return id;
	// }

	// public int opTensorReLU(int t1){
	// 	Node node1 = idNodeTable.get(t1);
	// 	int[] node1Dimensions = node1.getDimensions();
	// 	int[] nodeOutDimensions = node1Dimensions;
	// 	int id = getNextId();
	// 	OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.TensorReLU());
	// 	outputNode.addInput(node1);
	// 	idNodeTable.put(id, outputNode);
		
	// 	return id;
	// }

	// public int opTensorSquare(int t1){
	// 	Node node1 = idNodeTable.get(t1);
	// 	int[] node1Dimensions = node1.getDimensions();
	// 	int[] nodeOutDimensions = node1Dimensions;
	// 	int id = getNextId();
	// 	OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.TensorSquare());
	// 	outputNode.addInput(node1);
	// 	idNodeTable.put(id, outputNode);
		
	// 	return id;
	// }

	// public int opMatSumCols(int t1){
	// 	Node node1 = idNodeTable.get(t1);
	// 	int[] node1Dimensions = node1.getDimensions();

	// 	if(node1Dimensions.length != 2){
	// 		System.out.println("OpError: MatAddVec input sizes");
	// 	}

	// 	int[] nodeOutDimensions = {node1Dimensions[0]};
	// 	int id = getNextId();
	// 	OpNode outputNode = new OpNode(id, nodeOutDimensions, new Operations.MatSumCols());
	// 	outputNode.addInput(node1);
	// 	idNodeTable.put(id, outputNode);
		
	// 	return id;
	// }



}