package jTensor;

import java.util.*;

public class Graph{

	public static boolean DEBUG_ON = false;
	public static boolean DEBUG_VERBOSE = false;

	private static int idCounter = 0;
	private HashMap<Integer, Node> idNodeTable;
	private ArrayList<VariableNode> variableNodes;
	private HashMap<String, VariableNode> namedVariables;

	private int getNextId(){
		int nextId = idCounter;
		idCounter += 1;
		return nextId;
	}

	public Graph(){
		variableNodes = new ArrayList<VariableNode>();
		namedVariables = new HashMap<String, VariableNode>();
		idNodeTable = new HashMap<Integer, Node>();
	}

	public int[] getNodeDimensions(int nodeId){
		return idNodeTable.get(nodeId).getDimensions();
	}

	public int[] getVariableIds(){
		int[] varIds = new int[variableNodes.size()];
		for(int j = 0; j < varIds.length; j++){
			varIds[j] = variableNodes.get(j).getId();
		}
		return varIds;
	}

	public void printIdNames(){
		for(int j = 0; j < idCounter; j++){
			Node n = idNodeTable.get(j);
			String opName = n instanceof OpNode && ((OpNode)n).getOperation() != null ? ((OpNode)n).getOperation().getClass().getSimpleName() : "";
			String dims = "[";
			int[] nDimensions = n.getDimensions();
			if(nDimensions != null){
				for(int d : nDimensions){
					dims = dims + d + ", ";
				}
				dims = dims.substring(0, dims.length() - 2) + "]";
			}else{
				dims += "]";
			}
			System.out.println("Node " + j + ": " + n.getClass().getSimpleName() + " " + opName + ", " + dims);
		}
	}

	public void setPlaceHolderInput(int placeHolderId, int inputId){
		PlaceHolderNode placeHolder = (PlaceHolderNode)idNodeTable.get(placeHolderId);
		Node input = (Node)idNodeTable.get(inputId);
		placeHolder.setInput(input);
	}

	private class IntegerWrapper{
		int x;
		public IntegerWrapper(int x){this.x = x;}
	}

	public int getParameterCount(){
		final IntegerWrapper count = new IntegerWrapper(0);
		for(VariableNode v: variableNodes){
			v.getTensor().operate(new CopyOp(){
				public double execute(double value, Index index){
					count.x += 1;
					return value;
				}
			});
		}
		return count.x;
	}

	public Tensor[] runGraph(int[] idRequests, HashMap<Integer, Tensor> placeHolderValues){

		ArrayList<OpNode> neededNodes = new ArrayList<OpNode>();
		ArrayList<PlaceHolderNode> placeHoldersWithInput = new ArrayList<PlaceHolderNode>();

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
							PlaceHolderNode placeHolder = ((PlaceHolderNode)child);
							if(!placeHolder.isInputSet()){
								if(t == null){
									System.out.println("Error: Missing PlaceholderId " + child.getId());
								}
								placeHolder.setPlaceHolder(t);
							}else{
								placeHoldersWithInput.add(placeHolder);
								if(t != null){
									placeHolder.setPlaceHolder(t);
								}
							}
						}
					}
					neededNodes.add((OpNode)currentNode);
				}
			}
		}

		// System.out.println("neededNodes: " + neededNodes.size());

		// reset opNodes
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

		for(PlaceHolderNode placeHolder: placeHoldersWithInput){
			placeHolder.copyFromInput();
		}

		// return tensors
		Tensor[] tensors = new Tensor[idRequests.length];
		for(int j = 0; j < tensors.length; j++){
			tensors[j] = idNodeTable.get(idRequests[j]).getTensor();
		}

		return tensors;

	}

	public void initializeVariables(){
		for(VariableNode node: variableNodes){
			node.initialize();
		}
		System.out.println("Total parameters: " + getParameterCount());
	}

	public void setVariable(int id, Tensor tensor){
		Node node = idNodeTable.get(id);
		node.setTensor(tensor);
	}

	public int trainMinimizer(int target, TrainingNode trainingNode){
		int trainId = getNextId();
		trainingNode.setId(trainId);
		idNodeTable.put(trainId, trainingNode);
		initTraining(target, trainingNode);
		return trainId;
	}

	public int[][] trainRawMinimizer(int target, TrainingNode trainingNode){
		int trainId = getNextId();
		trainingNode.setId(trainId);
		idNodeTable.put(trainId, trainingNode);
		return trainRawGradients(target, trainingNode);
	}

	// Returns int[][] = [ [trainId], [gradientNodeIds, ...], [gradientInputPlaceholderIds, ....]  ]
	private int[][] trainRawGradients(int target, TrainingNode tNode){
		TrainingNode trainingNode = tNode;
		int trainId = trainingNode.getId();
		initTraining(target, trainingNode);

		ArrayList<VariableNode> updateTargets = trainingNode.getUpdateTargets();
		ArrayList<Node> gradientInputs = trainingNode.getGradientInputs();

		int[] trainNode = {trainId};
		int[] placeHolders = new int[gradientInputs.size()];
		int[] gradientHolders = new int[gradientInputs.size()];
		for(int j = 0; j < gradientInputs.size(); j++){
			gradientHolders[j] = gradientInputs.get(j).getId();
			int[] placeHolderDimensions = gradientInputs.get(j).getDimensions();

			// Is this needed?
			if(gradientInputs.get(j) instanceof OpNode){
				OpNode opNode = ((OpNode)gradientInputs.get(j));
				ArrayList<Node> opNodeInputs = opNode.getInputs();
				int[][] inputDimensions = new int[opNodeInputs.size()][];
				for(int i = 0; i < opNodeInputs.size(); i++){
					inputDimensions[i] = opNodeInputs.get(i).getDimensions();
				}
				placeHolderDimensions = opNode.getOperation().getOutputDimensions(inputDimensions);
			}else{
				System.out.println("BADDDD#3");
			}

			placeHolders[j] = createPlaceholder(placeHolderDimensions);
			gradientInputs.set(j, idNodeTable.get(placeHolders[j]));
		}

		int[][] retArray = {trainNode, gradientHolders, placeHolders};

		return retArray;
	}

	private void initTraining(int target, TrainingNode trainingNode){

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

			int[][] currentNodeInputSizes = new int[currentNodeInputs.size()][];
			for(int j = 0; j < currentNodeInputs.size(); j++){
				Node currentInput = currentNodeInputs.get(j);
				currentNodeInputSizes[j] = currentInput.getDimensions();
			}			

			for(int j = 0; j < currentNodeInputs.size(); j++){
				Node currentInput = currentNodeInputs.get(j);

				if(currentInput instanceof PlaceHolderNode){
					continue;
				}

				TensorOperation.TensorDerivativeInfo derivOpInfo = currentNode.getOperation().getDerivative(j, currentNodeInputSizes);
				if(derivOpInfo == null){
					derivOpInfo = currentNode.getOperation().getDerivative(j);
				}

				ArrayList<Node> inputsToNewNode = new ArrayList<Node>();
				inputsToNewNode.add(lastGradientNode); // add gradients as first input always
				for(Integer inputNum: derivOpInfo.inputsNeeded){
					inputsToNewNode.add(currentNodeInputs.get(inputNum));
				}

				// int[][] inputSizes = new int[inputsToNewNode.size()][];
				// for(int i = 0; i < inputsToNewNode.size(); i++){
				// 	Node currentNewNodeInput = inputsToNewNode.get(i);
				// 	inputSizes[i] = currentNewNodeInput.getDimensions();
				// }

				// int[] newNodeDimensions = derivOpInfo.op.getOutputDimensions(inputSizes);
				int[] newNodeDimensions = currentInput.getDimensions();

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

	public VariableState saveVariableState(){
		VariableState vs = new VariableState();
		for(VariableNode variableNode : variableNodes){
			if(variableNode.name.charAt(0) != '_'){
				vs.addVariable(variableNode);
			}
		}
		return vs;
	}

	public void loadVariableState(VariableState vs){
		Iterator it = vs.map.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        String name = (String)(pair.getKey());
	        Tensor value = (Tensor)(pair.getValue());
	        namedVariables.get(name).setTensor(value);
	    }
	}

	// -1 as a dimension size for any amount
	public int createPlaceholder(int[] dimensions){
		int id = getNextId();
		PlaceHolderNode placeHolder = new PlaceHolderNode(id, dimensions);
		idNodeTable.put(id, placeHolder);
		return id;
	}

	public int createConstantNode(Tensor tensor){
		int id = getNextId();
		ConstantNode constantNode = new ConstantNode(id, tensor);
		idNodeTable.put(id, constantNode);
		return id;
	}

	// Unnamed varibles are prepended with "_" and therefore aren't saved by default
	public int createVariable(int[] dimensions){
		return createNamedVariable(null, dimensions, null);
	}

	public int createVariable(String name, int[] dimensions){
		return createNamedVariable(name, dimensions, null);
	}

	// Unnamed varibles are prepended with "_" and therefore aren't saved by default
	public int createVariable(int[] dimensions, Initializer init){
		return createNamedVariable(null, dimensions, init);
	}

	public int createVariable(String name, int[] dimensions, Initializer init){
		return createNamedVariable(name, dimensions, init);
	}

	private int createNamedVariable(String name, int[] dimensions, Initializer init){
		int id = getNextId();
		if(name == null){
			name = "_" + id;
		}
		if(init == null){
			// Default initializer uniform Glorot/Xavier inspired
			int dimensionSum = 0;
			for(int j = 0; j < dimensions.length; j++){
				dimensionSum += dimensions[j];
			}
			double range = Math.sqrt(6.0/dimensionSum)*2;
			init = new Initializer.UniformInitializer(range, 0);
		}
		VariableNode variableNode = new VariableNode(id, dimensions, name, init);
		idNodeTable.put(id, variableNode);
		namedVariables.put(name, variableNode);
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

}
