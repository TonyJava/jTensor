public class Operations{
	
	// Input: [a, b], [c, d]
	// Output: [a, d]
	public static class MatMul extends TensorOperation{

		public Tensor execute(Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			double[][] matrix2 = (double[][])(inputs[1].getObject());
			int[] outputMatrixDimensions = {matrix1.length, matrix2[0].length};
			int innerSize = matrix2.length;

			double[][] outputMatrix = new double[outputMatrixDimensions[0]][outputMatrixDimensions[1]];

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				for(int y = 0; y < outputMatrixDimensions[1]; y++){
					double sum = 0;
					for(int j = 0; j < innerSize; j++){
						sum += matrix1[x][j] * matrix2[j][y];
					}
					outputMatrix[x][y] = sum;
				}
			}

			Tensor tensor = new Tensor(outputMatrix, outputMatrixDimensions);
			return tensor;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp;
			switch(inputIndex){
				// wrt inputs
				case 0: derivativeOp = new TensorOperation(){
					public Tensor execute(Tensor... inputs){
						double[][] matrix1 = (double[][])(inputs[0].getObject()); // gradients [n, o]
						double[][] matrix2 = (double[][])(inputs[1].getObject()); // weights [i, o]
						int[] outputMatrixDimensions = {matrix1.length, matrix2.length};
						int innerSize = matrix1[0].length;
						double[][] outputMatrix = new double[outputMatrixDimensions[0]][outputMatrixDimensions[1]];
						for(int n = 0; n < outputMatrixDimensions[0]; n++){
							for(int x = 0; x < outputMatrixDimensions[1]; x++){
								double sum = 0;
								for(int j = 0; j < innerSize; j++){
									sum += matrix2[x][j] * matrix1[n][j];
								}
								outputMatrix[n][x] = sum;
							}
						}
						Tensor tensor = new Tensor(outputMatrix, outputMatrixDimensions);
						return tensor;
					}
				};break;
				// wrt weights
				default: derivativeOp = new TensorOperation(){
					public Tensor execute(Tensor... inputs){
						double[][] matrix1 = (double[][])(inputs[0].getObject()); // gradients [n, o]
						double[][] matrix2 = (double[][])(inputs[1].getObject()); // inputs [n, i]
						int[] outputMatrixDimensions = {matrix1.length, matrix2[0].length, matrix1[0].length};
						double[][][] outputMatrix = new double[outputMatrixDimensions[0]][outputMatrixDimensions[1]][outputMatrixDimensions[2]];
						for(int n = 0; n < outputMatrixDimensions[0]; n++){
							for(int x = 0; x < outputMatrixDimensions[1]; x++){
								for(int y = 0; y < outputMatrixDimensions[2]; y++){
									outputMatrix[n][x][y] = matrix2[n][x] * matrix1[n][y];
								}
							}
						}
						Tensor tensor = new Tensor(outputMatrix, outputMatrixDimensions);
						return tensor;
					}
				};
			}
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}

	}


	// Input: [a, b], [b]
	// Output: [a, b]
	public static class MatAddVec extends TensorOperation{

		public Tensor execute(Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			double[] vec = (double[])(inputs[1].getObject());
			int[] outputMatrixDimensions = {matrix1.length, matrix1[0].length};

			double[][] outputMatrix = new double[outputMatrixDimensions[0]][outputMatrixDimensions[1]];

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				for(int y = 0; y < outputMatrixDimensions[1]; y++){
					outputMatrix[x][y] = matrix1[x][y] + vec[y];
				}
			}

			Tensor tensor = new Tensor(outputMatrix, outputMatrixDimensions);
			return tensor;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
					public Tensor execute(Tensor... inputs){
						Tensor tensor = new Tensor(inputs[0]);
						return tensor;
					}
				};
			int[] inputsNeeded = {};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: [a, b], [a, b]
	// Output: [a, b]
	public static class MatSub extends TensorOperation{

		public Tensor execute(Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			double[][] matrix2 = (double[][])(inputs[1].getObject());
			int[] outputMatrixDimensions = {matrix1.length, matrix1[0].length};

			double[][] outputMatrix = new double[outputMatrixDimensions[0]][outputMatrixDimensions[1]];

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				for(int y = 0; y < outputMatrixDimensions[1]; y++){
					outputMatrix[x][y] = matrix1[x][y] - matrix2[x][y];
				}
			}

			Tensor tensor = new Tensor(outputMatrix, outputMatrixDimensions);
			return tensor;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = null;
			switch(inputIndex){
				case 0: {
					derivativeOp = new TensorOperation(){
						public Tensor execute(Tensor... inputs){
							Tensor tensor = new Tensor(inputs[0]);
							return tensor;
						}
					};
				}break;
				default: {
					derivativeOp = new TensorOperation(){
						public Tensor execute(Tensor... inputs){
							Tensor tensor = new Tensor(inputs[0], new CopyOp(){
								public double execute(double input){
									return -1 * input;
								}
							});
							return tensor;
						}
					};
				}
			}
			int[] inputsNeeded = {};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T
	// Output: T
	public static class TensorSigmoid extends TensorOperation{

		public Tensor execute(Tensor... inputs){
			Tensor tensor = new Tensor(inputs[0], new CopyOp(){
				public double execute(double input){
					return 1.0/(1 + Math.exp(-1.0 * input));
				}
			});
			return tensor;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public Tensor execute(Tensor... inputs){
					Tensor tensor = new Tensor(inputs[0]);
					Index index = new Index(tensor.getOrder());
					do{
						double value = inputs[1].getValue(index);
						value = 1.0 / (1 + Math.exp(-1 * value)); // sigmoid
						value = value * (1 - value); // sigmoid derivative
						value *= inputs[0].getValue(index); // multiply with gradient
						tensor.setValue(index, value);
					}while(index.increment(tensor));
					return tensor;
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T
	// Output: T
	public static class TensorSquare extends TensorOperation{

		public Tensor execute(Tensor... inputs){
			Tensor tensor = new Tensor(inputs[0], new CopyOp(){
				public double execute(double input){
					return Math.pow(input, 2);
				}
			});
			return tensor;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public Tensor execute(Tensor... inputs){
					Tensor tensor = new Tensor(inputs[0]);
					Index index = new Index(tensor.getOrder());
					do{
						double value = 2 * inputs[1].getValue(index);
						value *= inputs[0].getValue(index); // multiply with gradient
						tensor.setValue(index, value);
					}while(index.increment(tensor));
					return tensor;
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: [a, b]
	// Output: [a]
	public static class MatSumCols extends TensorOperation{

		public Tensor execute(Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			int[] outputMatrixDimensions = {matrix1.length, matrix1[0].length};

			double[] outputMatrix = new double[outputMatrixDimensions[0]];

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				double sum = 0;
				for(int y = 0; y < outputMatrixDimensions[1]; y++){
					sum += matrix1[x][y];
				}
				outputMatrix[x] = sum;
			}

			Tensor tensor = new Tensor(outputMatrix, outputMatrixDimensions);
			return tensor;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public Tensor execute(Tensor... inputs){
					double[] gradientVector = (double[])(inputs[0].getObject());
					double[][] inputMatrix = (double[][])(inputs[1].getObject());
					int[] outputMatrixDimensions = {inputMatrix.length, inputMatrix[0].length};

					double[][] outputMatrix = new double[outputMatrixDimensions[0]][outputMatrixDimensions[1]];

					for(int x = 0; x < outputMatrixDimensions[0]; x++){
						for(int y = 0; y < outputMatrixDimensions[1]; y++){
							outputMatrix[x][y] = gradientVector[x];
						}
					}

					Tensor tensor = new Tensor(outputMatrix, outputMatrixDimensions);
					return tensor;
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}
}