public class Operations{
	
	// Input: [a, b], [c, d]
	// Output: [a, d]
	public static class MatMult extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			double[][] matrix2 = (double[][])(inputs[1].getObject());
			int[] outputMatrixDimensions = {matrix1.length, matrix2[0].length};
			int innerSize = matrix2.length;

			double[][] outputMatrix = (double[][])(output.getObject());

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				for(int y = 0; y < outputMatrixDimensions[1]; y++){
					double sum = 0;
					for(int j = 0; j < innerSize; j++){
						sum += matrix1[x][j] * matrix2[j][y];
					}
					outputMatrix[x][y] = sum;
				}
			}
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int[] retVal = {inputDimensions[0][0], inputDimensions[1][1]};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp;
			switch(inputIndex){
				// wrt inputs
				case 0: 
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							double[][] matrix1 = (double[][])(inputs[0].getObject()); // gradients [n, o]
							double[][] matrix2 = (double[][])(inputs[1].getObject()); // weights [i, o]
							int[] outputMatrixDimensions = {matrix1.length, matrix2.length};
							int innerSize = matrix1[0].length;
							double[][] outputMatrix = (double[][])(output.getObject());
							for(int n = 0; n < outputMatrixDimensions[0]; n++){
								for(int x = 0; x < outputMatrixDimensions[1]; x++){
									double sum = 0;
									for(int j = 0; j < innerSize; j++){
										sum += matrix2[x][j] * matrix1[n][j];
									}
									outputMatrix[n][x] = sum;
								}
							}
						}
						@Override
						public int[] getOutputDimensions(int[][] inputDimensions){
							int[] retVal = {inputDimensions[0][0], inputDimensions[1][0]};
							return retVal;
						}
					};
				break;
				// wrt weights
				default: 
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							double[][] matrix1 = (double[][])(inputs[0].getObject()); // gradients [n, o]
							double[][] matrix2 = (double[][])(inputs[1].getObject()); // inputs [n, i]
							int[] outputMatrixDimensions = {matrix2[0].length, matrix1[0].length};
							double[][] outputMatrix = (double[][])(output.getObject());
							for(int x = 0; x < outputMatrixDimensions[0]; x++){
								for(int y = 0; y < outputMatrixDimensions[1]; y++){
									double sum = 0;
									for(int n = 0; n < matrix1.length; n++){
										sum += matrix2[n][x] * matrix1[n][y];
									}
									outputMatrix[x][y] = sum;
								}
							}
						}
						@Override
						public int[] getOutputDimensions(int[][] inputDimensions){
							int[] retVal = {inputDimensions[1][1], inputDimensions[0][1]};
							return retVal;
						}
					};
			}
			int[] inputsNeeded = {1 - inputIndex};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}

	}


	// Input: [a, b], [b]
	// Output: [a, b]
	public static class MatAddVec extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			double[] vec = (double[])(inputs[1].getObject());
			int[] outputMatrixDimensions = {matrix1.length, matrix1[0].length};

			double[][] outputMatrix = (double[][])(output.getObject());

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				for(int y = 0; y < outputMatrixDimensions[1]; y++){
					outputMatrix[x][y] = matrix1[x][y] + vec[y];
				}
			}
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp;
			switch(inputIndex){
				// wrt inputs
				case 0: 
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							inputs[0].copyTo(output, CopyOp.identity);
						}
					};
				break;
				// wrt weights
				default: 
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							double[][] matrix1 = (double[][])(inputs[0].getObject()); // gradients [n, o]
							double[] outputMatrix = (double[])(output.getObject());
							for(int x = 0; x < matrix1[0].length; x++){
								double sum = 0;
								for(int n = 0; n < matrix1.length; n++){
									sum += matrix1[n][x];
								}
								outputMatrix[x] = sum;
							}
						}
						@Override
						public int[] getOutputDimensions(int[][] inputDimensions){
							int[] retVal = {inputDimensions[0][1]};
							return retVal;
						}
					};
			}
			int[] inputsNeeded = {};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: [a, b], [a, b]
	// Output: [a, b]
	public static class MatSub extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			double[][] matrix2 = (double[][])(inputs[1].getObject());
			int[] outputMatrixDimensions = {matrix1.length, matrix1[0].length};

			double[][] outputMatrix = (double[][])(output.getObject());

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				for(int y = 0; y < outputMatrixDimensions[1]; y++){
					outputMatrix[x][y] = matrix1[x][y] - matrix2[x][y];
				}
			}
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = null;
			switch(inputIndex){
				case 0: {
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							inputs[0].copyTo(output, CopyOp.identity);
						}
					};
				}break;
				default: {
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							inputs[0].copyTo(output, new CopyOp(){
								public double execute(double input, Index index){
									return -1 * input;
								}
							});
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

		public void execute(Tensor output, Tensor... inputs){
			inputs[0].copyTo(output, new CopyOp(){
				public double execute(double input, Index index){
					return 1.0/(1 + Math.exp(-1.0 * input));
				}
			});
			
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					Tensor originalInputs = new Tensor(inputs[1]);
					inputs[0].copyTo(output, new CopyOp(){
						public double execute(double input, Index index){
							double ogInput = originalInputs.getValue(index);
							double value = 1.0 / (1 + Math.exp(-1 * ogInput)); // sigmoid
							value = value * (1 - value); // sigmoid derivative
							return input * value;
						}
					});
					
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T
	// Output: T
	public static class TensorReLU extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			inputs[0].copyTo(output, new CopyOp(){
				public double execute(double input, Index index){
					return (input < 0 ? 0 : input);
				}
			});
			
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					Tensor originalInputs = inputs[1];
					inputs[0].copyTo(output, new CopyOp(){
						public double execute(double input, Index index){
							double ogInput = originalInputs.getValue(index);
							return input * (ogInput < 0 ? .01 : 1);
						}
					});
					
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T
	// Output: T
	public static class TensorSquare extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			inputs[0].copyTo(output, new CopyOp(){
				public double execute(double input, Index index){
					return Math.pow(input, 2);
				}
			});
			
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					Tensor originalInputs = new Tensor(inputs[1]);
					inputs[0].copyTo(output, new CopyOp(){
						public double execute(double input, Index index){
							double ogInput = originalInputs.getValue(index);
							return input * ogInput * 2;
						}
					});
					
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: [a, b]
	// Output: [a]
	public static class MatSumCols extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			int[] outputMatrixDimensions = {matrix1.length};

			double[] outputMatrix = (double[])(output.getObject());

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				double sum = 0;
				for(int y = 0; y < matrix1[0].length; y++){
					sum += matrix1[x][y];
				}
				outputMatrix[x] = sum;
			}
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int[] retVal = {inputDimensions[0][0]};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					double[] gradientVector = (double[])(inputs[0].getObject());
					double[][] inputMatrix = (double[][])(inputs[1].getObject());
					int[] outputMatrixDimensions = {inputMatrix.length, inputMatrix[0].length};

					double[][] outputMatrix = (double[][])(output.getObject());

					for(int x = 0; x < outputMatrixDimensions[0]; x++){
						for(int y = 0; y < outputMatrixDimensions[1]; y++){
							outputMatrix[x][y] = gradientVector[x];
						}
					}
				}
				@Override
				public int[] getOutputDimensions(int[][] inputDimensions){
					int[] retVal = {inputDimensions[0][0], inputDimensions[1][1]};
					return retVal;
				}

			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}
}