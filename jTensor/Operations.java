package jTensor;

public class Operations{


	// Params: stride
	// Input: [n, x, y, z], [a, b, z, f]
	// Output: [n, v, w, f]
	public static class Conv2d extends TensorOperation{

		int stride;

		public Conv2d(int stride){
			this.stride = stride;
		}

		public void execute(Tensor output, Tensor... inputs){
			double[][][][] matrix1 = (double[][][][])(inputs[0].getObject());
			double[][][][] filters = (double[][][][])(inputs[1].getObject());
			int[] outputMatrixDimensions = output.getDimensions();

			double[][][][] outputMatrix = (double[][][][])(output.getObject());

			for(int n = 0; n < outputMatrixDimensions[0]; n++){
				for(int v = 0; v < outputMatrixDimensions[1]; v++){
					int matX = v*stride;
					for(int w = 0; w < outputMatrixDimensions[2]; w++){
						int matY = w*stride;
						for(int f = 0; f < outputMatrixDimensions[3]; f++){
							double sum = 0;
							for(int a = 0; a < filters.length; a++){
								for(int b = 0; b < filters[0].length; b++){
									for(int z = 0; z < filters[0][0].length; z++){
										sum += matrix1[n][matX+a][matY+b][z] * filters[a][b][z][f];
									}
								}
							}
							outputMatrix[n][v][w][f] = sum;
						}
					}
				}
			}
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int x = inputDimensions[0][1];
			int y = inputDimensions[0][2];
			int fx = inputDimensions[1][0];
			int fy = inputDimensions[1][1];
			int[] retVal = {inputDimensions[0][0], (x-(fx-stride))/stride, (y-(fy-stride))/stride, inputDimensions[1][3]};
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
							double[][][][] matrix1 = (double[][][][])(inputs[0].getObject()); // gradients [n, v, w, f]
							double[][][][] matrix2 = (double[][][][])(inputs[1].getObject()); // weights [a, b, z, f]
							int[] outputMatrixDimensions = output.getDimensions();
							int innerSize = matrix1[0].length;
							double[][][][] outputMatrix = (double[][][][])(output.getObject());
							for(int n = 0; n < matrix1.length; n++){
								for(int v = 0; v < matrix1[0].length; v++){
									int matX = v*stride;
									for(int w = 0; w < matrix1[0][0].length; w++){
										int matY = w*stride;
										for(int f = 0; f < matrix1[0][0][0].length; f++){
											for(int a = 0; a < matrix2.length; a++){
												for(int b = 0; b < matrix2[0].length; b++){
													for(int z = 0; z < matrix2[0][0].length; z++){
														outputMatrix[n][matX+a][matY+b][z] += matrix1[n][v][w][f] * matrix2[a][b][z][f];
													}
												}
											}
										}
									}
								}
							}
						}

						@Override
						public int[] getOutputDimensions(int[][] inputDimensions){
							int a = inputDimensions[1][0];
							int b = inputDimensions[1][1];
							int n = inputDimensions[0][0];
							int v = inputDimensions[0][1];
							int w = inputDimensions[0][2];
							int z = inputDimensions[1][2];

							int x = stride * (v - 1) + a;
							int y = stride * (w - 1) + b;

							// [n, x, y, z]
							int[] retVal = {n, x, y, z};
							return retVal;
						}
					};
				break;
				// wrt weights
				default: 
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							double[][][][] matrix1 = (double[][][][])(inputs[0].getObject()); // gradients [n, v, w, f]
							double[][][][] matrix2 = (double[][][][])(inputs[1].getObject()); // inputs [n, x, y, z]
							int[] outputMatrixDimensions = output.getDimensions();
							int innerSize = matrix1[0].length;
							double[][][][] outputMatrix = (double[][][][])(output.getObject());
							for(int n = 0; n < matrix1.length; n++){
								for(int v = 0; v < matrix1[0].length; v++){
									int matX = v*stride;
									for(int w = 0; w < matrix1[0][0].length; w++){
										int matY = w*stride;
										for(int f = 0; f < matrix1[0][0][0].length; f++){
											for(int a = 0; a < outputMatrixDimensions[0]; a++){
												for(int b = 0; b < outputMatrixDimensions[1]; b++){
													for(int z = 0; z < outputMatrixDimensions[2]; z++){
														double mat1Val = matrix1[n][v][w][f];
														double mat2Val = matrix2[n][matX+a][matY+b][z];
														outputMatrix[a][b][z][f] +=  mat1Val * mat2Val;
													}
												}
											}
										}
									}
								}
							}
						}

						@Override
						public int[] getOutputDimensions(int[][] inputDimensions){
							int n = inputDimensions[0][0];
							int v = inputDimensions[0][1];
							int w = inputDimensions[0][2];
							int z = inputDimensions[1][3];
							int x = inputDimensions[1][1];
							int y = inputDimensions[1][2];
							int f = inputDimensions[0][3];
	
							int a = -stride * (v - 1) + x;
							int b = -stride * (w - 1) + y;

							// [a, b, z, f]
							int[] retVal = {a, b, z, f};
							return retVal;
						}
					};
			}
			int[] inputsNeeded = {1 - inputIndex};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}

	}

	// Params: stride, filterSize
	// Input: [n, x, y, z]
	// Output: [n, v, w, z]
	// Probably won't work correctly if filters overlap (if filterSize > stride)
	public static class MaxPool2d extends TensorOperation{

		int filterSize;
		int stride;

		public MaxPool2d(int stride, int filterSize){
			this.stride = stride;
			this.filterSize = filterSize;
		}

		public void execute(Tensor output, Tensor... inputs){
			double[][][][] matrix1 = (double[][][][])(inputs[0].getObject());
			int[] outputMatrixDimensions = output.getDimensions();

			double[][][][] outputMatrix = (double[][][][])(output.getObject());

			for(int n = 0; n < outputMatrixDimensions[0]; n++){
				for(int v = 0; v < outputMatrixDimensions[1]; v++){
					int matX = v*stride;
					for(int w = 0; w < outputMatrixDimensions[2]; w++){
						int matY = w*stride;
						for(int f = 0; f < outputMatrixDimensions[3]; f++){
							double highest = 0;
							boolean initialized = false;
							for(int a = 0; a < filterSize; a++){
								for(int b = 0; b < filterSize; b++){
									double value = matrix1[n][matX+a][matY+b][f];
									if(value > highest || !initialized){
										initialized = true;
										highest = value;
									}
								}
							}
							outputMatrix[n][v][w][f] = highest;
						}
					}
				}
			}
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int x = inputDimensions[0][1];
			int y = inputDimensions[0][2];
			int fx = filterSize;
			int fy = filterSize;
			int[] retVal = {inputDimensions[0][0], (x-(fx-stride))/stride, (y-(fy-stride))/stride, inputDimensions[0][3]};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp;
			derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					double[][][][] matrix0 = (double[][][][])(inputs[0].getObject()); // gradients [n, v, w, z]
					double[][][][] matrix1 = (double[][][][])(inputs[1].getObject()); // inputs [n, x, y, z]
					int[] gradientDimensions = inputs[0].getDimensions();
					double[][][][] outputMatrix = (double[][][][])(output.getObject());

					
					for(int n = 0; n < gradientDimensions[0]; n++){
						for(int v = 0; v < gradientDimensions[1]; v++){
							int matX = v*stride;
							for(int w = 0; w < gradientDimensions[2]; w++){
								int matY = w*stride;
								for(int f = 0; f < gradientDimensions[3]; f++){
									double highest = 0;
									boolean initialized = false;
									for(int a = 0; a < filterSize; a++){
										for(int b = 0; b < filterSize; b++){
											double value = matrix1[n][matX+a][matY+b][f];
											if(value > highest || !initialized){
												initialized = true;
												highest = value;
											}
										}
									}
									for(int a = 0; a < filterSize; a++){
										for(int b = 0; b < filterSize; b++){
											double value = matrix1[n][matX+a][matY+b][f];
											if(value == highest){
												outputMatrix[n][matX+a][matY+b][f] = matrix0[n][v][w][f];
											}else{
												outputMatrix[n][matX+a][matY+b][f] = 0.0;
											}
										}
									}
								}
							}
						}
					}
				}

				@Override
				public int[] getOutputDimensions(int[][] inputDimensions){
					return inputDimensions[1];
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}
	
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

	// Input: [A, B]
	// Output: [A, B]
	public static class MatSoftmax extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			final double[][] matrix = ((double[][])inputs[0].getObject());
			// Get max
			final double[] max = new double[matrix.length];
			for(int j = 0; j < matrix.length; j++){
				max[j] = matrix[j][0];
				for(int i = 1; i < matrix[j].length; i++){
					if(max[j] < matrix[j][i]){
						max[j] = matrix[j][i];
					}
				}
			}
			// Get sum
			final double[] sum = new double[matrix.length];
			for(int j = 0; j < matrix.length; j++){
				for(int i = 0; i < matrix[j].length; i++){
					sum[j] += Math.exp(matrix[j][i] - max[j]);
				}
			}
			// Get softmax
			output.operate(new CopyOp(){
				public double execute(double input, Index index){
					int j = index.getValues()[0];
					int i = index.getValues()[1];
					return Math.exp(matrix[j][i] - max[j])/sum[j];
				}
			});
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					final double[][] matrix = ((double[][])inputs[1].getObject());
					// Get max
					final double[] max = new double[matrix.length];
					for(int j = 0; j < matrix.length; j++){
						max[j] = matrix[j][0];
						for(int i = 1; i < matrix[j].length; i++){
							if(max[j] < matrix[j][i]){
								max[j] = matrix[j][i];
							}
						}
					}
					// Get sum
					final double[] sum = new double[matrix.length];
					for(int j = 0; j < matrix.length; j++){
						for(int i = 0; i < matrix[j].length; i++){
							sum[j] += Math.exp(matrix[j][i] - max[j]);
						}
					}
					// Get softmax
					inputs[0].copyTo(output, new CopyOp(){
						public double execute(double input, Index index){
							int j = index.getValues()[0];
							int i = index.getValues()[1];
							double softmaxValue = Math.exp(matrix[j][i] - max[j])/sum[j];
							return input * (softmaxValue * (1 - softmaxValue));
						}
					});
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T
	// Param: T'
	// Output: T'
	public static class TensorReshape extends TensorOperation{

		int[] dimensions;

		public TensorReshape(int[] dimensions){
			this.dimensions = dimensions;
		}

		public void execute(Tensor output, Tensor... inputs){
			final int[] outputDimensions = dimensions;
			final int[] inputDimensions = inputs[0].getDimensions();
			final int[] inputMultDimensions = new int[inputDimensions.length];
			inputMultDimensions[inputMultDimensions.length - 1] = 1;
			for(int j = inputMultDimensions.length - 2; j >= 0; j--){
				inputMultDimensions[j] = inputMultDimensions[j + 1] * inputDimensions[j + 1];
			}
			final int[] outputMultDimensions = new int[outputDimensions.length];
			outputMultDimensions[outputMultDimensions.length - 1] = 1;
			for(int j = outputMultDimensions.length - 2; j >= 0; j--){
				outputMultDimensions[j] = outputMultDimensions[j + 1] * outputDimensions[j + 1];
			}
			Index outputIndex = new Index(outputDimensions.length);
			int[] outputIndexValues = outputIndex.getValues();
			inputs[0].operate(new CopyOp(){
				public double execute(double input, Index index){
					int totalIndex = 0;
					int[] inputIndexValues = index.getValues();
					for(int j = 0; j < inputDimensions.length; j++){
						totalIndex += inputMultDimensions[j] * inputIndexValues[j];
					}
					for(int j = 0; j < outputDimensions.length; j++){
						outputIndexValues[j] = totalIndex/outputMultDimensions[j];
						totalIndex -= outputMultDimensions[j] * outputIndexValues[j];
					}
					output.setValue(outputIndex, input);
					return input;
				}
			});
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			return dimensions;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					
					final int[] inputDimensions = dimensions;
					final int[] outputDimensions = output.getDimensions();
					final int[] inputMultDimensions = new int[inputDimensions.length];
					inputMultDimensions[inputMultDimensions.length - 1] = 1;
					for(int j = inputMultDimensions.length - 2; j >= 0; j--){
						inputMultDimensions[j] = inputMultDimensions[j + 1] * inputDimensions[j + 1];
					}
					final int[] outputMultDimensions = new int[outputDimensions.length];
					outputMultDimensions[outputMultDimensions.length - 1] = 1;
					for(int j = outputMultDimensions.length - 2; j >= 0; j--){
						outputMultDimensions[j] = outputMultDimensions[j + 1] * outputDimensions[j + 1];
					}
					Index outputIndex = new Index(outputDimensions.length);
					int[] outputIndexValues = outputIndex.getValues();
					inputs[0].operate(new CopyOp(){
						public double execute(double input, Index index){
							int totalIndex = 0;
							int[] inputIndexValues = index.getValues();
							for(int j = 0; j < inputDimensions.length; j++){
								totalIndex += inputMultDimensions[j] * inputIndexValues[j];
							}
							for(int j = 0; j < outputDimensions.length; j++){
								outputIndexValues[j] = totalIndex/outputMultDimensions[j];
								totalIndex -= outputMultDimensions[j] * outputIndexValues[j];
							}
							output.setValue(outputIndex, input);
							return input;
						}
					});
				}

				@Override
				public int[] getOutputDimensions(int[][] inputDimensions){
					return inputDimensions[1];
				}
			};
			int[] inputsNeeded = {0};
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
	// Actually a Leaky ReLU (ie derivative at x < 0 is .01)
	public static class TensorReLU extends TensorOperation{

		private final static double leakyValue = 0.01;

		public void execute(Tensor output, Tensor... inputs){
			inputs[0].copyTo(output, new CopyOp(){
				public double execute(double input, Index index){
					return (input < 0 ? leakyValue*input : input);
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
							return input * (ogInput < 0 ? leakyValue : 1);
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

	// Input: T
	// Output: T
	public static class TensorLn extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			inputs[0].copyTo(output, new CopyOp(){
				public double execute(double input, Index index){
					return Math.log(input);
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
							return input * (1/ogInput);
						}
					});
					
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T, T
	// Output: T
	public static class TensorAdd extends TensorOperation{
		public void execute(Tensor output, Tensor... inputs){
			final Tensor t1 = inputs[0];
			final Tensor t2 = inputs[1];
			int[] t1Dimensions = t1.getDimensions();
			int[] t2Dimensions = t2.getDimensions();
			if(t1Dimensions.length == t2Dimensions.length){
				t1.copyTo(output, new CopyOp(){
					public double execute(double input, Index index){
						return input + t2.getValue(index);
					}
				});
				return;
			}
			boolean t1Higher = t1Dimensions.length > t2Dimensions.length;
			Tensor higher = t1Higher ? t1 : t2;
			final Tensor lower = !t1Higher ? t1 : t2;
			final int[] outputDimensions = higher.getDimensions();
			final int[] lowerDimensions = lower.getDimensions();

			final int[] lowerIndexValues = new int[lowerDimensions.length];
			final Index lowerIndex = new Index(lowerIndexValues);

			higher.copyTo(output, new CopyOp(){
				public double execute(double input, Index index){
					int[] indexValues = index.getValues();
					
					for(int j = 0; j < lowerDimensions.length; j++){
						lowerIndexValues[lowerDimensions.length-1-j] = indexValues[outputDimensions.length-1-j];
					}
					return input + lower.getValue(lowerIndex);
				}
			});
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			boolean i0Higher = inputDimensions[0].length > inputDimensions[1].length;
			int[] higher = i0Higher ? inputDimensions[0] : inputDimensions[1];
			return higher;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex, int[][] inputDimensions){
			TensorOperation derivativeOp;
			int[] inputsNeeded;

			int[] t1Dimensions = inputDimensions[0];
			int[] t2Dimensions = inputDimensions[1];
			if(t1Dimensions.length == t2Dimensions.length){
				derivativeOp = new TensorOperation(){
					public void execute(Tensor output, Tensor... inputs){
						inputs[0].copyTo(output, CopyOp.identity);
					}
				};
				inputsNeeded = new int[0];
			}else{
				// Accounts for gradient in index (ie index of execute param: inputs)
				final int higherIndex = t1Dimensions.length > t2Dimensions.length ? 1 : 2;
				// Accounts for gradient
				final int finalInputIndex = inputIndex + 1;
				if(finalInputIndex == higherIndex){
						derivativeOp = new TensorOperation(){
							public void execute(Tensor output, Tensor... inputs){
								inputs[0].copyTo(output, CopyOp.identity);
							}
							@Override
							public int[] getOutputDimensions(int[][] inputDimensions){
								int[] retVal = inputDimensions[finalInputIndex];
								return retVal;
							}
						};
				}else{
					// wrt lower
					derivativeOp = new TensorOperation(){
						public void execute(Tensor output, Tensor... inputs){
							// inputs[0] is gradient with dimensions of higher
							// output has dimensions of lower

							final Tensor higher = inputs[0];
							final Tensor lower = output;
							
							final int[] outputDimensions = lower.getDimensions();
							final int[] higherDimensions = higher.getDimensions();

							final int[] lowerIndexValues = new int[outputDimensions.length];
							final Index lowerIndex = new Index(lowerIndexValues);

							// Clear output to allow for accumulating
							output.operate(CopyOp.zero);

							inputs[0].operate(new CopyOp(){
								public double execute(double input, Index index){
									int[] indexValues = index.getValues();
									
									for(int j = 0; j < outputDimensions.length; j++){
										lowerIndexValues[outputDimensions.length-1-j] = indexValues[higherDimensions.length-1-j];
									}
									output.setValue(lowerIndex, output.getValue(lowerIndex) + input);
									return input;
								}
							});
						}
						@Override
						public int[] getOutputDimensions(int[][] inputDimensions){
							int[] retVal = inputDimensions[finalInputIndex];
							return retVal;
						}
					};
				}
				inputsNeeded = new int[inputDimensions.length];
				for(int j = 0; j < inputsNeeded.length; j++){
					inputsNeeded[j] = j;
				}
			}
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T, T
	// Output: T
	public static class TensorScale extends TensorOperation{
		public void execute(Tensor output, Tensor... inputs){
			inputs[0].copyTo(output, new CopyOp(){
				public double execute(double input, Index index){
					return input * inputs[1].getValue(index);
				}
			});
		}

		@Override
		public TensorDerivativeInfo getDerivative(final int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					final Tensor originalInputs = new Tensor(inputs[1 - inputIndex]);
					inputs[inputIndex].copyTo(output, new CopyOp(){
						public double execute(double input, Index index){
							double ogInput = originalInputs.getValue(index);
							return input * ogInput;
						}
					});
				}
			};
			int[] inputsNeeded = {1 - inputIndex};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T
	// Output: [1]
	public static class TensorAverage extends TensorOperation{
		public void execute(Tensor output, Tensor... inputs){
			final double[] outputArray = (double[])(output.getObject());
			outputArray[0] = 0;
			inputs[0].operate(new CopyOp(){
				public double execute(double input, Index index){
					outputArray[0] += input;
					return input;
				}
			});
			outputArray[0] /= output.getSize();
		}

		@Override
		public TensorDerivativeInfo getDerivative(final int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					final int size = inputs[1].getSize();
					inputs[0].copyTo(output, new CopyOp(){
						public double execute(double input, Index index){
							return input / size;
						}
					});
				}
			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: T, T (same shape)
	// Output: T (0, 1)
	public static class TensorEquals extends TensorOperation{
		public void execute(Tensor output, Tensor... inputs){
			output.operate(new CopyOp(){
				public double execute(double input, Index index){
					return inputs[0].getValue(index) == inputs[1].getValue(index) ? 1 : 0;
				}
			});
		}

		@Override
		public TensorDerivativeInfo getDerivative(final int inputIndex){
			return null;
		}
	}


	// Input: [a, b]
	// Output: [a] (int)
	public static class MatArgmax extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[][] matrix1 = (double[][])(inputs[0].getObject());
			int[] outputMatrixDimensions = {matrix1.length};

			double[] outputMatrix = (double[])(output.getObject());

			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				int highIndex = 0;
				double highVal = matrix1[x][0];
				for(int y = 1; y < matrix1[x].length; y++){
					if(matrix1[x][y] > highVal){
						highVal = matrix1[x][y];
						highIndex = y;
					}
				}
				outputMatrix[x] = highIndex;
			}
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int[] retVal = {inputDimensions[0][0]};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			return null;
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

	// Input: T
	// Output: [1]
	public static class TensorSum extends TensorOperation{
		
		public void execute(Tensor output, Tensor... inputs){
			final double[] outputMatrix = (double[])(output.getObject());
			inputs[0].operate(new CopyOp(){
				public double execute(double value, Index index){
					outputMatrix[0] += value;
					return value;
				}
			});
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int[] retVal = {1};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					double gradient = ((double[])(inputs[0].getObject()))[0];
					output.operate(new CopyOp(){
						public double execute(double value, Index index){
							return gradient;
						}
					});
				}

				@Override
				public int[] getOutputDimensions(int[][] inputDimensions){
					int[] retVal = inputDimensions[1];
					return retVal;
				}

			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}

	}

	// Input: [a]
	// Output: [1]
	public static class VecSum extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[] matrix1 = (double[])(inputs[0].getObject());
			int[] outputMatrixDimensions = {1};

			double[] outputMatrix = (double[])(output.getObject());

			double sum = 0;
			for(int x = 0; x < matrix1.length; x++){
				sum += matrix1[x];
			}
			outputMatrix[0] = sum;
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int[] retVal = {1};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					double[] gradientVector = (double[])(inputs[0].getObject());
					double[] inputMatrix = (double[])(inputs[1].getObject());
					int[] outputMatrixDimensions = {inputMatrix.length};

					double[] outputMatrix = (double[])(output.getObject());

					for(int x = 0; x < outputMatrixDimensions[0]; x++){
						outputMatrix[x] = gradientVector[0];
					}
				}
				@Override
				public int[] getOutputDimensions(int[][] inputDimensions){
					int[] retVal = {inputDimensions[1][0]};
					return retVal;
				}

			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// Input: [a]
	// Output: [1]
	public static class VecAvg extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[] matrix1 = (double[])(inputs[0].getObject());
			double[] outputMatrix = (double[])(output.getObject());

			double sum = 0;
			for(int x = 0; x < matrix1.length; x++){
				sum += matrix1[x];
			}
			outputMatrix[0] = sum/matrix1.length;
		}

		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int[] retVal = {1};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			TensorOperation derivativeOp = new TensorOperation(){
				public void execute(Tensor output, Tensor... inputs){
					double[] gradientVector = (double[])(inputs[0].getObject());
					double[] inputMatrix = (double[])(inputs[1].getObject());
					int[] outputMatrixDimensions = {inputMatrix.length};

					double[] outputMatrix = (double[])(output.getObject());

					for(int x = 0; x < outputMatrixDimensions[0]; x++){
						outputMatrix[x] = gradientVector[0]/inputMatrix.length;
					}
				}
				@Override
				public int[] getOutputDimensions(int[][] inputDimensions){
					int[] retVal = {inputDimensions[1][0]};
					return retVal;
				}

			};
			int[] inputsNeeded = {0};
			return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
		}
	}

	// inputs: [n, a], [n]
	// outputs: [n]
	public static class SparseCrossEntropySoftmax extends TensorOperation{

		public void execute(Tensor output, Tensor... inputs){
			double[][] logits = (double[][])(inputs[0].getObject());
			double[] targets = (double[])(inputs[1].getObject());
			int[] outputMatrixDimensions = {logits.length};

			double[] outputMatrix = (double[])(output.getObject());


			for(int x = 0; x < outputMatrixDimensions[0]; x++){
				double max = logits[x][0];
				for(int y = 1; y < logits[0].length; y++){
					if(max < logits[x][y]){
						max = logits[x][y];
					}
				}

				double softmaxSum = 0;
				for(int y = 0; y < logits[0].length; y++){
					softmaxSum += Math.exp(logits[x][y] - max);
					// System.out.println("y="+y+": "+logits[x][y]);
				}
				int correctIndex = (int)(targets[x]);

				// outputMatrix[x] = -1 * Math.log(Math.exp(logits[x][correctIndex] - max)/softmaxSum);
				outputMatrix[x] = -1 * ((logits[x][correctIndex] - max) - Math.log(softmaxSum));

				// System.out.println((logits[x][correctIndex] - max));
				// System.out.println(Math.log(softmaxSum));
				// System.out.println(outputMatrix[x]);
			}
		}
		
		@Override
		public int[] getOutputDimensions(int[][] inputDimensions){
			int[] retVal = {inputDimensions[0][0]};
			return retVal;
		}

		@Override
		public TensorDerivativeInfo getDerivative(int inputIndex){
			if(inputIndex == 0){
				TensorOperation derivativeOp = new TensorOperation(){
					public void execute(Tensor output, Tensor... inputs){
						double[] gradientVector = (double[])(inputs[0].getObject());
						double[][] logits = (double[][])(inputs[1].getObject());
						double[] targets = (double[])(inputs[2].getObject());

						double[][] outputMatrix = (double[][])(output.getObject());

						for(int x = 0; x < logits.length; x++){
							double max = logits[x][0];
							for(int y = 1; y < logits[x].length; y++){
								if(max < logits[x][y]){
									max = logits[x][y];
								}
							}

							double softmaxSum = 0;
							for(int y = 0; y < logits[x].length; y++){
								softmaxSum += Math.exp(logits[x][y] - max);
								// System.out.println(softmaxSum);
							}

							for(int y = 0; y < logits[x].length; y++){
								// System.out.println("ogsm output["+y+"]: " + (Math.exp(logits[x][y] - max)/softmaxSum));
								// System.out.println((y == (int)(targets[x]) ? 1.0 : 0.0));
								outputMatrix[x][y] = gradientVector[x] * ((Math.exp(logits[x][y] - max)/softmaxSum) - (y == (int)(targets[x]) ? 1.0 : 0.0));
							}
						}
					}
					@Override
					public int[] getOutputDimensions(int[][] inputDimensions){
						int[] retVal = {inputDimensions[0][0], inputDimensions[1][1]};
						return retVal;
					}

				};
				int[] inputsNeeded = {0, 1};
				return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
			}else{
				return null;
			}
		}


	}


}


