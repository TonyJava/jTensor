package jTensor;

public class Tensor{
		private Object object;
		private int[] dimensions; // don't change (no setter)
		private int order;

		public Tensor(Object object, int[] dimensions){
			this.object = object;
			this.dimensions = dimensions;
			order = dimensions.length;
		}

		public Tensor(int[] dimensions){
			this.dimensions = dimensions;
			order = dimensions.length;
			switch(order){
				case 0: object = 0;break;
				case 1: object = new double[dimensions[0]];break;
				case 2: object = new double[dimensions[0]][dimensions[1]];break;
				case 3: object = new double[dimensions[0]][dimensions[1]][dimensions[2]];break;
				case 4: object = new double[dimensions[0]][dimensions[1]][dimensions[2]][dimensions[3]];break;
				default: System.out.println("Unexpected order");
			}
		}

		public Tensor(int[] dimensions, InitOp op){
			this.dimensions = dimensions; // does not copy
			this.order = dimensions.length;
			Index index = new Index(order);
			switch(order){
				case 0: object = op.execute(dimensions, index);break;
				case 1: {
					double[] valsCopy = new double[dimensions[0]];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						valsCopy[index.values[0]] = op.execute(dimensions, index);
					}
					object = valsCopy;
					break;
				}
				case 2: {
					double[][] valsCopy = new double[dimensions[0]][dimensions[1]];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							valsCopy[index.values[0]][index.values[1]] = op.execute(dimensions, index);
						}
					}
					object = valsCopy;
					break;
				}
				case 3: {
					double[][][] valsCopy = new double[dimensions[0]][dimensions[1]][dimensions[2]];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								valsCopy[index.values[0]][index.values[1]][index.values[2]] = op.execute(dimensions, index);
							}
						}
					}
					object = valsCopy;
					break;
				}
				case 4: {
					double[][][][] valsCopy = new double[dimensions[0]][dimensions[1]][dimensions[2]][dimensions[3]];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								for(index.values[3] = 0; index.values[3] < dimensions[3]; index.values[3]++){
									valsCopy[index.values[0]][index.values[1]][index.values[2]][index.values[3]] = op.execute(dimensions, index);
								}
							}
						}
					}
					object = valsCopy;
					break;
				}
				default: System.out.println("Unexpected order");
			}
		}


		public Tensor(Tensor tensor, CopyOp op){
			this.dimensions = tensor.dimensions; // does not copy
			this.order = tensor.order;
			Index index = new Index(order);
			switch(order){
				case 0: Double val = (Double)tensor.object; object = op.execute(val, index);break;
				case 1: {
					double[] vals = ((double[])tensor.object);
					double[] valsCopy = new double[vals.length];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						valsCopy[index.values[0]] = op.execute(vals[index.values[0]], index);
					}
					object = valsCopy;
					break;
				}
				case 2: {
					double[][] vals = ((double[][])tensor.object);
					double[][] valsCopy = new double[vals.length][vals[0].length];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							valsCopy[index.values[0]][index.values[1]] = op.execute(vals[index.values[0]][index.values[1]], index);
						}
					}
					object = valsCopy;
					break;
				}
				case 3: {
					double[][][] vals = ((double[][][])tensor.object);
					double[][][] valsCopy = new double[vals.length][vals[0].length][vals[0][0].length];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								valsCopy[index.values[0]][index.values[1]][index.values[2]] = op.execute(vals[index.values[0]][index.values[1]][index.values[2]], index);
							}
						}
					}
					object = valsCopy;
					break;
				}
				case 4: {
					double[][][][] vals = ((double[][][][])tensor.object);
					double[][][][] valsCopy = new double[vals.length][vals[0].length][vals[0][0].length][vals[0][0][0].length];
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								for(index.values[3] = 0; index.values[3] < dimensions[3]; index.values[3]++){
									valsCopy[index.values[0]][index.values[1]][index.values[2]][index.values[3]] = op.execute(vals[index.values[0]][index.values[1]][index.values[2]][index.values[3]], index);
								}
							}
						}
					}
					object = valsCopy;
					break;
				}
				default: System.out.println("Unexpected order");
			}
		}

		public Tensor(Tensor tensor){
			this(tensor, CopyOp.identity);
		}

		public void copyTo(Tensor destTensor, CopyOp op){
			Tensor sourceTensor = this;
			Index index = new Index(order);
			switch(order){
				case 0: Double val = (Double)sourceTensor.object; destTensor.object = op.execute(val, index);break;
				case 1: {
					double[] vals = ((double[])sourceTensor.object);
					double[] valsCopy = (double[])(destTensor.getObject());
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						valsCopy[index.values[0]] = op.execute(vals[index.values[0]], index);
					}
					break;
				}
				case 2: {
					double[][] vals = ((double[][])sourceTensor.object);
					double[][] valsCopy = (double[][])(destTensor.getObject());
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							valsCopy[index.values[0]][index.values[1]] = op.execute(vals[index.values[0]][index.values[1]], index);
						}
					}
					break;
				}
				case 3: {
					double[][][] vals = ((double[][][])sourceTensor.object);
					double[][][] valsCopy = (double[][][])(destTensor.getObject());
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								valsCopy[index.values[0]][index.values[1]][index.values[2]] = op.execute(vals[index.values[0]][index.values[1]][index.values[2]], index);
							}
						}
					}
					break;
				}
				case 4: {
					double[][][][] vals = ((double[][][][])sourceTensor.object);
					double[][][][] valsCopy = (double[][][][])(destTensor.getObject());
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								for(index.values[3] = 0; index.values[3] < dimensions[3]; index.values[3]++){
									valsCopy[index.values[0]][index.values[1]][index.values[2]][index.values[3]] = op.execute(vals[index.values[0]][index.values[1]][index.values[2]][index.values[3]], index);
								}
							}
						}
					}
					break;
				}
				default: System.out.println("Unexpected order");
			}
		}

		public void operate(CopyOp op){
			Tensor sourceTensor = this;
			Index index = new Index(order);
			switch(order){
				case 0: Double val = (Double)sourceTensor.object; object = op.execute(val, index);break;
				case 1: {
					double[] vals = ((double[])sourceTensor.object);
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						vals[index.values[0]] = op.execute(vals[index.values[0]], index);
					}
					break;
				}
				case 2: {
					double[][] vals = ((double[][])sourceTensor.object);
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							vals[index.values[0]][index.values[1]] = op.execute(vals[index.values[0]][index.values[1]], index);
						}
					}
					break;
				}
				case 3: {
					double[][][] vals = ((double[][][])sourceTensor.object);
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								vals[index.values[0]][index.values[1]][index.values[2]] = op.execute(vals[index.values[0]][index.values[1]][index.values[2]], index);
							}
						}
					}
					break;
				}
				case 4: {
					double[][][][] vals = ((double[][][][])sourceTensor.object);
					for(index.values[0] = 0; index.values[0] < dimensions[0]; index.values[0]++){
						for(index.values[1] = 0; index.values[1] < dimensions[1]; index.values[1]++){
							for(index.values[2] = 0; index.values[2] < dimensions[2]; index.values[2]++){
								for(index.values[3] = 0; index.values[3] < dimensions[3]; index.values[3]++){
									vals[index.values[0]][index.values[1]][index.values[2]][index.values[3]] = op.execute(vals[index.values[0]][index.values[1]][index.values[2]][index.values[3]], index);
								}
							}
						}
					}
					break;
				}
				default: System.out.println("Unexpected order");
			}
		}

		public double getValue(Index index){
			double returnValue = 0;
			switch(order){
				case 0: returnValue = (Double)object;break;
				case 1: returnValue = ((double[])object)[index.values[0]];break;
				case 2: returnValue = ((double[][])object)[index.values[0]][index.values[1]];break;
				case 3: returnValue = ((double[][][])object)[index.values[0]][index.values[1]][index.values[2]];break;
				case 4: returnValue = ((double[][][][])object)[index.values[0]][index.values[1]][index.values[2]][index.values[3]];break;
				default: System.out.println("Unexpected order");
			}
			return returnValue;
		}

		public void setValue(Index index, double value){
			switch(order){
				case 0: Double val = value; object = val;break;
				case 1: ((double[])object)[index.values[0]] = value;break;
				case 2: ((double[][])object)[index.values[0]][index.values[1]] = value;break;
				case 3: ((double[][][])object)[index.values[0]][index.values[1]][index.values[2]] = value;break;
				case 4: ((double[][][][])object)[index.values[0]][index.values[1]][index.values[2]][index.values[3]] = value;break;
				default: System.out.println("Unexpected order");
			}
		}

		class AvgCountWrapper{
			double sum;
			int count;
		}

		public double getAverage(){
			final AvgCountWrapper avgCount = new AvgCountWrapper();
			avgCount.sum = 0;
			avgCount.count = 0;
			operate(new CopyOp(){
				public double execute(double value, Index index){
					avgCount.sum += value;
					avgCount.count++;
					return value;
				}
			});
			return avgCount.sum/avgCount.count;
		}

		public double getAverageMagnitude(){
			final AvgCountWrapper avgCount = new AvgCountWrapper();
			avgCount.sum = 0;
			avgCount.count = 0;
			operate(new CopyOp(){
				public double execute(double value, Index index){
					avgCount.sum += Math.abs(value);
					avgCount.count++;
					return value;
				}
			});
			return avgCount.sum/avgCount.count;
		}

		public Object getObject(){
			return object;
		}

		public int[] getDimensions(){
			return dimensions;
		}

		public int getOrder(){
			return order;
		}

		public void printTensor(){
			Index index = new Index(getOrder());
			int size = 1;
			for(int j = 0; j < getOrder(); j++){
				size *= getDimensions()[j];
			}
			size = size > 30 ? 30 : size;
			do{
				System.out.print(getValue(index) + ", ");
			}while(index.increment(this) && (size--) > 0);
			System.out.print("\n");
		}
}