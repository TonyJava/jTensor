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
			int[] index = new int[order];
			switch(order){
				case 0: object = op.execute(dimensions, index);break;
				case 1: {
					double[] valsCopy = new double[dimensions[0]];
					for(index[0] = 0; index[0] < dimensions[0]; index[0]++){
						valsCopy[index[0]] = op.execute(dimensions, index);
					}
					object = valsCopy;
					break;
				}
				case 2: {
					double[][] valsCopy = new double[dimensions[0]][dimensions[1]];
					for(index[0] = 0; index[0] < dimensions[0]; index[0]++){
						for(index[1] = 0; index[1] < dimensions[1]; index[1]++){
							valsCopy[index[0]][index[1]] = op.execute(dimensions, index);
						}
					}
					object = valsCopy;
					break;
				}
				case 3: {
					double[][][] valsCopy = new double[dimensions[0]][dimensions[1]][dimensions[2]];
					for(index[0] = 0; index[0] < dimensions[0]; index[0]++){
						for(index[1] = 0; index[1] < dimensions[1]; index[1]++){
							for(index[2] = 0; index[2] < dimensions[2]; index[2]++){
								valsCopy[index[0]][index[1]][index[2]] = op.execute(dimensions, index);
							}
						}
					}
					object = valsCopy;
					break;
				}
				case 4: {
					double[][][][] valsCopy = new double[dimensions[0]][dimensions[1]][dimensions[2]][dimensions[3]];
					for(index[0] = 0; index[0] < dimensions[0]; index[0]++){
						for(index[1] = 0; index[1] < dimensions[1]; index[1]++){
							for(index[2] = 0; index[2] < dimensions[2]; index[2]++){
								for(index[3] = 0; index[3] < dimensions[3]; index[3]++){
									valsCopy[index[0]][index[1]][index[2]][index[3]] = op.execute(dimensions, index);
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
			switch(order){
				case 0: Double val = (Double)tensor.object; object = op.execute(val);break;
				case 1: {
					double[] vals = ((double[])tensor.object);
					double[] valsCopy = new double[vals.length];
					for(int a = 0; a < vals.length; a++){
						valsCopy[a] = op.execute(vals[a]);
					}
					object = valsCopy;
					break;
				}
				case 2: {
					double[][] vals = ((double[][])tensor.object);
					double[][] valsCopy = new double[vals.length][vals[0].length];
					for(int a = 0; a < vals.length; a++){
						for(int b = 0; b < vals[0].length; b++){
							valsCopy[a][b] = op.execute(vals[a][b]);
						}
					}
					object = valsCopy;
					break;
				}
				case 3: {
					double[][][] vals = ((double[][][])tensor.object);
					double[][][] valsCopy = new double[vals.length][vals[0].length][vals[0][0].length];
					for(int a = 0; a < vals.length; a++){
						for(int b = 0; b < vals[0].length; b++){
							for(int c = 0; c < vals[0][0].length; c++){
								valsCopy[a][b][c] = op.execute(vals[a][b][c]);
							}
						}
					}
					object = valsCopy;
					break;
				}
				case 4: {
					double[][][][] vals = ((double[][][][])tensor.object);
					double[][][][] valsCopy = new double[vals.length][vals[0].length][vals[0][0].length][vals[0][0][0].length];
					for(int a = 0; a < vals.length; a++){
						for(int b = 0; b < vals[0].length; b++){
							for(int c = 0; c < vals[0][0].length; c++){
								for(int d = 0; d < vals[0][0][0].length; d++){
									valsCopy[a][b][c][d] = op.execute(vals[a][b][c][d]);
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
			do{
				System.out.print(getValue(index) + ", ");
			}while(index.increment(this));
			System.out.print("\n");
		}
}