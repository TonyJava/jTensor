public class Index{
		int[] values; // length of this is order

		public Index(int order){
			values = new int[order];
		}

		public Index(int[] values){
			this.values = values;
		}

		// returns false if finished
		public boolean increment(Tensor t){
			for(int j = values.length - 1; j >= 0; j--){
				values[j] += 1;
				if(values[j] >= t.getDimensions()[j]){
					values[j] = 0;
				}else{
					return true;
				}
			}
			return false;
		}

		// returns false if finished
		// param incrementOrder default [2, 1, 0]
		public boolean increment(Tensor t, int[] incrementOrder){
			for(int j = 0; j < incrementOrder.length; j++){
				int nextDim = incrementOrder[j];
				values[nextDim] += 1;
				if(values[nextDim] >= t.getDimensions()[nextDim]){
					values[nextDim] = 0;
				}else{
					return true;
				}
			}
			return false;
		}
	}