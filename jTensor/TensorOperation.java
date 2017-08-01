package jTensor;

public abstract class TensorOperation{


	// Executes operation, derivative inputs should come after output gradient
	public abstract void execute(Tensor outputTensor, Tensor... inputs);

	// Gives tensor output dimensions given array of input dimensions (by default identity)
	// returns null if illegal input
	// Is checked every run to see if a tensor can be reused
	public int[] getOutputDimensions(int[][] inputDimensions){
		return inputDimensions[0];
	}

	// Returns gradients of operation with respect to given input (index base 0 of execute param)
	public TensorDerivativeInfo getDerivative(final int inputIndex){
		TensorOperation derivativeOp = new TensorOperation(){
			public void execute(Tensor output, Tensor... inputs){
				inputs[0].copyTo(output, CopyOp.identity);
			}
		};
		int[] inputsNeeded = {};
		return new TensorDerivativeInfo(derivativeOp, inputsNeeded);
	}

	// Returns gradients of operation with respect to given input (index base 0 of execute param)
	// If returns null, getDerivative(final int) will be called instead
	public TensorDerivativeInfo getDerivative(final int inputIndex, int[][] inputDimensions){
		return null;
	}
	
	public static class TensorDerivativeInfo{
		TensorOperation op;
		int[] inputsNeeded; // original inputs that the derivative will also 
		                    // need, will become input[i+1] (due to gradient being first)

		public TensorDerivativeInfo(TensorOperation op, int[] inputsNeeded){
			this.op = op;
			this.inputsNeeded = inputsNeeded;
		}
	}
}