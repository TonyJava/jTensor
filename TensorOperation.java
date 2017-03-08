public abstract class TensorOperation{

	// Executes operation, derivative inputs should come after output gradient
	public abstract Tensor execute(Tensor... inputs);

	// Returns gradients of operation with respect to given input (index base 0 of execute param)
	public TensorDerivativeInfo getDerivative(int inputIndex){
		return null;
	}
	
	public static class TensorDerivativeInfo{
		TensorOperation op;
		int[] inputsNeeded;

		public TensorDerivativeInfo(TensorOperation op, int[] inputsNeeded){
			this.op = op;
			this.inputsNeeded = inputsNeeded;
		}
	}
}