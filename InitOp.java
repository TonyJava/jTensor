// Wrapper for an init function
public abstract class InitOp{
	public abstract double execute(int[] dimensions, Index index);

	public static final InitOp initRandomUniform = new InitOp(){
		public double execute(int[] dimensions, Index index){
			return Math.random();
		}
	};
}