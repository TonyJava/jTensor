import jREC.*;

public class TestRabbit{
	public static void main(String[] args) throws Exception{
		Rabbit env = new Rabbit();
		while(true){
			boolean finished = false;
			env.reset();
			while(!finished){
				Rabbit.Action a = new Rabbit.Action();
				a.action = (int)(Math.random() * 3);
				ROF rof = env.step(a);
				finished = rof.finished;
				env.render();
				Thread.sleep(10);
			}
		}
	}
}