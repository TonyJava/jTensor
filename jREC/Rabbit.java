package jREC;

import java.awt.Graphics;
import java.awt.Color;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class Rabbit extends Environment{

	// Observations:
	// o[0]: angle diff in rads to food
	// o[1]: normalized distance to food

	// Params
	private final double hopDistance = 6.0;
	private final double foodRadius = 4;
	private final double rabbitRadius = 8;
	private final double turnSize = Math.PI/8;
	private final double[] arenaSize = {0, 0, 100, 100};
	private final double arenaDiagonal = Math.sqrt(Math.pow(arenaSize[2], 2) + Math.pow(arenaSize[3], 2));

	// Actions:
	// 0: change dir left
	// 1: change dir right
	// 2: move forward
	public static class Action extends Info{
		public int action;
	}

	class RObservation extends Info{
		double[] data;

		@Override
		public double[] getDouble1(){
			return data;
		}
	}

	class RState extends Info{
		double foodX, foodY;
		double rabbitX, rabbitY, rabbitDir;
		double rabbitHealth;
	}

	private void spawnFood(){
		RState rs = (RState)state;
		rs.foodX = (Math.random() * arenaSize[2]) + arenaSize[0];
		rs.foodY = (Math.random() * arenaSize[3]) + arenaSize[1];
	}

	private void spawnRabbit(){
		RState rs = (RState)state;
		rs.rabbitX = (Math.random() * arenaSize[2]) + arenaSize[0];
		rs.rabbitY = (Math.random() * arenaSize[2]) + arenaSize[0];
		rs.rabbitDir = Math.random() * Math.PI * 2;
		rs.rabbitHealth = 100;
	}

	private RObservation getObservation(RState state){
		RObservation ro = new RObservation();
		ro.data = new double[2];
		ro.data[0] = Math.atan2(state.foodY - state.rabbitY, state.foodX - state.rabbitX);
		ro.data[1] = Math.sqrt(Math.pow(state.foodX - state.rabbitX, 2) + Math.pow(state.foodY - state.rabbitY, 2)) / (arenaDiagonal / 3);
		return ro;
	}

	// Returns amount/type of info available
	public Space getObservationSpace(){
		Space space = new Space();
		space.type = Space.Type.CONTINUOUS;
		space.dimensions = new int[]{2};
		return space;
	}

	// Returns number of actions available
	public Space getActionSpace(){
		Space space = new Space();
		space.type = Space.Type.DISCRETE;
		space.dimensions = new int[]{3};
		return space;
	}

	// Resets the environment, returns initial observation
	public Info reset(){
		state = new RState();
		spawnRabbit();
		spawnFood();
		return getObservation((RState)state);
	}

	// Implementation for next state
	protected ROF nextState(Info state, Info action){
		ROF rof = new ROF();
		rof.reward = 1.0;
		RState rs = (RState)state;
		int choice = ((Action)action).action;
		rs.rabbitHealth -= 1;
		if(rs.rabbitHealth <= 0){
			rof.finished = true;
			rof.nextState = null;
		}else{
			rof.finished = false;
			rof.nextState = rs;

			double distance = Math.sqrt(Math.pow(rs.foodX - rs.rabbitX, 2) + Math.pow(rs.foodY - rs.rabbitY, 2));
			if(distance <= foodRadius + rabbitRadius){
				rs.rabbitHealth += 100;
				spawnFood();
			}

			if(choice < 2){ // Turn
				double modifier = choice == 0 ? 1 : -1;
				rs.rabbitDir += modifier * turnSize;
			}else{ // Move forward
				rs.rabbitX += hopDistance * Math.cos(rs.rabbitDir);
				rs.rabbitY += hopDistance * Math.sin(rs.rabbitDir);
				
				if(rs.rabbitX < arenaSize[0]){
					rs.rabbitX = arenaSize[0];
				}else if(rs.rabbitX > arenaSize[2]){
					rs.rabbitX = arenaSize[2];
				}

				if(rs.rabbitY < arenaSize[1]){
					rs.rabbitY = arenaSize[1];
				}else if(rs.rabbitY > arenaSize[3]){
					rs.rabbitY = arenaSize[3];
				}
			}

		}
		rof.observation = getObservation(rs);
		return rof;
	}

	// Implementation for renderer, return true on success
	// If observation or g is null return true if implemented
	protected boolean draw(Graphics g,  Info state){
		if(g == null || state == null){
			return true;
		}

		RState rs = (RState)state;

		g.setColor(Color.WHITE);
		g.fillRect((int)(arenaSize[0]), (int)(arenaSize[1]), (int)(arenaSize[2]), (int)(arenaSize[3]));
		g.setColor(Color.BLACK);
		g.drawRect((int)(arenaSize[0]), (int)(arenaSize[1]), (int)(arenaSize[2]), (int)(arenaSize[3]));
		
		g.setColor(Color.GREEN);
		g.fillOval((int)(rs.foodX - foodRadius), (int)(rs.foodY - foodRadius), (int)(foodRadius * 2), (int)(foodRadius * 2));
		g.setColor(Color.BLACK);
		g.drawOval((int)(rs.foodX - foodRadius), (int)(rs.foodY - foodRadius), (int)(foodRadius * 2), (int)(foodRadius * 2));

		g.setColor(Color.GRAY);
		g.fillArc((int)(rs.rabbitX - rabbitRadius), (int)(rs.rabbitY - rabbitRadius), (int)(rabbitRadius * 2), (int)(rabbitRadius * 2), 0, (int)((rs.rabbitHealth / 100.0) * 360));
		g.setColor(Color.BLACK);
		g.drawOval((int)(rs.rabbitX - rabbitRadius), (int)(rs.rabbitY - rabbitRadius), (int)(rabbitRadius * 2), (int)(rabbitRadius * 2));

		return true;
	}
}