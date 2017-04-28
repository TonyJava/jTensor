package jREC;

import java.awt.Graphics;
import java.awt.Color;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class SmallRandom extends Environment{

	/*
		X X X X
		OROSORO
		G  X  G
		
		X Death Reward -1 Terminate
		O Safe
		R Safe, 50% chance to move randomly
		G Success Reward 1

		If a G is not reached within timeLimit, Terminate -1
	*/  

	// Observations:
	// One hot encoded vector of all 14 possible states

	// Actions:
	// 0: move down
	// 1: move right
	// 2: move up
	// 3: move left

	public static class Action extends Info{
		public int action;
	}

	public static class SRObservation extends Info{
		double[] data;

		@Override
		public double[] getDouble1(){
			return data;
		}
	}

	static class SRState extends Info{
		int state;
	}

	private SRObservation getObservation(SRState state){
		SRObservation ro = new SRObservation();
		ro.data = new double[4];
		ro.data[0] = Math.atan2(state.foodY - state.SRY, state.foodX - state.SRX);
		ro.data[1] = Math.sqrt(Math.pow(state.foodX - state.SRX, 2) + Math.pow(state.foodY - state.SRY, 2)) / (arenaDiagonal);
		ro.data[2] = (arenaSize[2]/2 - state.SRX) / (arenaDiagonal);
		ro.data[3] = (arenaSize[3]/2 - state.SRY) / (arenaDiagonal);
		return ro;
	}

	// Returns amount/type of info available
	public Space getObservationSpace(){
		Space space = new Space();
		space.type = Space.Type.DISCRETE;
		space.dimensions = new int[]{14};
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
		SRState rs = new SRState();
		state = rs;
		rs.foodX = 30;
		rs.foodY = 60;
		rs.SRX = 90;
		rs.SRY = 120;
		rs.SRDir = 0;
		spawnSR();
		spawnFood();
		return getObservation((SRState)state);
	}

	// Implementation for next state
	protected ROF nextState(Info state, Info action){
		ROF rof = new ROF();
		// rof.reward = 0.0;
		rof.reward = 1.0;
		SRState rs = (SRState)state;
		int choice = ((Action)action).action;
		rs.SRHealth -= 1;
		if(rs.SRHealth <= 0){
			rof.finished = true;
			rof.nextState = null;
		}else{
			rof.finished = false;
			rof.nextState = rs;

			double distance = Math.sqrt(Math.pow(rs.foodX - rs.SRX, 2) + Math.pow(rs.foodY - rs.SRY, 2));
			// rof.reward = 1/distance;

			if(distance <= foodRadius + SRRadius){
				rs.SRHealth += 100;
				spawnFood();
				// rof.reward = 1.0;

			}

			if(choice < 2){ // Turn
				double modifier = choice == 0 ? 1 : -1;
				rs.SRDir += modifier * turnSize;
				rs.SRDir = rs.SRDir % (2 * Math.PI);
			}else{ // Move forward
				rs.SRX += hopDistance * Math.cos(rs.SRDir);
				rs.SRY += hopDistance * Math.sin(rs.SRDir);
				
				if(rs.SRX < arenaSize[0]){
					rs.SRX = arenaSize[0];
				}else if(rs.SRX > arenaSize[2]){
					rs.SRX = arenaSize[2];
				}

				if(rs.SRY < arenaSize[1]){
					rs.SRY = arenaSize[1];
				}else if(rs.SRY > arenaSize[3]){
					rs.SRY = arenaSize[3];
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

		SRState rs = (SRState)state;

		g.setColor(Color.WHITE);
		g.fillRect((int)(arenaSize[0]), (int)(arenaSize[1]), (int)(arenaSize[2]), (int)(arenaSize[3]));
		g.setColor(Color.BLACK);
		g.drawRect((int)(arenaSize[0]), (int)(arenaSize[1]), (int)(arenaSize[2]), (int)(arenaSize[3]));
		
		g.setColor(Color.GREEN);
		g.fillOval((int)(rs.foodX - foodRadius), (int)(rs.foodY - foodRadius), (int)(foodRadius * 2), (int)(foodRadius * 2));
		g.setColor(Color.BLACK);
		g.drawOval((int)(rs.foodX - foodRadius), (int)(rs.foodY - foodRadius), (int)(foodRadius * 2), (int)(foodRadius * 2));

		g.setColor(Color.GRAY);
		g.fillArc((int)(rs.SRX - SRRadius), (int)(rs.SRY - SRRadius), (int)(SRRadius * 2), (int)(SRRadius * 2), 0, (int)((rs.SRHealth / 100.0) * 360));
		g.setColor(Color.BLACK);
		g.drawOval((int)(rs.SRX - SRRadius), (int)(rs.SRY - SRRadius), (int)(SRRadius * 2), (int)(SRRadius * 2));

		return true;
	}
}