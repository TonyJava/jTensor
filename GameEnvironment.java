import jREC.*;
import jGame.*;

import java.awt.Graphics;

public class GameEnvironment extends Environment{

	Game game;
	Player opponent;

	public GameEnvironment(Game game, Player opponent){
		this.game = game;
		this.opponent = opponent;
	}

	public class State extends Info{
		GameState gameState;
		boolean playerFirst;

		public State(){
			gameState = game.newGame();
			playerFirst = Math.random() < .5;
		}

		public double[] getDouble1(){ 
			double[][][] gameRep = gameState.getVolumeRepresentation();
			double[] gameUnrolled = new double[gameRep.length * gameRep[0].length * gameRep[0][0].length];
			for(int x = 0; x < gameRep.length; x++){
				for(int y = 0; y < gameRep[0].length; y++){
					for(int z = 0; z < gameRep[0][0].length; z++){
						gameUnrolled[x*gameRep[0].length*gameRep[0][0].length+y*gameRep[0][0].length+z] = gameRep[x][y][z];
					}
				}
			}
			return gameUnrolled;
		}
	}

	public class Action extends Info{
		int[] action;

		public Action(){
			action = new int[1];
		}

		@Override
		public int[] getInt1(){
			return action;
		}
	}

	// Returns amount/type of info available
	public Space getObservationSpace(){
		Space obsSpace = new Space();
		obsSpace.type = Space.Type.DISCRETE;
		State s = new State();
		double[][][] volumeRep = s.gameState.getVolumeRepresentation();
		int[] t_dims = {volumeRep.length * volumeRep[0].length * volumeRep[0][0].length};
		obsSpace.dimensions = t_dims;
		return obsSpace;
	}

	// Returns number of actions available
	// Caution gets num of outputs based on legal moves from initial state
	public Space getActionSpace(){
		State s = new State();
		Space actSpace = new Space();
		actSpace.type = Space.Type.DISCRETE;
		int[] t_dims = {game.legalMoves(s.gameState).size()};
		actSpace.dimensions = t_dims;
		return actSpace;
	}

	// Resets the environment, returns initial observation
	public Info reset(){
		State infoState = new State();
		state = infoState;
		opponent.restart();
		if(!infoState.playerFirst){
			int move = opponent.getMove(infoState.gameState);
			game.simMove(move, infoState.gameState);
		}
		return state;
	}

	// Returns initialized action object
	public Info createAction(){
		return new Action();
	}

	// Implementation for rendering env as a string
	// Used for console (or any non-JFrame) output
	// Will be called if draw(null, null) returns false
	protected void renderString(Info state){
		System.out.println(((State)state).gameState);
	}

	// Implementation for renderer, return true on success
	// If observation or g is null return true if implemented
	protected boolean draw(Graphics g,  Info observation){
		if(g == null && observation == null){
			return true;
		}
		game.drawBoard(g, ((State)observation).gameState, screenSize);
		return true;
	}

	// Implementation for next state
	protected ROF nextState(Info state, Info action){
		State infoState = ((State)state);
		int playerMove = ((Action)action).action[0];
		int finishedState = game.simMove(playerMove, infoState.gameState);
		opponent.update(playerMove);
		boolean finished = finishedState != 0;
		if(!finished){
			int opponentMove = opponent.getMove(infoState.gameState);
			finishedState = game.simMove(opponentMove, infoState.gameState);
			finished = finishedState != 0;
		}
		ROF rof = new ROF();
		rof.observation = infoState;
		rof.finished = finished;
		if(finished){
			// System.out.println(finishedState);
			rof.reward = ((infoState.playerFirst ? 1 : 2) == finishedState) ? 1 : -1;
		}else{
			rof.reward = 0;
		}
		rof.nextState = state;
		return rof;
	}

}