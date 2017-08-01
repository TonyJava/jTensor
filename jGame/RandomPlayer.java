package jGame;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class RandomPlayer extends Player{

    Point mouse;

    // This can be used to store information for multiclick moves, return -1 from getMove to continue getting input
    Object moveState = null;

    public RandomPlayer(Game game){
        super(game);
        this.mouse = new Point(-1, 0);
    }
    
    public int getMove(GameState gameState){
        ArrayList<Integer> moves = game.legalMoves(gameState);
        int move = moves.get((int)(Math.random() * moves.size()));
        return move;
    }
}
