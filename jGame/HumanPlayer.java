package jGame;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class HumanPlayer extends Player{

    Point mouse;
    int[] screen = null;

    // This can be used to store information for multiclick moves, return -1 from getMove to continue getting input
    Object moveState = null;

    public HumanPlayer(Game game){
        super(game);
        this.screen = screen;
        this.mouse = new Point(-1, 0);
    }
    
    public int getMove(GameState gameState){
        int move = game.getMoveFromMouse(mouse, gameState, screen, this);
        return move;
    }
}
