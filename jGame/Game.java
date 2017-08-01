package jGame;

import java.util.ArrayList;
import java.awt.Graphics;

public abstract class Game {

    // newGame returns an initial game state object
    public abstract GameState newGame();

    // legalMoves returns an arraylist of moves (represented as bytes), if the game is over, the list should be empty
    public abstract ArrayList<Integer> legalMoves(GameState board);

    // Used to implement random events (abstracted as movesby nature)
    public int getMoveByNature(GameState board){
        return -1;
    }

    public int getHeavyPlayoutMove(GameState board){
        return -1;
    }

    public double[] getMoveProbabilities(GameState originalState){
        return null;
    }

    // simMove plays move m on game state board, board will be altered to the next game state.
    // returns: 0 if game is not over, returns player who won, -1 if illegal move, -2 for draw
    public abstract int simMove(int m, GameState board);

    // returns: 0 if game is not over, returns player who won, -1 if illegal move, -2 for draw
    public abstract int getState(GameState board);

    // boardScreen is [x, y, width, height]
    public abstract void drawBoard(Graphics g, GameState board, int[] boardScreen);

    public abstract int getMoveFromMouse(Point mouse, GameState board, int[] boardScreen, HumanPlayer player);

    public void drawMouseState(Graphics g, HumanPlayer player, GameState gameState, int[] boardScreen){

    }
}
