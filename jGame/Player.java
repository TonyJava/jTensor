package jGame;

import java.util.ArrayList;

public abstract class Player {
    protected Game game;

    public Player(Game game){
        this.game = game;
    }

    // returns move to make on board, -1 is resign
    public abstract int getMove(GameState gameState);

    // this method is called when a new game begins
    public void restart(){}

    // this method is called anytime another player makes a move, which is lastMove
    public void update(int lastMove){}
}
