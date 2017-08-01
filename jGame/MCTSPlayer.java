package jGame;

import java.util.ArrayList;

public class MCTSPlayer extends Player {

    MCTS ai;
    double seconds;
    int playouts;
    boolean weightMinimax = false;
    MCTS.HeavyPlayoutCall heavyPlayoutCall = null;
    MCTS.MoveProbabilityCall moveProbabilityCall = null;

    double explorationValue = 1.4;

    public MCTSPlayer(Game game, double seconds, int playouts) {
        super(game);
        this.seconds = seconds;
        this.playouts = playouts;
        ai = null;
    }

    public void setMoveProbabilityCall(MCTS.MoveProbabilityCall mpc){
        moveProbabilityCall = mpc;
    }

    public void setHeavyPlayoutCall(MCTS.HeavyPlayoutCall hpc){
        heavyPlayoutCall = hpc;
    }

    public void restart() {
        ai = null;
    }

    @Override
    public void update(int lastMove) {
        if (ai != null) {
            boolean changed = false;
            for (Node n : ai.start.children) {
                if (lastMove == n.move) {
                    ai.start = n;
                    game.simMove(n.move, ai.board_state);
                    ai.start.parent = null;
                    changed = true;
                    break;
                }
            }
            if (!changed) {
                restart();
            }
        }
        ai = null;
    }

    public int getMove(GameState gameState){

        if (ai == null) {
            GameState board_state = gameState.createCopy();
            ai = new MCTS(game, board_state, playouts);
            if(heavyPlayoutCall != null){
                ai.setHeavyPlayoutCall(heavyPlayoutCall);
            }
            if(moveProbabilityCall != null){
                ai.setMoveProbabilityCall(moveProbabilityCall);
            }
            ai.exploration = explorationValue;
        }

        boolean solved = false;
        long start = System.currentTimeMillis();
        int rounds = 1;
        while ((!solved) && (System.currentTimeMillis() - start) <= seconds * 1000) {
        // while ((!solved) && (rounds) <= seconds * 1000) {
            rounds++;
            solved = ai.round();
        }
        // System.out.println("Tree size:"+ai.start.getSize());
        // System.out.println("Average Depth:"+(int)(ai.start.avergeDepth()));
        // System.out.println("Average Playout:"+(int)(ai.totalMoves/rounds));
       
        // System.out.println("D: Sims:"+ai.start.sims);
        // System.out.println("D: Nodes "+ai.start.getSize());
        // System.out.println("D: Depth "+(int)(ai.start.avergeDepth()));
        // System.out.println("D: Playout: "+(int)(ai.totalMoves/rounds));
        // System.out.println("D: Branch: "+(int)(Math.pow(ai.start.getSize(), (double)1/ai.start.avergeDepth())));


        ai.totalMoves = 0;

        int move = ai.getMove();

        // System.out.println("M: " + move);

        if(!game.legalMoves(gameState).contains(move)){
            System.out.println("Illegal: " + move);
            System.out.println(gameState);
        }

        // update(move);

        return move;
    }
}
