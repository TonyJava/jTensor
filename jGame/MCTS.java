package jGame;

import java.util.ArrayList;
import java.util.Random;

public class MCTS {

    final int branch = 100; //set to max
    final double epsilon = 1e-5;
    double exploration = 1.4;
    int playouts;
    Game game;
    int player;
    Node start;
    GameState board_state; // root state

    long selectionTime = 0;
    long expansionTime = 0;
    long simulationTime = 0;
    long backpropTime = 0;

    long sims = 0;
    long totalMoves = 0;

    HeavyPlayoutCall heavyPlayoutCall = null;

    final double probabilityWeight = 1;
    MoveProbabilityCall moveProbabilityCall = null;

    public static class MoveProbabilityCall{
        public double[] getProbabilities(GameState gameState){
            return null;
        }
    }

    public static class HeavyPlayoutCall{
        public int getHeavyPlayoutMove(GameState gameState){
            return -1;
        }
    }

    public MCTS(Game game, GameState board_state, int playouts){
        this.playouts = playouts;
        this.game = game;
        this.board_state = board_state;
        this.player = board_state.getPlayerMove();
        start = new Node(null, (byte)-1);
    }

    public void setMoveProbabilityCall(MoveProbabilityCall mpc){
        moveProbabilityCall = mpc;
    }

    public void setHeavyPlayoutCall(HeavyPlayoutCall hpc){
        heavyPlayoutCall = hpc;
    }

    public int getMove() {

        // System.out.println("selectionTime: " + selectionTime);
        // System.out.println("expansionTime: " + expansionTime);
        // System.out.println("simulationTime: " + simulationTime);
        // System.out.println("backpropTime: " + backpropTime);


        selectionTime = 0;
        expansionTime = 0;
        simulationTime = 0;
        backpropTime = 0;

        if(start.children.isEmpty()){
            return -1;
        }

        // System.out.print("Children: ");

        int move = start.children.get(0).move;
        long high = -2;
        long totalSims = 0;
        double winPercentage = 0;
        for (int j = 0; j < start.children.size(); j++) {
            Node n = start.children.get(j);
        // System.out.print(n.move + ", ");

            if (n.finished == player) {
                return n.move;
            } else if (n.finished == 0 && n.sims > high) {
                high = n.sims;
                winPercentage = ((double)n.wins)/(n.sims + 1);
                move = n.move;
            } else if (n.finished == -2 && -1 > high) { //can draw
                high = -1;
                move = n.move;
            }
        }

        // System.out.print("\n");


        // System.out.println("Move Confidence: " + (double)high/(start.sims + 1));
        // System.out.println("Win Confidence: " + winPercentage);

        return move;
    }

    public boolean round() { //returns solved
        if (start.finished != 0) {
            // System.out.print("(SOLVED FOR: " + start.finished+") " + start.children.size() + ":");
            return true;
        }

        GameState board = board_state.createCopy();
        // for(int j = 0; j < board_state.size(); j++){
        //     board.set(j, board_state.get(j));
        // }

        long currentTime;
        
        long lastTime = System.currentTimeMillis();
        Node n = selection(board);
        currentTime = System.currentTimeMillis();
        selectionTime += currentTime - lastTime;
        lastTime = currentTime;

        if (n == null) { //solved a node
            return false;
        }

        Node leaf = expansion(n, board);
        currentTime = System.currentTimeMillis();
        expansionTime += currentTime - lastTime;
        lastTime = currentTime;

        if (leaf == null) { // reached a termating leaf node
            leaf = n;
        }


        for (int j = 0; j < playouts; j++) {

            int result = simulation(board);
            currentTime = System.currentTimeMillis();
            simulationTime += currentTime - lastTime;
            lastTime = currentTime;

            if (leaf == n) {

                // if (result == -2) {
                //     result = 3;
                // }
                n.finished = result;
                j = playouts;
            }

            Node p = leaf;
            while (p != null) {
                if (result == p.player) {
                    p.wins++;
                }
                p.sims++;
                p = p.parent;
            }
            currentTime = System.currentTimeMillis();
            backpropTime += currentTime - lastTime;
            lastTime = currentTime;
        }

        return false;
    }

    public Node selection(GameState board) { //select a leaf node return index, board is passed with the current state

        Node currentNode = start;
        int currentState = 0;

        while (currentNode.children.size() != 0) {

            totalMoves++;

            int index = -1;
            double high = -1;
            int finishedtype = 0; //-3 not yet solveable, -2 can draw, -1 can't win & can't draw & can't be solved (only in 3+ player games)

            boolean moveByNature = board.getPlayerMove() == -1;

            for (int j = 0; j < currentNode.children.size(); j++) {
                Node n = currentNode.children.get(j);

                // UPDATED: wins is only incremented if it is a win for the person who's move the node represents (new field int Node.player)
                // if (board.getPlayerMove() != player) {
                //     wins = n.sims - n.wins; //opponents move
                // }

                if (n.finished == 0) {
                    long wins = n.wins;
                    double winMult = 0;
                    double moveProbability = 0;
                    if(moveProbabilityCall != null){
                        moveProbability = n.moveProbability;
                    }
                    double uct = moveProbability  * probabilityWeight + (double)wins / (n.sims + epsilon) + exploration * Math.sqrt((Math.log(currentNode.sims + 2) / (n.sims + epsilon)));
                    if (uct > high) {
                        high = uct;
                        index = j;
                        finishedtype = -3;
                    }
                }else{
                    if (n.finished == board.getPlayerMove()) {
                        currentNode.finished = n.finished;
                        // currentNode.deleteChildren();
                        return null;
                    }else if(n.finished == -2 && index == -1){
                        finishedtype = -2;
                        index = j;
                    }
                    if (finishedtype == 0) {
                        finishedtype = n.finished;
                    } else if (finishedtype != n.finished && finishedtype >= -1 && n.finished >= -1) { // if finishedtype is different and there haven't been draws or unfinished nodes
                        finishedtype = -1;
                    }
                }
            }

            // if(finishedtype == -1){
            //     finishedtype = -2;
            // }

            

            if (finishedtype != -3) { // if something got solved
                currentNode.finished = finishedtype;
                // currentNode.deleteChildren();
                return null;
            } else {
                if(moveByNature){ // Move by nature (sim random move)
                    int childMove = game.getMoveByNature(board);
                    boolean found = false;
                    for(Node child: currentNode.children){
                        if(child.move == childMove){
                            currentNode = child;
                            found = true;
                            break;
                        }
                    }
                    if(!found){
                        System.out.println("Problem choosing move");
                    }
                    // currentNode = currentNode.children.get(childSelect);
                    currentState = game.simMove(currentNode.move, board);
                    continue;
                }
                if (index != -1) {
                    currentNode = currentNode.children.get(index);
                    currentState = game.simMove(currentNode.move, board);
                    if(currentState ==  -1){
                        System.out.println("Should not happen 425: " + currentNode.move);
                        // PlayGame.printList(board);
                    }
                } else if (index == -1) {
                    System.out.println("SHOULDNT HAPPEN");
                    currentNode.finished = finishedtype;
                    return null;
                }
            }
        }
        if(currentState != 0){
            currentNode.finished = currentState;//((currentState == -2) ? -2 : currentState);
            return null;
        }
        return currentNode;
    }

    public Node expansion(Node n, GameState board) { // choose child of leaf node

        ArrayList<Integer> moves = game.legalMoves(board);
        // if(n.parent == null){
        //     System.out.println("Root expansion: " + moves.size());
        // }

        double[] moveProbabilities = null;
        if(moveProbabilityCall != null){
            moveProbabilities = moveProbabilityCall.getProbabilities(board);
        }

        Node c = null;
        int nPlayerMove = board.getPlayerMove();
        for(int j = 0; j < moves.size(); j++){
            c = new Node(n, moves.get(j));
            c.player = nPlayerMove;
            if(moveProbabilityCall != null){
                c.moveProbability = moveProbabilities[j];
            }
            n.children.add(c);
        }

        if (c == null) {
            return null;
        }else{
            c = n.children.get(((int)(Math.random() * n.children.size())));
        }

        game.simMove(c.move, board);
        return c;
    }

    public int simulation(GameState board) { //random playout until end, returns player who won, -1 for draw
        int result = game.getState(board);
        if (result != 0) {
            return result;
        }
        // }
        // GameState board_copy = new ArrayList<Byte>(board.size());
        // for (Byte i : board) {
        //     board_copy.add(i);
        // }
        // boolean printEnd = totalMoves == 0;

        ArrayList<Integer> moves = null;
        int moveMade = 0;
        while (result == 0) {
            totalMoves++;
            
            if(heavyPlayoutCall != null){
                moveMade = heavyPlayoutCall.getHeavyPlayoutMove(board);
            }else{
                moves = game.legalMoves(board);
                if (moves.isEmpty()) {
                    return -1;
                }
                moveMade = moves.get((int) (Math.random() * moves.size()));
            }
            result = game.simMove(moveMade, board);
        }

        // if(printEnd){
        //     PlayLinkIt.printBoard(board);
        //     System.out.println("first State: " + new LinkIt().getState(board) + ", " + result);
        //     System.out.print("first legal moves: ");
        //     for(Byte b: moves){
        //         System.out.print(b + ", ");
        //     }
        //     System.out.println("\nFirst sim length: " + totalMoves);
        //     System.out.println("\nFirst last move made: " + moveMade);
        // }
        return result;

    }

}
