package jGame;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Stack;
import java.lang.StringBuilder;

public class Go extends Game {

    int boardSize;
    int totalPoints;
    int stateSize;
    int koIndex;
    int passIndex;

    int[] dirs;

    int maxGameLength;

    boolean trackMoves = false;

//board state: 1 is top left point, goes right, 82 is number of consecutive passes, 83 is (0/else=no ko, move which is illegal), 84 is white stones captured by black, 85 is whites pts for capture, 86 is number of turns


    public class GoGameState extends GameState{
        
        byte[] board;
        byte[] moveNumbers;

        public GoGameState(){
            board = new byte[stateSize];
            board[0] = 1;
            if(trackMoves){
                moveNumbers = new byte[totalPoints];
            }
        }

        public int getPlayerMove(){
            return board[0];
        }

        public void resetState(){
            board[0] = 1;
            for(int j = 1; j < stateSize; j++){
                board[j] = 0;
            }
            if(trackMoves){
                for(int j = 0; j < totalPoints; j++){
                    moveNumbers[j] = -1;
                }
            }
        }

        public GameState createCopy(){
            GoGameState state = new GoGameState();
            for(int j = 0; j < stateSize; j++){
                state.board[j] = board[j];
            }
            if(trackMoves){
                for(int j = 0; j < totalPoints; j++){
                    state.moveNumbers[j] = moveNumbers[j];
                }
            }
            return state;
        }

        public double[][][] oldGetVolumeRepresentation(){
            GoGameState originalState = this;
            double[][][] gameState = new double[9][9][3];

            byte[] rawMoves = ((GoGameState)originalState).board;
            byte[] moveNumbers = ((GoGameState)originalState).moveNumbers;
            int player = rawMoves[0];

            for(int x = 0; x < 9; x++){
                for(int y = 0; y < 9; y++){
                    int space = 1 + y*9 + x;
                    int value = rawMoves[space];
                    int activationIndex = 0;
                    if(value == player){
                        activationIndex = 1;
                    }else if(value == 3-player){
                        activationIndex = 2;
                    }    
                    
                    for(int j = 0; j < 3; j++){
                        gameState[x][y][j] = (j == activationIndex) ? 1 : 0;
                    }
                }
            }

            return gameState;
        }
	
	@Override    
	public double[][][] getVolumeRepresentation(){
		double[][][] gameState = new double[boardSize][boardSize][12]; // 3 for position status (player/opponent/empty) 5 for liberties of the chain (1/2/3/4/5 or more) 4 for most recent 3 moves (0/1/2/3 or more)
		GoGameState originalState = this;
		byte[] rawMoves = ((GoGameState)originalState).board;
		byte[] moveNumbers = ((GoGameState)originalState).moveNumbers;
		int player = rawMoves[0];

		for(int x = 0; x < boardSize; x++){
		    for(int y = 0; y < boardSize; y++){
			int space = 1 + y*boardSize + x;
			int value = rawMoves[space];
			int stoneColorIndex;
			int libertiesIndex = -1; // 1 of (n-1) encoding (non stones dont have a liberties value at any depth)
			int recentMoveIndex = -1;
			if(value == 0){
			    stoneColorIndex = 2;
			}else{
			    recentMoveIndex = (rawMoves[koIndex + 3] - moveNumbers[space - 1] > 2 ? 3 : (rawMoves[koIndex + 3] - moveNumbers[space - 1])) + 8;
			    if(value == player){
				stoneColorIndex = 0;
			    }else{
				stoneColorIndex = 1;
			    }
			    int currentLiberties = getLiberties(rawMoves, space);
			    libertiesIndex = currentLiberties >= 5 ? 7 : currentLiberties + 2;
			    if(libertiesIndex < 3){
				System.out.println("PROBLEMLIBERTIES: " + rawMoves[rawMoves.length - 1]);
				libertiesIndex = -1;
			    }
			}
			for(int j = 0; j < 12; j++){
			    gameState[x][y][j] = (j == stoneColorIndex || j == libertiesIndex || j == recentMoveIndex) ? 1 : 0;
			}
		    }
		}

		return gameState;
	}

        @Override
        public String toString(){
            StringBuilder sb = new StringBuilder();
            sb.append(board[0] + "\n");
            for(int j = 0; j < boardSize; j++){
                for(int i = 0; i < boardSize; i++){
                    int value = board[j*boardSize + i + 1];
                    String spot = "";
                    switch(value){
                        case 0: spot = "_";break;
                        case 1: spot = "O";break;
                        case 2: spot = "X";break;
                    }

                    sb.append(spot+" ");
                }
                sb.append("\n");
            }
            if(trackMoves){
               for(int j = 0; j < totalPoints; j++){
                sb.append(moveNumbers[j]+", ");
                }
            }
            return sb.toString();
        }
    }

// moves 1-81 and 0 is pass

    //Komi is 7.5, only stones left on board count as points (not spaces)
    public Go(int boardSize) {
        this.boardSize = boardSize;
        totalPoints = boardSize * boardSize;
        stateSize = totalPoints + 6;
        passIndex = totalPoints + 1;
        koIndex = totalPoints + 2;
        maxGameLength = (int)(totalPoints * 1.5);
        dirs = new int[4];
        dirs[0] = -1;
        dirs[1] = -boardSize;
        dirs[2] = 1;
        dirs[3] = boardSize;
    }

    public void setTrackMoves(boolean trackMoves){
        this.trackMoves = trackMoves;
    }

    public GameState newGame() {
        return new GoGameState();
    }

    public ArrayList<Integer> legalMoves(GameState originalState) {
        GoGameState gameState = (GoGameState)originalState;

        ArrayList<Integer> moves = new ArrayList<Integer>();

        moves.add(0);
        byte illegalMove = gameState.board[koIndex];


        for (int j = 1; j <= totalPoints; j++) {
            if (gameState.board[j] != 0 || j == illegalMove) {
                continue;
            }
            if(isLegal(j, gameState.board)){
                moves.add(j);
            }
        }
        return moves;
    }

    public boolean isLegal(int m, byte[] board) {

        // if player passes
        if(m == 0){
            return true;
        }

        //Moving on a stone or illegal ko
        if(m < 0 || m > totalPoints || board[m] != 0 || m == board[koIndex]){
            //System.out.println("Illegal: " + m + " = " + board[m]);
            return false;
        }

        byte turn = board[0];

        boolean returnValue = true;

        byte temp = board[m];
        board[m] = turn;

        //check for suicide
        // byte[] checkBoard = new byte[totalPoints + 1];
        // for(int j = 0; j < totalPoints + 1; j++){
        //     checkBoard[j] = 0;
        // }
        if(!hasLiberties(board, m)){
            if(!checkForCapturesConst(board, m)){
                returnValue = false;
            }
        }

        board[m] = temp;

        return returnValue;
    }

    //board state: 1 is top left point, goes right, 82 is number of consecutive passes, 83 is (0/else=no ko, move which is illegal), 84 is white stones captured by black, 85 is whites pts for capture
// moves 1-81 and 0 is pass
    public int simMove(int m, GameState originalState) {
        GoGameState gameState = (GoGameState)originalState;

        int totalMoves = gameState.board[koIndex + 3] + 1;
        gameState.board[koIndex + 3] = (byte)(totalMoves);
        if(totalMoves >= maxGameLength){
            return getState(gameState);
        }

        // if player passes
        if(m == 0){
            gameState.board[passIndex] = (byte)(gameState.board[passIndex] + 1);
            if(gameState.board[passIndex] == 2){
                // game over: 2 passes
                int state = getState(gameState);
                if (gameState.board[0] == 1) {
                    gameState.board[0] = (byte)2;
                } else {
                    gameState.board[0] = (byte)1;
                }
                return state;
            }else{
                if (gameState.board[0] == 1) {
                    gameState.board[0] = (byte)2;
                } else {
                    gameState.board[0] = (byte)1;
                }
                return 0;
            }
        }

        //Moving on a stone or illegal ko
        if(m < 0 || m > totalPoints || gameState.board[m] != 0 || m == gameState.board[koIndex]){
            //System.out.println("Illegal: " + m + " = " + gameState.board[m]);
            return -1;
        }

        gameState.board[koIndex] = (byte)0; // reset o

        byte turn = gameState.board[0];

        //place move check for captures
        gameState.board[m] = turn;
        int captures = checkForCaptures(gameState.board, m);

        if(!hasLiberties(gameState.board, m)){
            //System.out.println("Suicide: " + m);
            gameState.board[m] = (byte)0;
            return -1;
        }

        if(captures == 1 && isAlone(gameState.board, m, turn)){ // potential ko
            if(getLiberties(gameState.board, m) == 1){
                // ko exists
                gameState.board[koIndex] = (byte)getAtari(gameState.board, m);
            }
        }

        if(trackMoves){
          gameState.moveNumbers[m - 1] = (byte)totalMoves;
        }


        gameState.board[passIndex] = (byte)0;

        gameState.board[0] = (byte)(3-gameState.board[0]);

        return 0;
    }

    public int getState(GameState originalState) {
        GoGameState gameState = (GoGameState)originalState;


        int totalMoves = gameState.board[koIndex + 3];
        if(totalMoves < maxGameLength && gameState.board[passIndex] != 2){
            return 0;
        }

        int[] points = new int[2];

        //add points for captures
        points[0] = gameState.board[koIndex + 1];

        //white gets komi also
        points[1] = gameState.board[koIndex + 2] + 7;
        for(int j = 1; j <= totalPoints; j++){
            byte spaceValue = gameState.board[j];
            if(spaceValue > 0){
                points[spaceValue - 1] += 1;
            }
        }


        //white wins tie (ie white gets a half pt)
        return (points[1] >= points[0] ? 2 : 1);
    }


    public int getMoveFromMouse(Point mouse, GameState originalState, int[] originalScreen, HumanPlayer player){
        GoGameState gameState = (GoGameState)originalState;

        int originalWidth = (int)(originalScreen[2]);
        int originalHeight = (int)(originalScreen[3]);

        int xb = 0;
        int yb = 0;

        if(originalWidth > originalHeight){
            xb = (originalWidth - originalHeight) / 2;
            originalWidth = originalHeight;
        }else{
            yb = (originalHeight - originalWidth) / 2;
            originalHeight = originalWidth;
        }

        int[] screen = new int[]{originalScreen[0] + xb, originalScreen[1] + yb, (originalScreen[0] + originalWidth) - xb, (originalScreen[1] + originalHeight) - yb};

        int width = (int)(screen[2] - screen[0]);

        int chooseSpace = width / boardSize;

        int x = (int)(mouse.x - (screen[0]));
        int y = (int)(mouse.y - (screen[1]));


        int xmove = x/chooseSpace;
        int ymove = y/chooseSpace;

        int move = ymove * boardSize + xmove + 1;

        // System.out.println(move);

        if(move < 1 || move > totalPoints){
            move = 0;
        }
        return move;
    }

    public void drawBoard(Graphics g, GameState originalState, int[] originalScreen) {
        GoGameState gameState = (GoGameState)originalState;

        int originalWidth = (int)(originalScreen[2]);
        int originalHeight = (int)(originalScreen[3]);

        int xb = 0;
        int yb = 0;

        if(originalWidth > originalHeight){
            xb = (originalWidth - originalHeight) / 2;
            originalWidth = originalHeight;
        }else{
            yb = (originalHeight - originalWidth) / 2;
            originalHeight = originalWidth;
        }

        int[] screen = new int[]{originalScreen[0] + xb, originalScreen[1] + yb, (originalScreen[0] + originalWidth) - xb, (originalScreen[1] + originalHeight) - yb};

        int width = (int)(screen[2] - screen[0]);

        int rFac = 3;

        double dx = width/(rFac * (boardSize * 2) + (boardSize + 1));
        double r = rFac * dx;

        double space = 2 * r + dx;


        g.setColor(new Color(238, 118, 33));
        g.fillRect(screen[0], screen[1], screen[0] + screen[2], screen[1] + screen[3]);

        g.setColor(Color.BLACK);
        for(int j = 0; j < boardSize; j++){
            double point = dx + r + space * j;
            g.drawLine((int)(screen[0] + dx + r), (int)(screen[1] + point), (int)(screen[0] + dx + r + (boardSize - 1) * space), (int)(screen[1] + point));
            g.drawLine((int)(screen[0] + point), (int)(screen[1] + dx + r), (int)(screen[0] + point), (int)(screen[1] + dx + r + (boardSize - 1) * space));
        }

        double hoshiSize = r / 3;
        double hoshiPoint = (boardSize / 2);
        g.fillOval((int)(screen[0] + (dx + r + hoshiPoint * space) - hoshiSize), (int)(screen[1] + (dx + r + hoshiPoint * space) - hoshiSize), (int)hoshiSize*2, (int)hoshiSize*2);

        for(int j = 1; j <= totalPoints; j++){
            switch(gameState.board[j]){
                case 0: continue;
                case 1: g.setColor(Color.BLACK); break;
                case 2: g.setColor(Color.WHITE); break;
            }

            int x = (j - 1) % boardSize;
            int y = ((j - 1) / boardSize);

            g.fillOval((int)(screen[0] + (dx + x * space)), (int)(screen[1] + (dx + y * space)), (int)r*2, (int)r*2);

            g.setColor(Color.BLACK);
            g.drawOval((int)(screen[0] + (dx + x * space)), (int)(screen[1] + (dx + y * space)), (int)r*2, (int)r*2);
        }



    }

    // Go functions

    int indexAndDir(int p, int j){
        int nextIndex = p + dirs[j];
        if((j == 0 && nextIndex % boardSize == 0) || (j == 2 && nextIndex % boardSize == 1) || (nextIndex <= 0) || ( nextIndex > totalPoints)){
            return -1;
        }
        return nextIndex;
    }

    // p is the index in the board of a stone just played
    boolean checkForCapturesConst(byte[] board, int p){
        byte turn = (byte)(board[p] - 1);
        for(int j = 0; j < 4; j++){
            int nextIndex = indexAndDir(p, j);
            if(nextIndex != -1 && board[nextIndex] == 2-turn){
                if(!hasLiberties(board, nextIndex)){
                    return true;
                }
            }
        }
        return false;
    }

    // p is the index in the board of a stone just played
    int checkForCaptures(byte[] board, int p){
        int captures = 0;
        byte turn = (byte)(board[p] - 1);
        for(int j = 0; j < 4; j++){
            int nextIndex = indexAndDir(p, j);
            if(nextIndex != -1 && board[nextIndex] == 2-turn){
                if(!hasLiberties(board, nextIndex)){
                    // System.out.println("Capturing at " + nextIndex);
                    int before = board[koIndex + 1 + turn];
                    captureGroup(board, nextIndex, turn);
                    captures += (board[koIndex + 1 + turn]) - before;
                }
            }
        }
        return captures;
    }

    void captureGroup(byte[] board, int p, byte turn){
        board[p] = (byte)0;
        board[koIndex + 1 + turn] = (byte)(board[koIndex + 1 + turn] + 1);
        for(int j = 0; j < 4; j++){
            int nextIndex = indexAndDir(p, j);
            if(nextIndex != -1 && board[nextIndex] == 2-turn){
                captureGroup(board, nextIndex, turn);
            }
        }
    }

    // p is the index in the board of a stone in the group to count liberties for
    boolean hasLiberties(byte[] board, int p){
        //System.out.println("hasLib: checking " + p);
        int turn = board[p];
        HashSet<Integer> checkedIndexes = new HashSet<Integer>();
        Stack<Integer> searchStack = new Stack<Integer>();
        searchStack.add(p);
        while(!searchStack.empty()){
            int currentIndex = searchStack.pop();
            for(int j = 0; j < 4; j++){
                int nextIndex = indexAndDir(currentIndex, j);
                if(nextIndex != -1 && !checkedIndexes.contains(nextIndex)){
                    if(board[nextIndex] == 0){
                        return true;
                    }else if(board[nextIndex] == turn){
                        searchStack.add(nextIndex);
                    }

                }
            }
            checkedIndexes.add(currentIndex);
        }
        return false;
    }

    int getLiberties(byte[] board, int p){
        int liberties = 0;
        int turn = board[p];
        HashSet<Integer> checkedIndexes = new HashSet<Integer>();
        Stack<Integer> searchStack = new Stack<Integer>();
        searchStack.add(p);
        while(!searchStack.empty()){
            int currentIndex = searchStack.pop();
            for(int j = 0; j < 4; j++){
                int nextIndex = indexAndDir(currentIndex, j);
                if(nextIndex != -1 && !checkedIndexes.contains(nextIndex)){
                    if(board[nextIndex] == 0){
                        liberties++;
                    }else if(board[nextIndex] == turn){
                        searchStack.add(nextIndex);
                    }

                }
            }
            checkedIndexes.add(currentIndex);
        }
        return liberties;
    }


    boolean isAlone(byte[] board, int p, int pColor){
        for(int j = 0; j < 4; j++) {
            int nextIndex = indexAndDir(p, j);
            if(nextIndex != -1 && board[nextIndex] == pColor){
                return false;
            }
        }
        return true;
    }

    int getAtari(byte[] board, int p){
        for(int j = 0; j < 4; j++) {
            int nextIndex = indexAndDir(p, j);
            if(nextIndex != -1 && board[nextIndex] == 0){
                return nextIndex;
            }
        }
        return -1;
    }
}
