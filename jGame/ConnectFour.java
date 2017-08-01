package jGame;

import java.awt.*;

import java.util.ArrayList;

public class ConnectFour extends Game {
//board state: 1 is bottom row left space, increasing going right. 8 is second row, 15 is third row.... 0 is empty space, 1 is p1, 2 is p2

    public ConnectFour() {

    }

    class ConnectFourState extends GameState{

        byte[] board;

        ConnectFourState(){
            board = new byte[43];
        }

        public double[][][] getVolumeRepresentation(){
            ConnectFourState originalState = this;
            double[][][] gameState = new double[6][7][3];

            byte[] rawMoves = ((ConnectFourState)originalState).board;
            int player = rawMoves[0];

            for(int x = 0; x < 6; x++){
                for(int y = 0; y < 7; y++){
                    int space = 1 + y + x*7;
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

        public int getPlayerMove(){
            return board[0];
        }

        public void resetState(){
            board = new byte[43];
            board[0] = 1;
        }

        public GameState createCopy(){
            ConnectFourState newState = new ConnectFourState();
            for(int j = 0; j < board.length; j++){
                newState.board[j] = board[j];
            }
            return newState;
        }

    }

    public int minimaxValue(GameState originalGameState, int player){
        ConnectFourState gameState = (ConnectFourState)originalGameState;

        int opponent = 3 - player;
        int state = getState(gameState);
        if(state == -2){
            return 0;
        }
        if(player == state){
            return 10000;
        }else if(opponent == state){
            return -10000;
        }

        int playerScore = 0;
        int opponentScore = 0;

        // count horizontals
        for(int j = 1; j <= 39; j += (j%7 == 4 ? 4 : 1)){
            int playerStones = 0;
            int opponentStones = 0;
            int emptySpaces = 0;
            for(int i = 0; i < 4 && (playerStones == 0 || opponentStones == 0); i++){
                int spaceValue = gameState.board[j + i];
                if(spaceValue == player){
                    playerStones++;
                }else if(spaceValue == opponent){
                    opponentStones++;
                }else{
                    emptySpaces++;
                }
            }
            if(playerStones == 0 || opponentStones == 0){
                if(playerStones > 1){
                    playerScore += 4 - emptySpaces;
                }else if(opponentStones > 1){
                    opponentScore += 4 - emptySpaces;
                }
            }
        }

        // count verticals
        for(int j = 1; j <= 21; j++){
            int playerStones = 0;
            int opponentStones = 0;
            int emptySpaces = 0;
            for(int i = 0; i < 4 && (playerStones == 0 || opponentStones == 0); i++){
                int spaceValue = gameState.board[j + i*7];
                if(spaceValue == player){
                    playerStones++;
                }else if(spaceValue == opponent){
                    opponentStones++;
                }else{
                    emptySpaces++;
                }
            }
            if(playerStones == 0 || opponentStones == 0){
                if(playerStones > 1){
                    playerScore += 4 - emptySpaces;
                }else if(opponentStones > 1){
                    opponentScore += 4 - emptySpaces;
                }
            }
        }

        // count diagonals +9
        for(int j = 1; j <= 18; j += (j%7 == 4 ? 4 : 1)){
            int playerStones = 0;
            int opponentStones = 0;
            int emptySpaces = 0;
            for(int i = 0; i < 4 && (playerStones == 0 || opponentStones == 0); i++){
                int spaceValue = gameState.board[j + i*8];
                if(spaceValue == player){
                    playerStones++;
                }else if(spaceValue == opponent){
                    opponentStones++;
                }else{
                    emptySpaces++;
                }
            }
            if(playerStones == 0 || opponentStones == 0){
                if(playerStones > 1){
                    playerScore += 4 - emptySpaces;
                }else if(opponentStones > 1){
                    opponentScore += 4 - emptySpaces;
                }
            }
        }

        // count diagonals+7
        for(int j = 21; j >= 4; j -= (j%7 == 4 ? 4 : 1)){
            int playerStones = 0;
            int opponentStones = 0;
            int emptySpaces = 0;
            for(int i = 0; i < 4 && (playerStones == 0 || opponentStones == 0); i++){
                int spaceValue = gameState.board[j + i*6];
                if(spaceValue == player){
                    playerStones++;
                }else if(spaceValue == opponent){
                    opponentStones++;
                }else{
                    emptySpaces++;
                }
            }
            if(playerStones == 0 || opponentStones == 0){
                if(playerStones > 1){
                    playerScore += 4 - emptySpaces;
                }else if(opponentStones > 1){
                    opponentScore += 4 - emptySpaces;
                }
            }
        }

        return playerScore - opponentScore;

    }

    public GameState newGame() {
        ConnectFourState newState = new ConnectFourState();
        newState.resetState();
        return newState;
    }

    public ArrayList<Integer> legalMoves(GameState originalGameState) {
        ConnectFourState gameState = (ConnectFourState)originalGameState;
        ArrayList<Integer> moves = new ArrayList<Integer>();

        for (int j = 36; j < 43; j++) {
            if (gameState.board[j] == 0) {
                moves.add((j - 36));
            }
        }
        return moves;
    }

    public int simMove(int m, GameState originalGameState) { // moves 0-6 corresponding to column left to right, index 0 is player who just moved
        
        ConnectFourState gameState = (ConnectFourState)originalGameState;

        int j;
        for (j = 6; j >= 1 && gameState.board[(7 * (j - 1) + m + 1)] == 0; j--) {
        }
        if (j == 6) {
            return -1;
        }
        gameState.board[7 * (j) + m + 1] = gameState.board[0];

        int state = getState(gameState);
        int turn = gameState.board[0] - 1;
        gameState.board[0] = (byte)(2-turn);
        return state;
    }

    public int getState(GameState originalGameState) {
        ConnectFourState gameState = (ConnectFourState)originalGameState;

        for (int j = 1; j < 43; j++) {
            if (gameState.board[j] != 0) {

                int color = gameState.board[j];
                int i;
                int row;
//check hor 
                if (j % 7 == 4) { //huge optimization
                    row = 1;
                    for (i = j; (i - 1) % 7 != 0 && gameState.board[i - 1] == color && i > 1; i--) {
                    }
                    while (((i + 1) % 7) != 1 && gameState.board[i + 1] == color) {
                        i++;
                        row++;
                        if (row == 4) {
                            return color;
                        }
                    }
                }
//check vert
                if ((j-1)/7 == 2) { //huge optimization
                    row = 1;
                    for (i = j; (i - 7) > 0 && gameState.board[i - 7] == color; i -= 7) {
                    }
                    while ((i + 7) <= 42 && gameState.board[i + 7] == color) {
                        i += 7;
                        row++;
                        if (row == 4) {
                            return color;
                        }
                    }

//check down-left diag
                    row = 1;
                    for (i = j; (i - 8) > 0 && (i - 8) % 7 != 0 && gameState.board[i - 8] == color; i -= 8) {
                    }
                    while ((i + 8) <= 42 && (i + 8) % 7 != 1 && gameState.board[i + 8] == color) {
                        i += 8;
                        row++;
                        if (row == 4) {
                            return color;
                        }
                    }

//check down-right diag
                    row = 1;
                    for (i = j; (i - 6) > 0 && (i - 6) % 7 != 1 && gameState.board[i - 6] == color; i -= 6) {
                    }
                    while ((i + 6) <= 42 && (i + 6) % 7 != 0 && gameState.board[i + 6] == color) {
                        i += 6;
                        row++;
                        if (row == 4) {
                            return color;
                        }
                    }
                }

            }
        }
        //changed
        for (int j = 36; j < 43; j++) {
            if (gameState.board[j] == 0) {
                return 0;
            }
        }
        return -2;
    }

    public int getMoveFromMouse(Point mouse, GameState originalGameState, int[] screen, HumanPlayer player){
        ConnectFourState gameState = (ConnectFourState)originalGameState;

        int width = screen[2];
        int height = screen[3];
        int dy = height/19;
        int r = dy;
        int dx = (width - r * 14) / 8;

        int x = (int)(mouse.x - (screen[0]));

        int chooseSpace = width / 7;

        int move = x/chooseSpace;

        if(move < 0 || move > 6){
            move = 0;
        }
        return move;
    }

    public void drawBoard(Graphics g, GameState originalGameState, int[] screen) {
        ConnectFourState gameState = (ConnectFourState)originalGameState;


        int width = screen[2];
        int height = screen[3];

        int dy = height/19;
        int r = dy;
        int dx = (width - r * 14) / 8;

        int xSpace = 2 * r + dx;
        int ySpace = 2 * r + dy;

        g.setColor(Color.YELLOW);
        g.fillRect(screen[0], screen[1], screen[2], screen[3]);

        for(int j = 1; j < 43; j++){
            switch(gameState.board[j]){
                case 0: g.setColor(Color.WHITE); break;
                case 1: g.setColor(Color.RED); break;
                case 2: g.setColor(Color.BLUE); break;
            }

            int x = (j - 1) % 7;
            int y = 5 - ((j - 1) / 7);



            g.fillOval(screen[0] + (dx + r + x * xSpace), screen[1] + (dy + r + y * ySpace), r, r);

            g.setColor(Color.BLACK);
            g.drawOval(screen[0] + (dx + r + x * xSpace), screen[1] + (dy + r + y * ySpace), r, r);

        }

    }
}
