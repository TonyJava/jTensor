package jGame;

import java.awt.*;
import java.util.ArrayList;

public class TicTacToe extends Game {


    public TicTacToe() {
    }

    class TicTacToeState extends GameState{

        byte[] board;

        TicTacToeState(){
            board = new byte[10];
        }

        public double[][][] getVolumeRepresentation(){
            TicTacToeState originalState = this;
            double[][][] gameState = new double[3][3][3];

            byte[] rawMoves = ((TicTacToeState)originalState).board;
            int player = rawMoves[0];

            for(int x = 0; x < 3; x++){
                for(int y = 0; y < 3; y++){
                    int space = 1 + y + x*3;
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
            board = new byte[10];
            board[0] = 1;
            for(int j = 1; j < 10; j++){
                board[j] = 0;
            }
        }

        public GameState createCopy(){
            TicTacToeState newState = new TicTacToeState();
            for(int j = 0; j < board.length; j++){
                newState.board[j] = board[j];
            }
            return newState;
        }

    }

    public GameState newGame() {
        TicTacToeState newState = new TicTacToeState();
        newState.resetState();
        return newState;
    }

    
    public ArrayList<Integer> legalMoves(GameState originalGameState){
        TicTacToeState gameState = (TicTacToeState)originalGameState;
        ArrayList<Integer> moves = new ArrayList<Integer>();

        for(int j = 0; j <= 8; j++){
            if(gameState.board[j + 1] == 0){
                moves.add(j);
            }
        }
        return moves;
    }
    
    public int simMove(int m, GameState originalGameState){ //same as above except operates on board, returns victory state
        TicTacToeState gameState = (TicTacToeState)originalGameState;
        m += 1;
        if(gameState.board[m] == 0){
            gameState.board[m] = gameState.board[0];
            int ret = getState(gameState);
            int turn = gameState.board[0] - 1;
            gameState.board[0] = (byte)(2-turn);
            return ret;
        }else{
            return -1;
        }
    }

    public int getState(GameState originalGameState) {
        TicTacToeState gameState = (TicTacToeState)originalGameState;
    
        int player;
        for(int j = 1; j <= 7; j+=3){ //hor
            if(gameState.board[j] != 0 && gameState.board[j] == gameState.board[j+1] && gameState.board[j+1] == gameState.board[j+2]){
                return gameState.board[j];
            }
        }
        for(int j = 1; j <= 3; j++){ //vert
            if(gameState.board[j] != 0 && gameState.board[j] == gameState.board[j+3] && gameState.board[j+3] == gameState.board[j+6]){
                return gameState.board[j];
            }
        }
        //diags
        player = gameState.board[5];
        if(player != 0 && ((gameState.board[1]==player && gameState.board[9]==player) || (gameState.board[3]==player && gameState.board[7]==player))){
            return player;
        }
        for(int j = 1; j <= 9; j++){
            if(gameState.board[j]==0)
            return 0;
        }
        return -2;
    }


    public int getMoveFromMouse(Point mouse, GameState originalGameState, int[] screen, HumanPlayer player){

        int width = screen[2];
        int x = (int)(mouse.x - (screen[0]));
        int chooseSpace = width / 3;
        int xmove = x/chooseSpace;

        int y = (int)(mouse.y - (screen[1]));
        int ymove = y/chooseSpace;
        int move = ymove * 3 + xmove;

        if(move < 0 || move > 8){
            move = 0;
        }
        return move;
    }

    public void drawBoard(Graphics g, GameState originalGameState, int[] screen) {
        TicTacToeState gameState = (TicTacToeState)originalGameState;

        int width = screen[2];

        int r = width/6;

        int space = 2 * r;

        g.setColor(Color.WHITE);
        g.fillRect(screen[0], screen[1], screen[2], screen[3]);


        for(int j = 1; j <= 9; j++){

            int x = (j - 1) % 3;
            int y = ((j - 1) / 3);

            if(gameState.board[j] == 1){
                g.setColor(Color.RED);
                g.drawLine(screen[0] + (r + x * space) - r, screen[1] + (r + y * space) - r, screen[0] + (r + x * space) + r, screen[1] + (r + y * space) + r);
                g.drawLine(screen[0] + (r + x * space) + r, screen[1] + (r + y * space) - r, screen[0] + (r + x * space) - r, screen[1] + (r + y * space) + r);
            }else if(gameState.board[j] == 2){
                g.setColor(Color.BLUE);
                g.fillOval(screen[0] + (r + x * space) - r, screen[1] + (r + y * space) - r, screen[0] + (r + x * space) + r, screen[1] + (r + y * space) + r);
            }

        }

        g.setColor(Color.BLACK);
        for(int j = 0; j < 4; j++){
            g.drawLine(screen[0], screen[1] + space*(j), screen[0] + screen[2], screen[1] + space*(j));
            g.drawLine(screen[0] + space*(j), screen[1], screen[0] + space*(j), screen[1] + screen[3]);
        }


    }
    
}
