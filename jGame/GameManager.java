package jGame;

import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
import java.io.*;


import java.util.ArrayList;

public class GameManager{

    boolean gameFinished = false;

    int gameStatus; // 0 continue, else player who won, 3 is draw
    boolean waitingForInput = false;

    ArrayList<GameState> boardStates;
    int[] lastMoves;
    Player[] players; // used to update players of moves played
    Game game;
    int history; // size of states

    int mouseX;
    int mouseY;
    
    static final int windowWidth = 800;
    static final int windowHeight = 800;

    int[] boardScreen;

    boolean saveGames = false;
    String filename = "Game-records.txt";
    BufferedWriter writer = null;
    int goodPlayer = -1; // either index base 0 player or -1 for all

    JFrame jFrame = null;
    JPanel jPanel = null;

    boolean verbose = false;

    public GameManager(Game game, Player[] players, boolean verbose, boolean defaultDraw){
        if(saveGames){
            try{
                File file = new File(filename);
                file.createNewFile();
                writer = new BufferedWriter(new FileWriter(file));
            }catch(Exception e){}
        }
        this.players = players;
        this.verbose = verbose;
        this.game = game;
        
        resetGame();

        int p = boardStates.get(0).getPlayerMove() - 1;

        // If there's a human player
        if(defaultDraw || (gameStatus == 0 && (p >= 0 && players[p] instanceof HumanPlayer))){
            boardScreen = new int[]{50, 50, windowWidth - 100, windowHeight - 100};

            for(int j = 0; j < players.length; j++){
                if(players[j] instanceof HumanPlayer){
                    ((HumanPlayer)players[j]).screen = boardScreen;
                }
            }

            jPanel = new JPanel(){
                @Override
                public void paintComponent(Graphics g){

                    g.clearRect(0, 0, windowWidth, windowHeight);


                    game.drawBoard(g, boardStates.get(0), boardScreen);
                    
                    int p = boardStates.get(0).getPlayerMove() - 1;
                    if(p >= 0 &&  players[p] instanceof HumanPlayer){
                        game.drawMouseState(g, (HumanPlayer)(players[p]), boardStates.get(0), boardScreen);
                    }
                    
                    g.setColor(Color.BLACK);
                    g.drawOval(mouseX - 2, mouseY - 2, 4, 4);
                    g.drawString("Player " + boardStates.get(0).getPlayerMove() + "'s turn", 50, 30);
                }
            };
            jFrame = new JFrame("Game Manager: " + game.getClass());
            jFrame.add(jPanel);
            jFrame.addMouseListener(new MouseListener(){

                public void mouseEntered(MouseEvent e){}

                public void mouseExited(MouseEvent e){}

                public void mouseClicked(MouseEvent e){}

                public void mousePressed(MouseEvent e){}

                public void mouseReleased(MouseEvent e){
                    int p = boardStates.get(0).getPlayerMove() - 1;
                    if(players[p] instanceof HumanPlayer){
                        if(jPanel != null){
                            jPanel.repaint();
                        }
                        gameStep();
                        if(jPanel != null){
                            jPanel.repaint();
                        }
                    }
                }
            });
            jFrame.addMouseMotionListener(new MouseMotionListener(){
                

                public void mouseMoved(MouseEvent e){
                    int pt = boardStates.get(0).getPlayerMove() - 1;
                    if(pt >= 0 && players[pt] instanceof HumanPlayer){
                        HumanPlayer human = (HumanPlayer)players[pt];
                        human.mouse.x = e.getX();
                        human.mouse.y = e.getY();            
                    }
                    mouseX = e.getX();
                    mouseY = e.getY();     
                    if(jPanel != null){
                        jPanel.repaint();
                    }
                }

                public void mouseDragged(MouseEvent e){}
            });
            jFrame.setSize(windowWidth, windowHeight);
            jFrame.setVisible(true);
            // jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }

        // Start the game
        if(p < 0 || !(players[p] instanceof HumanPlayer)){
            gameStep();
        }

        // new Thread(){
        //     public void run() {
                
        //     }
        // }.start();

        while(true){
            synchronized(this){
                if(gameFinished){
                    if(saveGames){
                        System.out.println("Writing files...");
                        try{
                            writer.flush();
                            writer.close();
                        }catch(Exception er){}
                    }
                    if(jFrame != null){
                        jFrame.dispatchEvent(new WindowEvent(jFrame, WindowEvent.WINDOW_CLOSING));
                    }
                    return;
                }
            }
        }

    }

    public int getState(){
        return gameStatus;
    }

    public void resetGame(){
        for(Player player: players){
            player.restart();
        }
        history = 20;
        boardStates = new ArrayList<GameState>(history);
        lastMoves = new int[history];
        for(int j = 0; j < history; j++){
            boardStates.add(game.newGame());
            lastMoves[j] = -2;
        }
        lastMoves[0] = 0;
        gameStatus = 0;
        waitingForInput = false;
    }

    public void undo(){
        reverseState();
        int p = boardStates.get(0).getPlayerMove() - 1;



        // String className = players[p].getClass().getName();

        // System.out.println("Class " + className);

//        if not a humans turn
        while(!(players[p] instanceof HumanPlayer)){
            reverseState();
        }

    }

    public void reverseState(){
        for(int i = 0; i < boardStates.size() - 1 && lastMoves[i + 1] != -1; i++) {
            GameState currentBoard = boardStates.get(i);
            GameState nextBoard = boardStates.get(i + 1);
            currentBoard = nextBoard;
            lastMoves[i] = lastMoves[i + 1];
            gameStatus = game.getState(boardStates.get(0));
        }
    }


    public void gameStep(){
        int p;
        do{
            stepPlayer();
            p = boardStates.get(0).getPlayerMove() - 1;
            if(jPanel != null){
                jPanel.repaint();
            }
        }while(gameStatus == 0 && (p < 0 || !(players[p] instanceof HumanPlayer)));

        if(gameStatus != 0){
            synchronized(this){
                gameFinished = true;
            }
        }
    }

    public void stepPlayer(){ // requests next move, updates players



        waitingForInput = false;

        if(gameStatus != 0){
            return;
        }

        GameState originalBoard = boardStates.get(0);
        // printList(game.legalMoves(originalBoard));



        int p = originalBoard.getPlayerMove() - 1;

        // String className = players[p].getClass().getName();
//        if human player
        // if(players[p] instanceof HumanPlayer){
        //     HumanPlayer player = (HumanPlayer)players[p];
        //     if(player.mouse.x == -1){
        //         waitingForInput = true;
        //         return;
        //     }
        // }
        if(verbose){
            System.out.println("\nPlayer " + (p + 1));
            System.out.println("Legal moves: " + game.legalMoves(originalBoard).size());
        }

        int pmove;

        if(p == -2){ // move by nature
            pmove = game.getMoveByNature(originalBoard);
        }else{
            pmove = players[p].getMove(originalBoard);

            if(players[p] instanceof HumanPlayer){
                if(!game.legalMoves(originalBoard).contains(pmove)){
                    System.out.println("Illegal move by human, try again");
                    // System.out.println("Illegal move by human, try again");
                    //illegal move by human
                    waitingForInput = true;
                    return;
                }
            }
        }

        if(verbose){
            System.out.println("Move: " + pmove);
        }
        

        // for(int i = boardStates.size() - 1; i >= 1; i--) {
        //     GameState currentBoard = boardStates.get(i);
        //     GameState lastBoard = boardStates.get(i - 1);
        //     currentBoard = lastBoard;
        //     lastMoves[i] = lastMoves[i - 1];
        // }

        lastMoves[0] = pmove;

        if (pmove == -1) { // resignation
            // gameStatus = 2-p;
            gameStatus = -1;
            return;
        }

        // before simulation write game in format
        // game state
        // move
        // game state
        // move
        if(saveGames){
            if(goodPlayer == -1 || p == goodPlayer){
                try{
                    writer.write(originalBoard.toString());
                    writer.newLine();
                    writer.write(""+pmove);
                    writer.newLine();
                }catch(Exception e){}
            }
        }

        gameStatus = game.simMove(pmove, originalBoard);
        for(Player player: players){
            player.update(pmove);
        }

        // printList(originalBoard);

        if(verbose){
            System.out.println("New Board: " + originalBoard.toString());
        }


        if (gameStatus == -1) { // resignation
            // gameStatus = 2-p;
            gameStatus = -1;
            return;
        }

    }

    public static void printLegalMoves(ArrayList<Integer> legalMoves){
        System.out.print("Legal Moves: ");
        for(Integer b: legalMoves){
            System.out.print(b + ", ");
        }
        System.out.print("\n");
    }

    


    // public Color getColor(int boardSpace){
    //     switch((int)boardSpace){
    //         case 1: return Color.RED;
    //         case 2: return Color.BLUE;
    //         case 3: return Color.GREEN;
    //     }
    //     return Color.BLACK;
    // }

    

    // public static void printBoard(ArrayList<Byte> board){
    //     System.out.println("Player to move: " + board.get(0));
    //     for(int i = 0; i < 3; i++){
    //         for(int j = 0; j < 27; j++){
    //             System.out.print(board.get(1 + j + i*27) + " ");
    //         }
    //         System.out.println("");
    //     }
    //     for(int j = 0; j < 9; j++){
    //         System.out.print(board.get(1 + 3*27 + j) + " ");
    //     }
    //     System.out.println("");
    // }

    


    

}
