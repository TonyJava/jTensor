package jGame;

import java.awt.*;
import java.awt.geom.GeneralPath;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class Backgammon extends Game {
    // board[0] is player turn (0 or 1), 1 to 24 pieces in the order black moves in (black bears off from 19 to 24)
    // each piece contains number of black pieces (or negative number of white pieces)

    // SkipLists contain indexes into board of occupied spaces

    public class BackgammonState extends GameState{

        byte[] board;
        SkipListSet<Integer> blackOccupied;
        SkipListSet<Integer> whiteOccupied;
        int dieLow, dieHigh;
        int blackJail, whiteJail;
        boolean repeatTurn; // used for doubles

        public BackgammonState(){
            board = new byte[1 + 24 + 1];
            blackOccupied = new SkipListSet<Integer>();
            whiteOccupied = new SkipListSet<Integer>();
        }

        public void resetState(){
            board[0] = -1;

            board[1] = 2;
            board[6] = -5;
            board[8] = -3;
            board[12] = 5;
            board[13] = -5;
            board[17] = 3;
            board[19] = 5;
            board[24] = -2;

            board[25] = (byte)(random.nextInt(2) + 1); // next player

            blackJail = 0;
            whiteJail = 0;

            blackOccupied.insert(1);
            blackOccupied.insert(12);
            blackOccupied.insert(17);
            blackOccupied.insert(19);

            whiteOccupied.insert(6);
            whiteOccupied.insert(8);
            whiteOccupied.insert(13);
            whiteOccupied.insert(24);

            dieLow = -1;
            dieHigh = -1;

            repeatTurn = false;
        }

        @Override 
        public int getPlayerMove(){
            return board[0];
        }

        @Override
        public GameState createCopy(){
            BackgammonState original = this;
            BackgammonState state = new BackgammonState();
            for(int j = 0; j < state.board.length; j++){
                state.board[j] = original.board[j];
            }
            // int c = 0;
            for(SkipListSet<Integer>.Iterator it = original.blackOccupied.begin(); !it.equals(original.blackOccupied.end()); it.next()){
                // if(it.getKey() == null){
                //     System.out.println("null " + original.blackOccupied.size() + ", " + c);
                //     original.blackOccupied.printMap();
                // }
                //     c++;
                state.blackOccupied.insert(it.getKey());
            }
            for(SkipListSet<Integer>.Iterator it = original.whiteOccupied.begin(); !it.equals(original.whiteOccupied.end()); it.next()){
                state.whiteOccupied.insert(it.getKey());
            }
            state.dieLow = original.dieLow;
            state.dieHigh = original.dieHigh;

            state.blackJail = original.blackJail;
            state.whiteJail = original.whiteJail;

            state.repeatTurn = original.repeatTurn;

            return state;
        }
    }

    Random random;

    public Backgammon() {
        random = new Random();
    }

    // newGame returns an initial game state object
    public GameState newGame(){
        BackgammonState state = new BackgammonState();
        state.resetState();
        return state;
    }

    // die rolls are represented as possibilities all 36. only the 21 unique ones are encoded as legal moves

    // each move moves one piece for one of the dice
    // moves are (1-24 representing what space)*2 (+1 for high die roll)

    // if theres a piece in jail the move is (1-6)*2 (+1 for high die roll)

    // legalMoves returns an arraylist of moves (represented as bytes), if the game is over, the list should be empty
    public ArrayList<Integer> legalMoves(GameState gameState){

        ArrayList<Integer> legalMoves = new ArrayList<Integer>();

        BackgammonState state = (BackgammonState)gameState;

        int player = state.board[0];
        if(player == -1){
            for(int j = 0; j < 6; j++){
                for(int i = j; i < 6; i++){
                    legalMoves.add(j*6 + i);
                }
            }
            return legalMoves;
        }

        SkipListSet<Integer> playerOccupied = player == 1 ? state.blackOccupied : state.whiteOccupied;
        int colorModifier = player == 1 ? 1 : -1;
        int playerJail = player == 1 ? state.blackJail : state.whiteJail;

        if(playerJail > 0){ // player needs to move out of jail
            int roll = state.dieLow;
            for(int dieRoll = 0; dieRoll < 2; dieRoll++){
                if(dieRoll == 1){
                    roll = state.dieHigh;
                    if(roll == -1){
                        break;
                    }
                }
                int startSpace = player == 1 ? 1 : 24;
                if(state.board[startSpace + (colorModifier*(roll-1))] * colorModifier >= -1){
                    legalMoves.add(dieRoll);
                }
            }
        }else{

            int firstToken;
            int bearOff;
            boolean canFinish;
            if(player == 1){
                firstToken = state.blackOccupied.begin().getKey();
                bearOff = 25;
                canFinish = firstToken >= 19;
            }else{
                SkipListSet<Integer>.Iterator it = state.whiteOccupied.end();
                it.previous();
                firstToken = it.getKey();
                bearOff = 0;
                canFinish = firstToken <= 6;
            }

            for(SkipListSet<Integer>.Iterator it = playerOccupied.begin(); !it.equals(playerOccupied.end()); it.next()){
                for(int dieRoll = 0; dieRoll < 2; dieRoll++){
                    int roll = state.dieLow;
                    if(dieRoll == 1){
                        roll = state.dieHigh;
                        if(roll == -1){
                            break;
                        }
                    }
                    int nextSpace = it.getKey() + roll*colorModifier;
                    boolean oob = (nextSpace > 24 || nextSpace < 1);
                    if((canFinish && (firstToken == it.getKey() || (nextSpace == bearOff))) || (!oob && state.board[nextSpace] * colorModifier >= -1)){
                        legalMoves.add((it.getKey()-1)*2 + dieRoll);
                    }
                }
            }
        }

        return legalMoves;
    }

    public int getMoveByNature(GameState board){
        int d1 = (random.nextInt(6));
        int d2 = (random.nextInt(6));
        if(d2 < d1){
            return (d2*6)+d1;
        }
        return (d1*6)+d2;
    }

    // simMove plays move m on game state board, board will be altered to the next game state.
    // returns: 0 if game is not over, returns player who won, -1 if illegal move, -2 for draw
    public int simMove(int m, GameState gameState){
        BackgammonState state = (BackgammonState)gameState;

        int player = state.board[0];

        if(player == -1){
            int d1 = m/6 + 1;
            int d2 = m%6 + 1;
            state.repeatTurn = d1 == d2;
            state.dieLow = d1;
            state.dieHigh = d2;
            state.board[0] = state.board[25];
            return 0;
        }

        SkipListSet<Integer> playerOccupied = player == 1 ? state.blackOccupied : state.whiteOccupied;
        int colorModifier = player == 1 ? 1 : -1;
        int playerJail = player == 1 ? state.blackJail : state.whiteJail;

        if(playerJail > 0){ // player needs to move out of jail
            int roll = m == 0 ? state.dieLow : state.dieHigh;
            if(roll == -1){
                System.out.println("Used die already");
                return -1;
            }
            int startSpace = player == 1 ? 1 : 24;
            int nextSpace = startSpace + (colorModifier*(roll-1));
            int nextRelativeValue = state.board[nextSpace] * colorModifier;
            if(nextRelativeValue == -1){ // capture opponent piece
                state.board[nextSpace] = 0;
                SkipListSet<Integer> opponentOccupied = player == 1 ? state.whiteOccupied : state.blackOccupied;
                opponentOccupied.erase(nextSpace);
                playerOccupied.insert(nextSpace);
                if(player == 1){
                    state.whiteJail++;
                }else{
                    state.blackJail++;
                }
            }else if(nextRelativeValue == 0){
                playerOccupied.insert(nextSpace);
            }else if(nextRelativeValue < -1){
                System.out.println("move: " + m);
                System.out.println("Illegal move, at least two opponent pieces 1");
                return -1;
            }
            if(player == 1){
                state.blackJail--;
            }else{
                state.whiteJail--;
            }
            state.board[nextSpace] += colorModifier;
        }else{
            boolean canFinish = false;
            if(player == 1){
                canFinish = state.blackOccupied.begin().getKey() >= 19;
            }else{
                SkipListSet<Integer>.Iterator it = state.whiteOccupied.end();
                it.previous();
                canFinish = it.getKey() <= 6;
            }

            int roll = m%2 == 0 ? state.dieLow : state.dieHigh;
            if(roll == -1){
                System.out.println("Illegal move, die already used 1");
            }
            int currentSpace = m/2 + 1;
            int nextSpace = currentSpace + roll*colorModifier;
            if(canFinish || ((nextSpace <= 24 && nextSpace >= 1) && state.board[nextSpace] * colorModifier >= -1)){
                if(nextSpace <= 24 && nextSpace >= 1){
                    int nextRelativeValue = state.board[nextSpace] * colorModifier;
                    if(nextRelativeValue == -1){ // capture opponent piece
                        state.board[nextSpace] = 0;
                        SkipListSet<Integer> opponentOccupied = player == 1 ? state.whiteOccupied : state.blackOccupied;
                        opponentOccupied.erase(nextSpace);
                        playerOccupied.insert(nextSpace);    
                        if(player == 1){
                            state.whiteJail++;
                        }else{
                            state.blackJail++;
                        }
                    }else if(nextRelativeValue == 0){
                        playerOccupied.insert(nextSpace);
                    }
                    state.board[nextSpace] += colorModifier;
                }

                state.board[currentSpace] -= colorModifier;
                if(state.board[currentSpace] == 0){
                    playerOccupied.erase(currentSpace);
                }
            }else{
                System.out.println("Illegal move, at least two opponent pieces 2");
                return -1;
            }
        }

        if(state.dieHigh == -1){
            if(state.repeatTurn){
                state.repeatTurn = false;
                state.dieHigh = state.dieLow;
            }else{
                state.board[25] = (byte)((state.board[25] % 2) + 1);
                state.board[0] = (byte)(-1);
            }
        }else{
            if(m%2 == 0){
                state.dieLow = state.dieHigh;
            }
            state.dieHigh = -1;    
        }

        return getState(gameState);
    }

    // returns: 0 if game is not over, returns player who won, -1 if illegal move, -2 for draw
    public int getState(GameState gameState){
        BackgammonState state = (BackgammonState)gameState;
        if(state.blackOccupied.empty() && state.blackJail == 0){
            return 1;
        }
        if(state.whiteOccupied.empty() && state.whiteJail == 0){
            return 2;
        }
        return 0;
    }

    // boardScreen is [x, y, width, height]
    public void drawBoard(Graphics g, GameState gameState, int[] boardScreen){
        BackgammonState state = (BackgammonState)gameState;

        int space = boardScreen[2] / 5;
        int bar = boardScreen[3] / 25;

        g.setColor(new Color(120, 40, 0, 255));
        g.fillRect(boardScreen[0], boardScreen[1], boardScreen[2], boardScreen[3]);
        g.setColor(Color.BLACK);
        g.drawRect(boardScreen[0], boardScreen[1], boardScreen[2], boardScreen[3]);

        for(int j = 0; j < 12; j++){
            int yRectStart = (j*2)*bar + boardScreen[1];
            if(j >= 6){
                yRectStart += bar;
            }
            // g.setColor(Color.BLACK);
            // g.drawRect(boardScreen[0], yRectStart, space*2, bar*2);

            Graphics2D g2 = (Graphics2D) g;

            g2.setStroke(new BasicStroke(1.0f));
            g2.setPaint(j%2 == 0 ? Color.WHITE : Color.RED);
            int[] xPoints = {boardScreen[0], boardScreen[0] + space*2, boardScreen[0]};
            int[] yPoints = {yRectStart + (int)(.25*bar), yRectStart + bar,  yRectStart + (int)(1.75*bar)};
            GeneralPath path = new GeneralPath(GeneralPath.WIND_EVEN_ODD,
                    xPoints.length);
            path.moveTo(xPoints[0], yPoints[0]);
            for (int i = 1; i < xPoints.length; i++) {
                path.lineTo(xPoints[i], yPoints[i]);
            }
            // path.lineTo(xPoints[0], yPoints[0]);
            path.closePath();
            g2.fill(path);
            g2.setPaint(Color.BLACK);
            g2.draw(path);

            g2.setPaint(j%2 == 1 ? Color.WHITE : Color.RED);
            int[] xPoints2 = {boardScreen[0] + space*5, boardScreen[0] + space*3, boardScreen[0] + space*5};
            int[] yPoints2 = {yRectStart + (int)(.25*bar), yRectStart + bar,  yRectStart + (int)(1.75*bar)};
            path = new GeneralPath(GeneralPath.WIND_EVEN_ODD,
                    xPoints2.length);
            path.moveTo(xPoints2[0], yPoints2[0]);
            for (int i = 1; i < xPoints2.length; i++) {
                path.lineTo(xPoints2[i], yPoints2[i]);
            }
            // path.lineTo(xPoints2[0], yPoints2[0]);
            path.closePath();
            g2.fill(path);
            g2.setPaint(Color.BLACK);
            g2.draw(path);


            // g.drawRect(boardScreen[0] + space*3, yRectStart, space*2, bar*2);

            int stoneRadius = (int)(bar*1.5);

            int leftValue = state.board[j+1];
            if(leftValue != 0){
                int count = leftValue < 0 ? leftValue*-1 : leftValue;
                for(int i = 0; i < count; i++){
                    g.setColor((leftValue > 0) ? Color.WHITE : Color.RED);
                    g.fillOval(boardScreen[0] + i*(bar*2), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
                    g.setColor(Color.BLACK);
                    g.drawOval(boardScreen[0] + i*(bar*2), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
                }
            }

            int rightValue = state.board[25-(j+1)];
            if(rightValue != 0){
                int count = rightValue < 0 ? rightValue*-1 : rightValue;
                for(int i = 0; i < count; i++){
                    g.setColor((rightValue > 0) ? Color.WHITE : Color.RED);
                    g.fillOval((boardScreen[0]+boardScreen[2]) - ((i+1)*(bar*2)), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
                    g.setColor(Color.BLACK);
                    g.drawOval((boardScreen[0]+boardScreen[2]) - ((i+1)*(bar*2)), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
                }
            }
        }


        g.setColor(Color.BLACK);
        g.fillRect(boardScreen[0], boardScreen[1] + 12*bar, boardScreen[2], bar);

        g.setColor(Color.WHITE);
        g.drawString(state.dieLow + " : " + state.dieHigh, boardScreen[0] + boardScreen[2]/2, boardScreen[1] + 3*boardScreen[3]/4);
    }

    @Override
  public int getMoveFromMouse(Point mouse, GameState gameState, int[] screen, HumanPlayer player){
        int width = screen[2];

        if(player.moveState == null){
            player.moveState = new BGMS();
        }
        BGMS bgms = (BGMS)player.moveState;

        if(mouse.x < screen[0] + screen[2]/2){
            ArrayList<Integer> legalMoves = legalMoves(gameState);
            int currentIndex = legalMoves.indexOf(bgms.move);
            if(currentIndex == -1){
                bgms.move = legalMoves.get(0);
            }else{
                bgms.move = legalMoves.get((currentIndex + 1)%legalMoves.size());
            }
            return -1;
        }

        return bgms.move;
    }

    class BGMS{
        int move;
    }

 @Override
  public void drawMouseState(Graphics g, HumanPlayer playerInfo, GameState gameState, int[] boardScreen){

    if(playerInfo.moveState == null){
        playerInfo.moveState = new BGMS();
    }
    BGMS bgms = (BGMS)playerInfo.moveState;

    g.setColor(Color.BLACK);
    g.drawString(""+bgms.move, boardScreen[0] + boardScreen[2]/2, boardScreen[1] + boardScreen[3] - 50);

    int space = boardScreen[2] / 5;
    int bar = boardScreen[3] / 25;



    BackgammonState state = (BackgammonState)gameState;

    int player = state.board[0];


    SkipListSet<Integer> playerOccupied = player == 1 ? state.blackOccupied : state.whiteOccupied;
    int colorModifier = player == 1 ? 1 : -1;
    int playerJail = player == 1 ? state.blackJail : state.whiteJail;

    if(playerJail == 0){
        int currentSpace = bgms.move/2 + 1;
        int row = currentSpace - 1;
        if(row >= 12){
            row = 23 - row;
        }
        int yRectStart = (row*2)*bar + boardScreen[1];
        if(row >= 6){
            yRectStart += bar;
        }
       
        int stoneRadius = (int)(bar*1.5);

        if(currentSpace <= 12){
            int leftValue = state.board[currentSpace];
            if(leftValue != 0){
                int count = leftValue < 0 ? leftValue*-1 : leftValue;
                g.setColor(Color.YELLOW);
                g.fillOval(boardScreen[0] + (count - 1)*(bar*2), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
                g.setColor(Color.BLACK);
                g.drawOval(boardScreen[0] + (count - 1)*(bar*2), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
            }
        }else{
            int rightValue = state.board[currentSpace];
            if(rightValue != 0){
                int count = rightValue < 0 ? rightValue*-1 : rightValue;
                g.setColor(Color.YELLOW);
                g.fillOval((boardScreen[0]+boardScreen[2]) - (count*(bar*2)), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
                g.setColor(Color.BLACK);
                g.drawOval((boardScreen[0]+boardScreen[2]) - (count*(bar*2)), yRectStart + (int)(.25*bar), stoneRadius, stoneRadius);
            }
        }
    }


    String die = "" + (bgms.move%2 == 0 ? state.dieLow : state.dieHigh);
    g.drawString(die, boardScreen[0]+boardScreen[2]/2, boardScreen[1]+boardScreen[3]/5);

  }

}
