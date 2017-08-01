package jGame;

import java.awt.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class Trouble extends Game {
//board state: 1 is bottom row left space, increasing going right. 8 is second row, 15 is third row.... 0 is empty space, 1 is p1, 2 is p2

    final int players = 4;
    Random random;

    int move = 1;

    public Trouble() {
        // this.players = players;
        random = new Random();
    }

    public int minimaxValue(ArrayList<Byte> board, int player){
        return 0;
    }

    public ArrayList<Byte> newGame() {
        ArrayList<Byte> board = new ArrayList<Byte>((players * 4) + 3); // secoond last element is next person to move (or current), last element holds last die roll
        board.add((byte)-1); // start with move by nature (die roll)
        for (int j = 0; j < 4 * 4; j++) {
            board.add((byte)-1); // pieces not yet out
        }
        board.add((byte)1);
        board.add((byte)0);

        return board;
    }

    public byte getMoveByNature(ArrayList<Byte> board){
        return (byte)(random.nextInt(6) + 1);
    }

    public ArrayList<Byte> legalMoves(ArrayList<Byte> board) {

        int playerMove = board.get(0);
        ArrayList<Byte> moves = new ArrayList<Byte>();

        if(playerMove == -1){
            for(int j = 1; j <= 6; j++){
                moves.add((byte)j);
            }
        }else{
            int dieRoll = board.get(players*4 + 2);
            for(int j = 0; j < 4; j++){
                int nextSpace;
                int currentSpace = board.get(1 + (playerMove - 1)*4 + j);
                if(currentSpace == -1){
                    if(dieRoll == 6){
                        int startSpace = (playerMove == 1 ? 4 : playerMove - 1) * 7;
                        nextSpace = (startSpace + dieRoll) % 28;
                    }else{
                        continue;
                    }
                }else{
                    nextSpace = (currentSpace + dieRoll) % 28;
                    int startSpace = (playerMove == 1 ? 4 : playerMove - 1) * 7;
                    if(currentSpace >= 28 || (currentSpace < startSpace && (currentSpace + dieRoll) >= startSpace)){
                        int relativeFinishSpace = (currentSpace + dieRoll) - startSpace;
                        if(relativeFinishSpace < 4){
                            nextSpace = 28 + (playerMove - 1) * 4 + relativeFinishSpace;
                        }else{
                            continue;
                        }
                    }
                }
                // if(!occupiedSpaces.contains(nextSpace)){
                //     moves.add((byte)(j+1));
                // }
                // boolean canMove = true;
                // for(int i = 0; i < 16; i++){
                //     if(board.get(i + 1) == nextSpace){ // collision
                //         canMove = i / 4 != (playerMove - 1);
                //         break;
                //     }
                // }
                // if(canMove){
                //     moves.add((byte)(j+1));
                //     continue;
                // }
                int occIndex = board.indexOf((byte)nextSpace);
                if(occIndex != -1){
                    if((occIndex-1)/4 == (playerMove-1)){
                        continue;
                    }
                }
                moves.add((byte)(j+1));

            }
        }

        return moves;
    }

    public int simMove(byte m, ArrayList<Byte> board) { // moves 0-6 corresponding to column left to right, index 0 is player who just moved
        
        int playerMove = board.get(0);

        if(playerMove == -1){
            board.set(players*4 + 2, m);
            byte np = board.get(players*4 + 1);
            board.set(0, np);
            if(legalMoves(board).isEmpty()){
                board.set(0, (byte)(-1));
                board.set(players*4 + 1, (byte)((np%4) + 1));
            }
            return 0;
        }
        int dieRoll = board.get(players*4 + 2);
        int pieceIndex = 1 + (playerMove-1)*4 + (m-1);
        int currentSpace = board.get(pieceIndex);

        int nextSpace;

        if(currentSpace == -1){
            if(dieRoll == 6){
               int startSpace = (playerMove == 1 ? 4 : playerMove - 1) * 7;
                nextSpace = (startSpace + dieRoll) % 28;
            }else{
                return -1;
            }
        }else{
            nextSpace = (currentSpace + dieRoll) % 28;
            int startSpace = (playerMove == 1 ? 4 : playerMove - 1) * 7;
            if(currentSpace >= 28 || (currentSpace < startSpace && (currentSpace + dieRoll) >= startSpace)){
                int relativeFinishSpace = (currentSpace + dieRoll) - startSpace;
                if(relativeFinishSpace < 4){
                    nextSpace = 28 + (playerMove - 1) * 4 + relativeFinishSpace;
                }else{
                    return -1;
                }
            }
        }

        // if(board.get((playerMove-1)*4 + m) == -1){
        //     if(dieRoll)
        //     nextSpace = (startSpace + dieRoll) % 28;
        // }
            
        int occIndex = board.indexOf((byte)nextSpace);
        if(occIndex != -1){
            if((occIndex-1)/4 == (pieceIndex-1)/4){
                return -1;
            }else{
                board.set(occIndex, (byte)-1);
            }
        }
        board.set(pieceIndex, (byte)nextSpace);

        if(board.get(players*4 + 2) != 6){
            board.set(players*4 + 1, (byte)((board.get(players*4 + 1)%players) + 1));
        }
        board.set(0, (byte)-1);

        int state = getState(board);
        // int turn = board.get(0) - 1;
        // board.set(0, (byte)(2-turn));
        return state;
    }

    public int getState(ArrayList<Byte> board) {
        int playersCanWin = 4;
        boolean[] canWin = new boolean[4];
        int[] finishSpace = new int[4];
        for(int j = 0; j < 4; j++){
            for(int i = 0; i < 4; i++){
                if(j == 0){
                    canWin[i] = true;
                    finishSpace[i] = 28 + (i - 1) * 4 + j;
                }
                if(canWin[i] && board.get(1 + (i*4) + j) < finishSpace[i]){
                    if(canWin[i]){
                        canWin[i] = false;
                        playersCanWin--;
                        if(playersCanWin == 0){
                            return 0;
                        }
                    }
                }
            }
        }
        for(int j = 0; j < 4; j++){
            if(canWin[j]){
                return j + 1;
            }
        }
        System.out.println("Shouldn't happen (trouble)");
        return 0;
    }

    public byte getMoveFromMouse(Point mouse, ArrayList<Byte> board, int[] screen, HumanPlayer player){
        int width = screen[2];
        
        if(mouse.x < screen[0] + screen[2]/2){
            move = (move%4) + 1;
            return -1;
        }

        return (byte)move;
    }

    public void drawBoard(Graphics g, ArrayList<Byte> board, int[] screen) {

        int width = screen[2];
        int height = screen[3];

        int dy = height/23;
        int r = dy;
        int dx = width/23;

        int xSpace = (int)(2.2 * r);
        int ySpace = (int)(2.2 * r);

        g.setColor(Color.GRAY);
        g.fillRect(screen[0], screen[1], screen[2], screen[3]);

        // HashSet<Byte> occupiedSpaces = new HashSet<Byte>();

        

        for(int j = 0; j < 28; j++){
            int x;
            int y;
            if((j/7)%2==0){
                if(j/7 > 0){
                    x = 21 - j;
                    y = 7;
                }else{
                    x = j;
                    y = 0;
                }
            }else{
                if(j/7 > 1){
                    x = 0;
                    y = 28 - j;
                }else{
                    x = 7;
                    y = j - 7;
                }
            }

            g.setColor(Color.WHITE);
            g.fillOval(screen[0] + dx + (dx + r + x * xSpace), screen[1] + dy + (dy + r + y * ySpace), r, r);

            g.setColor(Color.BLACK);
            g.drawOval(screen[0] + dx + (dx + r + x * xSpace), screen[1] + dy + (dy + r + y * ySpace), r, r);
        }

        for(int j = 0; j < 16; j++){
            int x;
            int y;
            int location = board.get(j + 1);
            if(location == -1){
                if(j/4 == 0 || j/4== 3){
                    x = j%4 - 1;
                }else{
                    x = 5 + j%4;
                }
                if(j/4 < 2){
                    y = -1;
                }else{
                    y = 8;
                }
            }else{
                if((location/7)%2==0){
                    if(location/7 > 0){
                        x = 21 - location;
                        y = 7;
                    }else{
                        x = location;
                        y = 0;
                    }
                }else{
                    if(location/7 > 1){
                        x = 0;
                        y = 28 - location;
                    }else{
                        x = 7;
                        y = location - 7;
                    }
                }
            }
            switch(j/4){
                case 0: g.setColor(Color.RED);break;
                case 1: g.setColor(Color.GREEN);break;
                case 2: g.setColor(Color.YELLOW);break;
                case 3: g.setColor(Color.BLUE);break;
                default: g.setColor(Color.BLACK);
            }
            g.fillOval(screen[0] + dx + (dx + r + x * xSpace), screen[1] + dy + (dy + r + y * ySpace), r, r);
        }

        g.drawString(""+board.get(18), screen[0] + screen[2]/2, screen[1] + screen[3]/2);

        // for(int j = 1; j <= 16; j++){
        //     occupiedSpaces.add(board.get(j));
        // }

    }

    @Override
  public void drawMouseState(Graphics g, HumanPlayer player, int[] screen){
    g.setColor(Color.BLACK);
    g.drawString(""+move, screen[0] + screen[2] - 50, screen[1] + screen[3] - 50);
  }

}
