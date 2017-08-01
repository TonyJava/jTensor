package jGame;

import java.util.ArrayList;

 public class Node {

        int move;
        int wins;
        Node parent;
        ArrayList<Node> children;
        int player; // -1 for move by nature
        int sims;
        int finished = 0; //0 = not fininshed, otherwise equals winner
        double boardScore = 0;
        double moveProbability = 0;

        public Node(Node parent, int move) {
            this.parent = parent;
            this.move = move;
            children = new ArrayList<Node>();
            wins = 0;
        }

        // public void deleteChildren(){
        //     if(parent != null){
        //         children.clear();
        //     }
        // }

        public int getSize(){
            int total = 0;
            for(Node child: children){
                total += child.getSize();
            }
            return total + 1;
        }

        public double avergeDepth(){
            if(children.isEmpty()){
                return 1;
            }
            double totalDepth = 0;
            for(Node c: children){
                totalDepth += c.avergeDepth();
            }
            return ((totalDepth/children.size()) + 1);

        }
    }