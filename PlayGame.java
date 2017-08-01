import jTensor.*;
import jGame.*;

import java.util.ArrayList;

public class PlayGame{

  int games = 100;

  String filename = "conv_go9_model"; 

  public static void main(String[] args){
    new PlayGame();
  }

  public PlayGame(){
    final Game go = new TicTacToe();
    // ((Go)go).setTrackMoves(true);

    RandomPlayer randomPlayer = new RandomPlayer(go);
    
    MCTSPlayer mctsPlayer = new MCTSPlayer(go, .2, 1);
    
    // final Go9ModelBot goBotPlayer = new Go9ModelBot(go, filename);
    
    // MCTSPlayer mctsHeavyPlayer = new MCTSPlayer(go, 1, 1);
    // final double heavyProb = .1;
    // mctsHeavyPlayer.setHeavyPlayoutCall(new MCTS.HeavyPlayoutCall(){
    //   @Override
    //   public int getHeavyPlayoutMove(GameState gameState){
    //     if(Math.random() < heavyProb){
    //       return goBotPlayer.getMove(gameState);
    //     }else{
    //       ArrayList<Integer> legalMoves = go.legalMoves(gameState);
    //       return legalMoves.get(((int)(Math.random() * legalMoves.size())));
    //     }
    //   }
    // });

    // MCTSPlayer mctsProbPlayer = new MCTSPlayer(go, 1, 1);
    // final double probScale = .1;
    // mctsProbPlayer.setMoveProbabilityCall(new MCTS.MoveProbabilityCall(){
    //   @Override
    //   public double[] getProbabilities(GameState gameState){
    //     double[] rawDist = goBotPlayer.getMoveDistribution(gameState);
    //     return rawDist;
    //   }
    // });
    
    HumanPlayer humanPlayer = new HumanPlayer(go);
    
    int wins = 0;
    int losses = 0;
    int ties = 0;
    Player player = randomPlayer;
    Player opponent = mctsPlayer;
    for(int j = 0; j < games; j++){
      Player[] players = null;
      if(j % 2 == 0){
        Player[] t_players = {opponent, player};
        players = t_players;
      }else{
        Player[] t_players = {player, opponent};
        players = t_players;
      }
      GameManager manager = new GameManager(go, players, false, true);
      int result = manager.getState();
      
      if(result == -2){
        ties += 1;
      }else if(result == -1){
        System.out.println("Illegal move, no result");
      }else if(result == 1 || result == 2){
        if(result == j%2 + 1){
          losses += 1;
        }else{
          wins += 1;
        }
        System.out.println("Winner:: " + (result == j%2 + 1 ? "Opponent" : "Player"));
      }else{
        System.out.println("wut");
      }
    }
    System.out.println(wins + " : " + losses + " : " + ties);
  }

}
