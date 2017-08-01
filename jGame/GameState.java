package jGame;

public abstract class GameState{
	public abstract int getPlayerMove();
    public abstract void resetState();
	public abstract GameState createCopy();
	public double[][][] getVolumeRepresentation(){
		return null;
	}
	public String toString(){
		return "";
	}
}