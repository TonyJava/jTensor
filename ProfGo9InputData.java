import jTensor.*;
import jGame.*;
import java.io.*;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class ProfGo9InputData extends InputData{

	public ProfGo9InputData(boolean flat, boolean oneHot){
		this(flat, oneHot, -1);
	}

	// inputSamples == -1 for all input samples
	public ProfGo9InputData(boolean flat, boolean oneHot, int inputSamples){

		System.out.println("Loading ProfGo9 training data...");

		int gameFiles = 150;
		InputData.TrainingData[][] dataFiles = new InputData.TrainingData[gameFiles][];
		int totalLength = 0;
		int discardFirst = 0;
		for(int j = 0; j < gameFiles; j++){
			dataFiles[j] = loadMoves("./9x9sgfs/game" + String.format("%03d", j + 1) + ".sgf", flat, oneHot);
			totalLength += dataFiles[j].length - discardFirst;
		}

		int testLength = 1800;
		int trainLength;
		if(inputSamples == -1){
			trainLength = totalLength - testLength;
		}else{
			trainLength = inputSamples;
		}

		trainData = new InputData.TrainingData[trainLength];
		testData = new InputData.TrainingData[testLength];
		int c = 0;
		GAMELOOP:
		for(int j = 0; j < gameFiles; j++){
			for(int i = discardFirst; i < dataFiles[j].length; i++){
				if(c < trainLength){
					trainData[c] = dataFiles[j][i - discardFirst];
				}else{
					testData[c - trainLength] = dataFiles[j][i - discardFirst];
				}
				c++;
				if(c >= trainLength + testLength){
					break GAMELOOP;
				}
			}
		}

		System.out.println("Loaded ProfGo9! Train:Test (" + trainData.length + ":" + testData.length + ")");
	}

	public int getOutputClasses(){
		return 82;
	}

	// aa is top left corner is 1
	// ab is to the right and is 2
	// Returns moves with 8 rotations/symmetries each
	public InputData.TrainingData[] loadMoves(String file, boolean flat, boolean oneHot){
		final int rotations = 8;
		BufferedReader br = null;
		ArrayList games = new ArrayList();
		ArrayList<ArrayList<Integer>> correctMoves = new ArrayList<ArrayList<Integer>>();
		for(int j = 0; j < rotations; j++){
			if(flat){
				games.add(new ArrayList<double[]>());				
			}else{
				games.add(new ArrayList<double[][][][]>());				
			}
			correctMoves.add(new ArrayList<Integer>());
		}

		try {

			br = new BufferedReader(new FileReader(file));

			String line = "";

			while ((line = br.readLine()) != null) {
				if(line.length() < 3 || !line.substring(0, 3).equals(";B[")){
					continue;
				}
				StringTokenizer st = new StringTokenizer(line, ";BW[]");
				while(st.hasMoreTokens()){
					String coordinates = st.nextToken();
					for(int i = 0; i < rotations; i++){
						int x = coordinates.charAt(1) - 'a';
						int y = coordinates.charAt(0) - 'a';
						if((i & 4) != 0){
							int t = x;
							x = y;
							y = t;
						}
						if((i & 1) != 0){
							x = 8 - x;
						}
						if((i & 2) != 0){
							y = 8 - y;
						}
						int move = y*9 + x + 1;
						// Returns a move for Go.java to use (0: pass, 1-81: move)
						correctMoves.get(i).add(move);
					}
				}
			}
		} catch (Exception e) {
			System.out.println("Error: " + e);
		}

		Go goGame = new Go(9);
		goGame.setTrackMoves(true);
		Go.GoGameState[] states = new Go.GoGameState[rotations];
		for(int i = 0; i < states.length; i++){
			states[i] = (Go.GoGameState)goGame.newGame();
		}

		for(int j = 0; j < states.length; j++){
			for(int i = 0; i < correctMoves.get(j).size(); i++){
			
				int correctMove = correctMoves.get(j).get(i);

				double[][][] gameRep = states[j].getVolumeRepresentation();
				if(flat){
					double[] gameUnrolled = new double[gameRep.length * gameRep[0].length * gameRep[0][0].length];
					for(int x = 0; x < gameRep.length; x++){
						for(int y = 0; y < gameRep[0].length; y++){
							for(int z = 0; z < gameRep[0][0].length; z++){
								gameUnrolled[x*gameRep[0].length*gameRep[0][0].length+y*gameRep[0][0].length+z] = gameRep[x][y][z];
							}
						}
					}
					((ArrayList)(games.get(j))).add(gameUnrolled);
				}else{
					((ArrayList)(games.get(j))).add(gameRep);					
				}

				int result = goGame.simMove(correctMove, states[j]);
				if(result != 0){
					System.out.println("Game over while parsing " + result);
				}
			}

		}
	
		int[] flatDimensions = {9*9*12};
		int[] raisedDimensions = {9, 9, 12};
		int[] inputDimensions = flat ? flatDimensions : raisedDimensions;

		int totalExamples = 0;
		for(int j = 0; j < rotations; j++){
			totalExamples += correctMoves.get(j).size();
		}

		InputData.TrainingData[] data = new InputData.TrainingData[totalExamples];
		int examplesSeen = 0;

		for(int i = 0; i < games.size(); i++){
			for(int j = 0; j < ((ArrayList)(games.get(i))).size(); j++){
				data[examplesSeen] = new InputData.TrainingData();
				data[examplesSeen].input = new Tensor(((ArrayList)(games.get(i))).get(j), inputDimensions);
				// data[j*games.size() + i].input.printTensor();
				if(!oneHot){
					Double outputArray = new Double(correctMoves.get(i).get(j));
					int[] labelDimensions = {};
					data[examplesSeen].output = new Tensor(outputArray, labelDimensions);
				}else{
					double[] outputArray = new double[82];
					int[] labelDimensions = {82};
					outputArray[correctMoves.get(i).get(j)] = 1;
					data[examplesSeen].output = new Tensor(outputArray, labelDimensions);
				}
				examplesSeen++;
			}
		}


		return data;
	}


}
