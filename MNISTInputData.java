import jTensor.*;
import java.io.*;
import java.util.ArrayList;

public class MNISTInputData extends InputData{

	public MNISTInputData(boolean flat, boolean oneHot){
		String[] trainFiles = {"./train-images.idx3-ubyte", "./train-labels.idx1-ubyte"};
		String[] testFiles = {"./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte"};

		System.out.println("Loading MNIST training data...");

		Tensor[] trainImages = loadImages(trainFiles[0], flat, 40000);
		Tensor[] trainLabels = loadLabels(trainFiles[1], oneHot, 40000);

		trainData = new InputData.TrainingData[trainImages.length];

		for(int j = 0; j < trainData.length; j++){
			trainData[j] = new InputData.TrainingData();
			trainData[j].input = trainImages[j]; 
			trainData[j].output = trainLabels[j]; 
		}

		System.out.println("Loading MNIST test data...");

		Tensor[] testImages = loadImages(testFiles[0], flat, 5000);
		Tensor[] testLabels = loadLabels(testFiles[1], oneHot, 5000);

		testData = new InputData.TrainingData[testImages.length];

		for(int j = 0; j < testData.length; j++){
			testData[j] = new InputData.TrainingData();
			testData[j].input = testImages[j]; 
			testData[j].output = testLabels[j]; 
		}

		System.out.println("Loaded MNIST! Train:Test (" + trainData.length + ":" + testData.length + ")");
	}

	public int getOutputClasses(){
		return 10;
	}

	public Tensor[] loadLabels(String file, boolean oneHot, int labelCount){
		BufferedInputStream br = null;
		ArrayList<Integer> labels = new ArrayList<Integer>();

		try {
			br = new BufferedInputStream(new FileInputStream(file));

			byte[] buffer = new byte[4];

			buffer[0] = (byte)br.read(); // magic number
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();
			int intRead = 0;

			buffer[0] = (byte)br.read(); // number of labels
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();
			for(int j = 0; j < 4; j++){
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}

			for(int j = 0; j < labelCount; j++){
				int b = br.read();
				if(b != -1){
					labels.add((int)(b));
				}else{
					System.out.println("Error2!" + b);
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("Error20: " + e);
		}

		Tensor[] data = new Tensor[labels.size()];
		
		if(!oneHot){
			int[] oneHotDimensions = {};

			for(int j = 0; j < data.length; j++){
				Double tensorObject = new Double(labels.get(j));
				data[j] = new Tensor(tensorObject, oneHotDimensions);
			}
		}else{
			int[] sparseDimensions = {10};

			for(int j = 0; j < data.length; j++){
				double[] tensorObject = new double[10];
				tensorObject[labels.get(j)] = 1;
				data[j] = new Tensor(tensorObject, sparseDimensions);
			}
		}

		return data;
	}

	public Tensor[] loadImages(String file, boolean flat, int labelCount){
		BufferedInputStream br = null;
		ArrayList images = new ArrayList();


		try {
			br = new BufferedInputStream(new FileInputStream(file));

			byte[] buffer = new byte[4];

			buffer[0] = (byte)br.read(); // magic number
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();		

			buffer[0] = (byte)br.read(); // number of images
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();		
			int intRead = 0;
			for(int j = 0; j < 4; j++){
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}

			buffer[0] = (byte)br.read(); // rows
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();				
			for(int j = 0; j < 4; j++){
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}
			int rows = intRead;

			buffer[0] = (byte)br.read(); // cols
			buffer[1] = (byte)br.read();
			buffer[2] = (byte)br.read();
			buffer[3] = (byte)br.read();
			for(int j = 0; j < 4; j++){
				int mask = (buffer[j] << ((3 - j)*8));
				mask &= (255) << (3-j)*8;
				intRead |= mask;
			}
			int cols = intRead;

			out_Label:
			for(int j = 0; j < labelCount; j++){
				Object image;
				if(flat){
					image = new double[784];
				}else{
					image = new double[28][28][1];
				}

				for(int x = 0; x < 28; x++){
					for(int y = 0; y < 28; y++){
						int b = br.read();
						if(b != -1){
							if(flat){
								((double[])image)[x*28 + y] = ((double)b)/255.0;
							}else{
								((double[][][])image)[x][y][0] = ((double)b)/255.0;
							}
						}else{
							System.out.println("Error1!");
							break out_Label;
						}
					}
				}
				images.add(image);
			}
		} catch (Exception e) {
			System.out.println("Error10: " + e);
			e.printStackTrace();
		}

		Tensor[] data = new Tensor[images.size()];

		int[] flatDimensions = {784};
		int[] cubeDimensions = {28, 28, 1};
		int[] inputDimensions = flat ? flatDimensions : cubeDimensions;

		for(int j = 0; j < data.length; j++){
			data[j] = new Tensor(images.get(j), inputDimensions);
		}

		return data;
	}


}
