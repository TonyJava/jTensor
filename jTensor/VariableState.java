package jTensor;

import java.util.*;
import java.io.*;

public class VariableState{
	HashMap<String, Tensor> map;

	public VariableState(){
		map = new HashMap<String, Tensor>();
	}

	public void addVariable(VariableNode variableNode){
		map.put(variableNode.name, new Tensor(variableNode.getTensor()));
	}

	public void writeToFile(String filename){
	   try {

			FileOutputStream fos = new FileOutputStream(filename);
	        ObjectOutputStream oos = new ObjectOutputStream(fos);
		
	        oos.writeObject((Integer)(map.size()));
	        // System.out.println("Map Size: " + map.size());

		    Iterator it = map.entrySet().iterator();
		    while (it.hasNext()) {
		        Map.Entry pair = (Map.Entry)it.next();
		        oos.writeObject(pair.getKey());
		        Tensor value = (Tensor)(pair.getValue());
		        oos.writeObject(value.getDimensions());
		        oos.writeObject(value.getObject());
		    }

	        // FileInputStream fis = new FileInputStream("test.dat");
	        // ObjectInputStream iis = new ObjectInputStream(fis);
	        // newTwoD = (int[][]) iis.readObject();

	    } catch (Exception e) {
		System.out.println("Failed to write file: " + filename);
	    }
	}

	public static VariableState readFromFile(String filename){

		VariableState vs = new VariableState();
		try {

			FileInputStream fis = new FileInputStream(filename);
	        ObjectInputStream iis = new ObjectInputStream(fis);
		
	        int size = (Integer) iis.readObject();
	        // System.out.println("Map Size: " + size);

		    for(int j = 0; j < size; j++) {
		        String name = (String) iis.readObject();
		        int[] dimensions = (int[]) iis.readObject();
		        Object object = null;
		        switch(dimensions.length){
		        	case 0: object = (Double) iis.readObject();break;
		        	case 1: object = (double[]) iis.readObject();break;
		        	case 2: object = (double[][]) iis.readObject();break;
		        	case 3: object = (double[][][]) iis.readObject();break;
		        	case 4: object = (double[][][][]) iis.readObject();break;
		        }
		        Tensor tensor = new Tensor(object, dimensions);
		        vs.map.put(name, tensor);
		    }

	    } catch (Exception e) {
		System.out.println("Failed to load file: " + filename);
	    }
	    return vs;
	}
}
