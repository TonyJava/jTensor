javac RunModel.java

java RunModel --model DNN --dataset MNIST --params layerSize:a={512,256}
java RunModel --model DNN --dataset GO9 --params layerSize:a={512,256}
java RunModel --model DNN --dataset MNIST
java RunModel --model DNN --dataset GO9

java RunModel --model CONV --dataset MNIST --params filters:a={32,64},filterWidth:a={5,5},fcSize:i=1024,pooling:a={2,2}
java RunModel --model CONV --dataset GO9 --params filters:a={48, 64},fcSize:i=1024
java RunModel --model CONV --dataset MNIST
java RunModel --model CONV --dataset GO9