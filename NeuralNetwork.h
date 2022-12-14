/**
 * NeuralNetwork.h
 * April 4, 2019
 * Ryan Wise
 * 
 * An implementation of a neural network that runs on a CUDA-enabled NVIDIA GPU.
 * 
 */


#ifndef ANN_H__
#define ANN_H__


#include <string>
#include <vector>

#include "GPU.h"


class NeuralNetwork {
	private:

		/**
			Holds matrices representing the weights of synapses that are inputs to a given layer in the network. Each index in the vector
			holds matrices for a given layer and the first element in the vector will always be null (the input layer has no input synapses).
			The count of matrices for a given layer will be equal to the network's batch size, however each matrix in a layer will be
			equal to one another. This is used to parallelize the training examples in a given batch.

			The pointers in this vector reference memory in the GPU and therefore cannot be directly dereferenced.
		*/
		std::vector<float **> m_SynapseMatrices;

		/**
			Holds vectors representing the biases of each neuron for a given layer. Each index in the vector holds a bias vector for a
			given layer. The contents are not batched, meaning there is only one copy of each bias vector regardless of batch size.
			The first element in the vector will always be null since the input layer does not have bias.

			The pointers in this vector reference memory in the GPU and therefore cannot be directly dereferenced.
		*/
		std::vector<float *> m_BiasVectors;

		/**
			Holds vectors representing the error of each neuron calculated on each training run. Each index in the vector represents
			a particular layer's error on a training example. The first element in the vector will always be null since the input
			layer does not have error.

			The pointers in this vector reference memory in the GPU and therefore cannot be directly dereferenced.
		*/
		std::vector<float *> m_ErrorVectors;

		/**
			Holds vectors representing the current values of each neuron in the network for a given input. Each index in the vector
			holds an array of vectors that represents the current values of a layer. Each vector for a given layer represents the value
			vector for the given layer and training example within the batch.

			The pointers in this vector reference memory in the GPU and therefore cannot be directly dereferenced.
		*/
		std::vector<float **> m_ValueVectors;

		/**
			References the same data as in *m_ValueVectors* but instead holds CPU memory pointers. This is used for performance reasons
			during training to allow us to reference specific vectors that reside in GPU memory without having to copy the pointers in 
			*m_ValueVectors* from the GPU to memory every time we need to do a lookup.
		*/
		std::vector<float **> m_CpuValueVectors;

		/**
			References the same data as in *m_SynapseMatrices* but instead holds CPU memory pointers. This is used for performance reasons
			during training to allow us to reference specific matrices that reside in GPU memory without having to copy the pointers in 
			*m_SynapseMatrices* from the GPU to CPU memory every time we need to do a lookup.
		*/
		std::vector<float **> m_CpuSynapseMatrices;

		/**
			A pointer to GPU memory containing a vector that holds the expected output for each training example in a batch. This
			vector is a consolidated vector of each training example in a batch.
		*/
		float *m_ExpectedOutput;

		/**
		 * The activation function to use for each layer
		 */
		std::vector<Activation *> m_ActivationFunctions;

		/**
			A vector containing the number of neurons in each layer.
		*/
		std::vector<unsigned int> m_LayerSizes;

		/**
			The number of training examples the network processes at a time. Each training example in a batch is calculated in parallel.
		*/
		unsigned int m_BatchSize;

		/**
			The learning rate for the network.
		*/
		float m_LearningRate;

		/**
		 * Specifies whether to calculate the error for the input layer of the network during backpropogation. This is useful if
		 * the error of this network will continued to be backpropogated to another neural network. However, turning this on requires
		 * another iteration of backpropogation which impacts performance so it is best to have disabled if not needed.
		 * 
		 * Default value is false.
		 */
		bool m_CalcInputLayerError;

		/**
		 * The loss function in use when calculating the total error of the network. Useful for tracking the performance of the network
		 * as it's training but causes a degradation in training speed. A value of NO_LOSS will stop the calculation of the network's
		 * total error during training.
		 * 
		 * Default value is NO_LOSS.
		 */
		LossFunction m_LossFunction;

		/**
		 * A vector containing the total error of the network on the last training run. Each entry in the vector represents the
		 * total error of the network for the n'th batch in the last training run. This vector will only be populated if a loss
		 * function has been set for the network (i.e. not NO_LOSS).
		 */
		std::vector<float> m_TotalError;
		

	public:
		/**
			Do not use.

			The default constructor for the neural network.
		*/
		NeuralNetwork();

		/**
			Constructs a new neural network. Each layer uses ReLU as the default activation function.

			@param neuronsPerLayer The number of neurons per layer.
			@param batchSize The batch size to be used in training.
			@param learningRate The learning rate of the network.
		*/
		NeuralNetwork::NeuralNetwork(std::vector<unsigned int> neuronsPerLayer, unsigned int batchSize, float learningRate);

		/**
			Deallocates all resources created by the network, including memory allocated in GPU memory.
		*/
		~NeuralNetwork();

		/**
			Train the neural network. The network makes predictions based on the training examples provided as input
			and corrects itself through backpropogation based on the expected outputs.

			@param batch A vector containing the input batch. The size of this vector should be the batch size of the 
						 network multiplied by the number of input neurons.
			@param expectedOutput The expected output values for the input. The size of this vector should be the batch
								  size of the network multiplied by the number of output neurons.

		*/
		void train(std::vector<float> batch, std::vector<float> expectedOutput);

		/**
			Load the input vector into the network.

			@param input A vector containing input values for the network.
		*/
		void loadInput(std::vector<float> input);

		/**
			Load the expected output vector.

			@param expectedOutput A vector containing the expected output of the network.
		*/
		void loadExpectedOutput(std::vector<float> expectedOutput);

		/**
			Get the output of the network given an input. The input can be batched at any size n for 1 <= n <= batchSize.

			@param input The input values to be assigned to the input neurons.
			@return A vector containing the values of the output neurons derived from the supplied input.
		*/
		std::vector<float> getOutputForInput(std::vector<float> input);
		
		/**
			Perform the feed forward step of training. This step takes the current input set on the network's input neurons
			and updates the values of the hidden and output layers of the network with their corresponding values according
			to the input, weights, and biases.

		*/
		void feedForward();

		/**
			Calculate the error of the network based on the loaded input and expected outputs. If a loss function is set using *setLossFunction()*,
			also calculate the total error of the network.
		*/
		void calculateError();

		/**
			Based on the calculated error of the output layer, backpropogate the error of the network to its hidden layers.
		*/
		void backpropogate();

		/**
			Update the weights and biases of the network based on the error calculated from backpropogation.
		*/
		void applyWeights();

		/**
			A wrapper method that calls *calculateError()*, *backpropogate()*, and *applyWeights()*
			If outputError is provided, the error vector of the output layer is set to that vector instead of calling *calculateError()*

			@param outputError The error vector of the output layer. If NULL, *calculateError()* is run to calculate
							   the error of the output layer instead.
		*/
		void updateNetwork(std::vector<float> outputError);

		/**
			The output values of the network. This is different from *getOutputLayer()* by returning by value instead of by reference.
			*getOutputLayer()* also returns GPU pointers that cannot be dereferenced until copied back to CPU memory.

			@return A vector containing the current values of the output neurons.
		*/
		std::vector<float> getCurrentOutput();

		/**
		 * The current values in the error vector for a given layer that is calculated during backpropagation.
		 * 
		 * @param layer The layer to get the error vector for.
		 * @return A vector containing the error of each neuron in the layer from the last run of backpropagation.
		 */
		std::vector<float> getErrorVectorForLayer(unsigned int layer);

		/**
			Save the current state of the neural network to a file. Saves network size, learning rate, batch size, 
			weights, and biases. To load a network saved with this method, use *networkFromFile(std::string filename)*.

			@param filename The path to save the network to on disk.
		*/
		void save(std::string filename);

		/**
			Sets the synapse matrix for a given layer.

			@param layer The layer index. Index cannot be 0 (the input layer).
			@param matrix The synapse matrix. Vector length must be the size of @layer multiplied by size of (@layer - 1).
		*/
		void setSynapseMatrix(unsigned int layer, std::vector<float> matrix);

		/**
			Sets the bias vector for a given layer.

			@param layer The layer index. Index cannot be 0 (the input layer).
			@param vector The bias vector. Vector length must be equal to size of @layer.
		*/
		void setBiasVector(unsigned int layer, std::vector<float> vector);

		/**
		 * Sets the learning rate of the network.
		 * 
		 * @param learningRate The new learning rate to set for the network.
		 */
		void setLearningRate(float learningRate);

		/**
		 * Sets the activation function to use for each layer in the network.
		 * 
		 * @param activations A vector of Activations representing the activation function to use.
		 */
		void setLayerActivations(std::vector<Activation> activations);

		/**
		 * Sets whether or not to calculate the error of the input layer
		 * 
		 * @param calculate Set to true if error calculation of the input layer is required.
		 */
		void setCalcInputLayerError(bool calculate);

		/**
		 * Sets the loss function to use when calculating the total error of the network after each training run. A value of
		 * NO_LOSS will stop the loss calculation during training.
		 * 
		 * @param lossFunction The loss function to use when calculating the total error.
		 */
		void setLossFunction(LossFunction lossFunction);

		/**
			The output values of the network.

			@return An array of vectors containing the output values of the network for each training example provided as input.
		*/
		float **getOutputLayer();

		/**
			The bias vectors for the network.

			@return A vector of bias vectors for each layer. The first vector will always be null (the input layer).
		*/
		std::vector<float *> getBiasVectors();

		/**
			The error vectors for the network.

			@return A vector of error vectors for each layer. The first vector will always be null (the input layer).
		*/
		std::vector<float *> getErrorVectors();

		/**
			The synapse matrices for the network.

			@return A vector of synapse matrices for each layer. The first matrix will always be null (the input layer).
		*/
		std::vector<float **> getSynapseMatrices();

		/**
		 * The synapse matrices for the network. The first pointer is stored on the CPU and can be dereferenced directly to get the GPU
		 * memory address of a particular batch.
		 * 
		 * @return A vector of synapse matrices for each layer. Each pointer in the vector is a pointer to CPU memory space and can be
		 * dereferenced to find the memory address of a batch in the GPU.
		 */
		std::vector<float **> getCPUSynapseMatrices();

		/**
			The value vectors for the network.

			@return A vector of vectors with the current values of the neurons in the network.
		*/
		std::vector<float **> getValueVectors();

		/**
		 * The values in the bias vector for a layer in the network.
		 * 
		 * @param layer The layer index in the network.
		 * @return A vector of values representing the bias for each neuron in the layer. The first bias vector will always be
		 * null (the input layer).
		 */
		std::vector<float> getBiasVectorValues(unsigned int layer);

		/**
		 * The values in the synapse matrix for a layer in the network.
		 * 
		 * @param The layer index in the network.
		 * @return A vector of values representing the weights for each synapse connecting @layer with @layer - 1. The first
		 * synapse matrix will always be null (the input layer).
		 */
		std::vector<float> getSynapseMatrixValues(unsigned int layer);

		/**
		 * The total error of the last training run for each batch. A loss function must be set for this vector to be populated
		 * using *setLossFunction()*.
		 * 
		 * @return A vector containing the total error of the network for each batch in the last training run.
		 */
		std::vector<float> getTotalError();

		/**
			The layer sizes of the network.

			@return A vector containing the sizes of each layer in the network.
		*/
		std::vector<unsigned int> getLayerSizes();

		/**
		 * The activation functions used for each layer in the network.
		 * 
		 * @return A vector containing the activation functions used for each layer in the network.
		 */
		std::vector<Activation> getLayerActivations();

		/**
			The expected output loaded in the network.

			@return A pointer to GPU memory that contains the current expected output vector.
		*/
		float *getExpectedOutput();

		/**
			The learning rate of the network.

			@return The learning rate.
		*/
		float getLearningRate();

		/**
			The batch size the network uses to train.

			@return The batch size of the network.
		*/
		unsigned int getBatchSize();

		/**
			The input size of the network.

			@return The number of neurons in the input layer.
		*/
		unsigned int getInputSize();

		/**
			The output size of the network.

			@return The number of neurons in the output layer.
		*/
		unsigned int getOutputSize();

		/**
			The number of layers in the network, including the input and output layers.

			@return The number of layers in the network.
		*/
		unsigned int getLayerCount();
};


/**
	Load a neural network from the file specified in @filename. The format expected is the same format used
	to save a neural network in the *save()* method.

	@param filename A path to a saved neural network.
	@return The loaded neural network.
*/
NeuralNetwork *networkFromFile(std::string filename);


#endif