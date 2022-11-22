
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>

#include <stdio.h>


#include "NeuralNetwork.h"
#include "Activation.h"
#include "Error.h"


#include <iomanip>
#include <windows.h>

/*The MIT License (MIT)
Copyright © 2022 Alexander Joseph Kostyak

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.*/

//Development credit to Ryan Wise GitHub user rdw88 for substantial codebase:
//https://github.com/rdw88/CUDA-Neural-Network

int main(int argc, char** argv)
{
	for (int i = 0; i < argc; ++i)
		std::cout << argv[i] << "\n";

	const unsigned int inputlayersize = 10;
	const unsigned int outputlayersize = 12;
	const unsigned int batchsizearg = 1;
	const float learningrate = 0.1;


	std::vector<int> layers;


	
	//activation parameters, as types and bests with exe argument and output validation;
	//function to csv of sorts

	NeuralNetwork network = NeuralNetwork({ inputlayersize, 50, 50, outputlayersize }, batchsizearg, learningrate);

	Activation act1 = newActivation(RELU);
	Activation act2 = newActivation(RELU);
	Activation act3 = newActivation(RELU);
	act1.maxThreshold = 5;
	act2.maxThreshold = 10;
	act3.maxThreshold = 15;
	act1.leakyReluGradient = 0.01;
	act2.leakyReluGradient = 0.01;
	act3.leakyReluGradient = 0.01;


	//generate executable that accepts necessary parameters (net name or filename / constructor filename, operations, etc...)

	network.setLayerActivations({ act1,		act2,		act3,		newActivation(SIGMOID) });

	network.setLossFunction(MEAN_SQUARED_ERROR);

	//all input/output as float from 0-1

	std::vector<float> input;
	//std::vector<float> output;




	std::vector<float> single_input { 0.15, 0.45, .78, 0.04, 0.45, 0.73, 1, 0.11, 0.01, 0.11 };
	std::vector<float> single_input_2 {0.38, 0.92, 0.16, 0.63, 0.82, 0.11, .2, 0.73, 0.25, 0.68};
	
	std::vector<float> input_values{ 0.15, 0.45, .78, 0.04, 0.45, 0.73, 1, 0.11, 0.01, 0.11 };
	std::vector<float> input_values_2{ 0.38, 0.92, 0.16, 0.63, 0.82, 0.11,.2, 0.73, 0.25, 0.68 };

	std::vector<float> output_values { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 1.0, 0.9, 1.0 };            
	std::vector<float> output_values_2 { 0.3, 0.9, 0.5, 0.0, 0.7, 0.4, 0.4, 0.1, 0.2, 0.3, 0.8, 0.3 };           


	std::vector<float> output_valuest{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,.9, 1.0 , 0.9, 1.0 };            
	std::vector<float> output_values_2t{ 0.3, 0.9, 0.5, 0.0, 0.7, 0.4, 0.4, 0.1,0.2, 0.3, 0.8, 0.3 };           


	
	//create input batch
	for (int i =0; i< batchsizearg * inputlayersize - inputlayersize; i++)
	{
		input_values.push_back(single_input[i % inputlayersize]);
		input_values_2.push_back(single_input_2[i % inputlayersize]);
	}

	//create output batch
	for (int i = 0; i < batchsizearg * outputlayersize - outputlayersize; i++)
	{
		output_values.push_back(output_valuest[i % outputlayersize]);
		output_values_2.push_back(output_values_2t[i % outputlayersize]);
	}

	std::cout << std::fixed;
	std::cout << std::setprecision(6);

	long int before = GetTickCount();


	std::cout << std::endl << std::endl << "batch size:   " << network.getBatchSize() << std::endl << "input layer size:   " << "should be 10" << std::endl << std::endl;
	for (int i = 0; i < 500; i++)
	{
		network.train(input_values, output_values);
		network.train(input_values_2, output_values_2);
		std::cout << " " << network.getTotalError()[0];
	}

	long int after = GetTickCount();

	std::cout << std::endl << "time elapsed ms:   " << after - before << std::endl;


	std::cout << std::endl;
			

	std::vector<float> output = network.getOutputForInput(single_input);
	//get_output(single_input)

	for (int i = 0; i < outputlayersize; i++) // (int i in output)
	{
		std::cout  << single_input[i%10] << "   " << output_values[i] << "   " << output[i] << std::endl ;
	}
			//for i, item in enumerate(output) :
				//print('%.1f' % item, output_values[i])

	std::cout << "---------------------------------" << std::endl;



	output = network.getOutputForInput(single_input_2);
	for (int i = 0; i < outputlayersize; i++) // (int i in output)
	{
		std::cout << single_input_2[i%10] << "   " << output_values_2[i] << "   " << output[i] << std::endl;
	}

	std::cin.get();



    return 0;
}