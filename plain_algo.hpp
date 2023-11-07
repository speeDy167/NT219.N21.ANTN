#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int64_t PlainVectorMultiplicationBGV(const vector<int64_t> &sample, const vector<int64_t> &weights)
{
    int64_t product = 0;
    for (size_t i = 0; i < sample.size(); ++i)
    {
        product += sample[i] * weights[i];
    }
    return product;
}

double PlainSigmoidBGV(const vector<int64_t> &sample, const vector<int64_t> &weights)
{
    double product = PlainVectorMultiplicationBGV(sample, weights);
    double sigmoid = 1.0 / (1 + exp(product * (-1)));
    return sigmoid;
}

double PlainVectorMultiplication(const vector<double> &sample, const vector<double> &weights)
{
    double product = 0;
    for (size_t i = 0; i < sample.size(); ++i)
    {
        product += sample[i] * weights[i];
    }
    return product;
}

double PlainSigmoid(const vector<double> &sample, const vector<double> &weights)
{
    double product = PlainVectorMultiplication(sample, weights);
    double sigmoid = 1.0 / (1 + exp(product * (-1)));
    return sigmoid;
}

double ComputeAccuracy(const vector<vector<double>> &features, const vector<double> &labels, const vector<double> &weights)
{
    vector<double> result(features.size());
    size_t correct = 0;
	int true_0=0;
    for (size_t i = 0; i < features.size(); ++i)
    {
        result[i] = PlainSigmoid(features[i], weights);
        if (round(result[i]) == labels[i])
        {
        	if (labels[i] == 0){
        		true_0++;
        	}
            ++correct;
        }
    }
	std::cout <<"correct: "<< correct <<"\n";
	std::cout <<"correct_0: "<< true_0 <<"\n";
	std::cout <<"correct_1: "<< correct - true_0 <<"\n";
	
    return double(correct) / result.size();
}



double ComputeAccuracy(const vector<vector<int64_t>> &features, const vector<int64_t> &labels, const vector<int64_t> &weights)
{
    vector<int64_t> result(features.size());
    size_t correct = 0;
	int true_0=0;
    for (size_t i = 0; i < features.size(); ++i)
    {
        result[i] = PlainSigmoidBGV(features[i], weights);
        // cout << "result: "<<result[i] <<"\n";
        if (round(result[i])*10000 == labels[i])
        {
        	if (labels[i] == 0){
        		true_0++;
        	}
            ++correct;
        }
    }
	std::cout <<"correct: "<< correct <<"\n";
	std::cout <<"correct_0: "<< true_0 <<"\n";
	std::cout <<"correct_1: "<< correct - true_0 <<"\n";
	
    return double(correct) / result.size();
}
