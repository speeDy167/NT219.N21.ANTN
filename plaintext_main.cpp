#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

vector<double> expectedOutput;
vector<vector<double>> inputValues;

double weights[] = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
double learningRate = 0.01;
long epoch = 20;
double e = M_E;
double final_result = 0;

double activation(double z);
void updateWeight(double predictedValue, double expectedOutput, vector<double> inputValue);
void calculateAccuracy();
void test();

vector<vector<double>> ReadDatasetFromCSV(string filename)
{
    fstream fin;
    fin.open(filename, ios::in);

    vector<double> row;
    vector<vector<double>> dataset;
    string line, word;
    double value;

    getline(fin, line);

    while (fin.good())
    {
        row.clear();
        getline(fin, line);
        stringstream ssline(line);

        while (getline(ssline, word, ','))
        {
            stringstream ssword(word);
            ssword >> value;
            row.push_back(value);
        }

        dataset.push_back(row);
    }

    fin.close();
    return dataset;
}

vector<double> ExtractLabel(vector<vector<double>> &dataset, int col_idx)
{
    vector<double> labels;
    for (int i = 0; i < dataset.size() - 1; ++i)
    {
        labels.push_back(dataset[i][col_idx]);
        dataset[i].erase(dataset[i].begin() + col_idx);
        // std::cout << i << "\n\n\n\n";
    }
    // std::cout << "ok\n\n\n";
    return labels;
}

int main()
{
    string line;
    long i, j;

    inputValues = ReadDatasetFromCSV("/home/im5hry/Project_Crypto/dataset/diabetes_normalized.csv");
    expectedOutput = ExtractLabel(inputValues, 8);

    while (epoch--)
    {
        cout << "\n\n#####Epoch " << 19 - epoch << " is running...######\n";
        calculateAccuracy();
        // cout << "\n\n\n/bbb\n\n\n";
        for (i = 0; i < inputValues.size() - 1; i++)
        {
            double predictedValue, z = 0;

            for (j = 0; j < inputValues[0].size(); j++)
            {
                z += weights[j] * inputValues[i][j];
            }

            z += weights[inputValues[0].size()]; // Add bias term
            predictedValue = activation(z);
            updateWeight(predictedValue, expectedOutput[i], inputValues[i]);
        }
    }

    calculateAccuracy();

    cout << "Best accuracy is: " << final_result << "%\n";
    for (int i =0 ;i < 8;i++) {
        cout << weights[i] << ", ";
    }
    cout << weights[8] << "\n";
    return 0;
}

double activation(double z)
{
    return 1 / (1 + pow(e, (-1 * z)));
}

void updateWeight(double predictedValue, double expectedOutput, vector<double> inputValue)
{
    for (int i = 0; i < inputValue.size(); i++)
    {
        double gradientDescent = (predictedValue - expectedOutput) * inputValue[i];
        weights[i] -= learningRate * gradientDescent;
    }

    // Update bias weight
    weights[inputValue.size()] -= learningRate * (predictedValue - expectedOutput);
}

void calculateAccuracy()
{
    long totalCorrect = 0, totalCases = inputValues.size();

    for (int i = 0; i < totalCases - 1; i++)
    {
        double predictedValue, z = 0;

        for (int j = 0; j < inputValues[0].size(); j++)
        {
            z += inputValues[i][j] * weights[j];
        }

        z += weights[inputValues[0].size()]; // Add bias term
        // cout << i << "is runnnnnnnning\n\n";
        predictedValue = round(activation(z));

        if (predictedValue == expectedOutput[i])
        {
            totalCorrect++;
        }
    }

    cout << "Accuracy is: " << (totalCorrect * 100.0) / totalCases << "%" << endl;
    final_result = max(final_result, (totalCorrect * 100.0) / totalCases);
}

void test()
{
    double z = 0;
    cout << "Enter the values" << endl;
    for (int i = 0; i < 8; i++)
    {
        double temp;
        cin >> temp;
        z += weights[i] * temp;
    }
    z += weights[8]; // Add bias term

    double predictedValue = activation(z);

    if (predictedValue < 0.5)
    {
        cout << "0" << endl;
    }
    else
    {
        cout << "1" << endl;
    }
}
