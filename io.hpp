#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;
int roundValue = 1e3;
vector<vector<double>> ReadDatasetFromCSV(string filename)
{
    fstream fin;
    fin.open(filename, ios::in);

    vector<double> row;
    vector<vector<double>> dataset;
    string line, word;
    double value;

    // get rid of the header row
    getline(fin, line);
    while (fin.good())
    {
        // std::cout << "good\n";
        row.clear();

        // Add bias
        // row.push_back(1);

        //        row.push_back(1);

        // read 1 line
        getline(fin, line);

        // put the line to stream
        stringstream ssline(line);

        // read one field at a time
        while (getline(ssline, word, ','))
        {
            stringstream ssword(word);

            // convert the value from string to double
            ssword >> value;
            row.push_back(value);
        }

        dataset.push_back(row);
    }
    fin.close();
    return dataset;
}


vector<vector<int64_t>> ReadDatasetBGVFromCSV(string filename)
{
    fstream fin;
    fin.open(filename, ios::in);

    vector<int64_t> row;
    vector<vector<int64_t>> dataset;
    string line, word;
    double value;

    // get rid of the header row
    getline(fin, line);
    while (fin.good())
    {
        // std::cout << "good\n";
        row.clear();

        // Add bias
        // row.push_back(1);

        //        row.push_back(1);

        // read 1 line
        getline(fin, line);

        // put the line to stream
        stringstream ssline(line);

        // read one field at a time
        while (getline(ssline, word, ','))
        {
            stringstream ssword(word);

            // convert the value from string to double
            ssword >> value;
            value*=roundValue;
            row.push_back((int)value);
        }

        dataset.push_back(row);
    }
    fin.close();
    return dataset;
}


// Write to a csv file
// If the file exists, its contents will be overwritten
// Otherwise, create a new file and write to it
void WriteWeightsToCSV(string filename, const vector<double> &data)
{
    fstream fout;
    fout.open(filename, ios::out);

    for (int i = 0; i < data.size(); ++i)
    {
        fout << data[i];
        if (i < data.size() - 1)
        {
            fout << ",";
        }
    }
    fout.close();
}

void WriteWeightsBGVToCSV(string filename, const vector<int64_t> &data)
{
    fstream fout;
    fout.open(filename, ios::out);

    for (int i = 0; i < data.size(); ++i)
    {
        fout << data[i];
        if (i < data.size() - 1)
        {
            fout << ",";
        }
    }
    fout.close();
}

vector<double> ReadWeightsFromCSV(string filename)
{
    fstream fin;
    fin.open(filename, ios::in);

    vector<double> weights;
    string line, word;
    double value;

    while (fin.good())
    {
        getline(fin, line);
        stringstream ssline(line);
        while (getline(ssline, word, ','))
        {
            stringstream ssword(word);
            ssword >> value;
            weights.push_back(value);
        }
    }
    return weights;
}


vector<int64_t> ReadWeightsBGVFromCSV(string filename)
{
    fstream fin;
    fin.open(filename, ios::in);

    vector<int64_t> weights;
    string line, word;
    double value;

    while (fin.good())
    {
        getline(fin, line);
        stringstream ssline(line);
        while (getline(ssline, word, ','))
        {
            stringstream ssword(word);
            ssword >> value;
            weights.push_back(value);
        }
    }
    return weights;
}


vector<double> ExtractLabel(vector<vector<double>> &dataset, int col_idx)
{
    vector<double> labels;
    for (int i = 0; i < dataset.size(); ++i)
    {
        labels.push_back(dataset[i][col_idx]);
        dataset[i].erase(dataset[i].begin() + col_idx);
    }
    return labels;
}


vector<int64_t> ExtractLabelBGV(vector<vector<int64_t>> &dataset, int col_idx)
{
    vector<int64_t> labels;
    for (int i = 0; i < dataset.size(); ++i)
    {
        labels.push_back(dataset[i][col_idx]);
        dataset[i].erase(dataset[i].begin() + col_idx);
    }
    return labels;
}
