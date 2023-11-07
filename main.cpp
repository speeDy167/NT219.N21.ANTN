#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "openfhe.h"
// #include "homomorphic.hpp"
#include "io.hpp"
#include "plain_algo.hpp"
using namespace std;
#define MAX_ITER 10
using namespace lbcrypto;
CCParams<CryptoContextCKKSRNS> parameters;
CryptoContext<DCRTPoly> cryptoContext;
uint32_t polyDegree = 59;
KeyPair<DCRTPoly> keyPair;
uint32_t multDepth = 7;
uint32_t scaleModSize = 59;
uint32_t batchSize = 8;
usint firstModSize = 60;

Ciphertext<DCRTPoly> Sigmoid(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly> &encrypted_product)
{
    auto x2 = cc->EvalSquare(encrypted_product);
    auto x4 = cc->EvalSquare(x2);
    auto x5 = cc->EvalMult(x4, encrypted_product);
    // perform 0.002*x^5
    x5 = cc->EvalMult(x5, 0.002);

    // perform 0.021*x^3
    auto x3 = cc->EvalMult(x2, encrypted_product);
    x3 = cc->EvalMult(x3, 0.021);
    
    // perform 0.25*x
    auto x1 = cc->EvalMult(encrypted_product, 0.25);

    // perform 0.002*x^5 - 0.021*x^3
    auto final_result = cc->EvalSub(x5, x3);

    // perform 0.002*x^5 - 0.021*x^3 + 0.25*x
    final_result = cc->EvalAdd(x1, final_result);
    
    // perform 0.002*x^5 - 0.021*x^3 + 0.25*x
    final_result = cc->EvalAdd(final_result, 0.5);

    return final_result;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypt(std::vector<double> x)
{
    lbcrypto::Plaintext plain = cryptoContext->MakeCKKSPackedPlaintext(x);
    auto cipher = cryptoContext->Encrypt(keyPair.publicKey, plain);
    return cipher;
}

Ciphertext<DCRTPoly> PartialDerivative(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly> &sigmoid,
                                       const Ciphertext<DCRTPoly> &x_encrypted,
                                       const Ciphertext<DCRTPoly> &y_encrypted)
{
    auto result = cc->EvalSub(y_encrypted, sigmoid);
    result = cc->EvalMult(result, x_encrypted);
    return result;
}

Ciphertext<DCRTPoly> SumPartialDerivative(CryptoContext<DCRTPoly> cc, const vector<Ciphertext<DCRTPoly>> &derivatives)
{
    auto encrypted_sum = derivatives[0];
    for (size_t i = 1; i < derivatives.size(); ++i)
    {
        encrypted_sum = cryptoContext->EvalAdd(encrypted_sum, derivatives[i]);
    }

    return encrypted_sum;
}

int main()
{
    // Set up parameters
    CCParams<CryptoContextCKKSRNS> parameters;

    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    parameters.SetFirstModSize(firstModSize);

    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetRingDim(1 << 15);
    parameters.SetPlaintextModulus(20);
    cryptoContext = GenCryptoContext(parameters);

    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);

    cryptoContext->Enable(ADVANCEDSHE);

    keyPair = cryptoContext->KeyGen();

    keyPair = cryptoContext->KeyGen();

    cryptoContext->EvalMultKeyGen(keyPair.secretKey);

    // std::cout << "CKKS scheme is using ring dimension " << cryptoContext->GetRingDimension() << std::endl << std::endl;

    auto train_features = ReadDatasetFromCSV("/home/im5hry/Project_Crypto/dataset/diabetes_normalized.csv");

    if (train_features.back().size() == 0)
    {
        train_features.pop_back();
    }

    auto labels = ExtractLabel(train_features, 8);

    /// Set up weights
    vector<double> weights = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};

    double learning_rate = 0.01;

    std::cout << "\tParameters: \n\n";
    std::cout << "\tSecret key: " << keyPair.secretKey << "\n\n";
    std::cout << "\tPublic key: " << keyPair.publicKey << "\n\n";
    std::cout << "\tMultiDepth: " << parameters.GetMultiplicativeDepth() << "\n\n";
    std::cout << "\tBatchSize: " << parameters.GetBatchSize() << "\n\n";
    std::cout << "\tRing Dimensions: " << parameters.GetRingDim() << "\n\n";
    std::cout << "\tScaling Mod size: " << parameters.GetScalingModSize() << "\n\n";

    std::cout << "Encrypting features...\n\n\n";

    // Set up learning rate

    vector<double> vector_lr;
    for (int i = 0; i < train_features[0].size(); i++)
    {
        vector_lr.push_back(learning_rate);
    }
    auto encrypted_learning_rate = encrypt(vector_lr);
    // auto scaling = encrypted_learning_rate->GetScalingFactor();
    //   std::cout  << " Scaling: " << scaling << std::endl;
    // Set up training_features
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_features;
    for (int i = 0; i < train_features.size(); ++i)
    {
        auto encrypted_feature = encrypt(train_features[i]);
        encrypted_features.push_back((encrypted_feature));
        // std::cout <<i << "\n";
    }

    // Set up training_labels
    std::cout << "Encrypting labels...\n\n\n";
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_labels;
    for (int i = 0; i < labels.size(); ++i)
    {
        vector<double> outputs;
        for (int j = 0; j < train_features[0].size(); ++j)
            outputs.push_back(labels[i]);
        auto encrypted_label = encrypt(outputs);

        encrypted_labels.push_back(encrypted_label);
    }

    // Homomorphic training
    double best_acc = 0;
    auto encrypted_weights = encrypt(weights);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 20; iter++)
    {
        cout << "Iteration " << iter << " is running...\n";
        /*Hospital works*/
        // encrypt product
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_products;
        vector<double> products;
        vector<vector<double>> set_products;
        for (int i = 0; i < train_features.size(); ++i)
        {
            double product = PlainVectorMultiplication(train_features[i], weights);

            for (int j = 0; j < train_features[0].size(); ++j)
                products.push_back(product);
            set_products.push_back(products);

            auto encrypted_product = encrypt(products);
            products.clear();
            encrypted_products.push_back(encrypted_product);
        }

        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/

        /*Cloud works*/
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> partial_derivatives;
        
        for (size_t i = 0; i < train_features.size(); ++i)
        {
            // ----------------------------------------------------------------- //
            // ----------------------------------------------------------------- //
            // Perform sigmoid function
            auto sigmoid = Sigmoid(cryptoContext, encrypted_products[i]);


            // ----------------------------------------------------------------- //
            // Compute the partial derivative of the weighted sample
            auto partial_derivative =
                PartialDerivative(cryptoContext, sigmoid, encrypted_features[i], encrypted_labels[i]);
            partial_derivatives.push_back(partial_derivative);
        }

        auto encrypted_derivatives_sum = SumPartialDerivative(cryptoContext, partial_derivatives);

        Plaintext plaintextDec;
        std::vector<std::complex<double>> finalResult;
        // Compute train_weight
        auto encrypted_train_weight = cryptoContext->EvalMult(encrypted_learning_rate, encrypted_derivatives_sum);
        
        auto train_weight = encrypted_weights;
        train_weight = cryptoContext->EvalAdd(train_weight, encrypted_train_weight);

        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/

        /*Hospital works*/

        cryptoContext->Decrypt(keyPair.secretKey, train_weight, &plaintextDec);

        finalResult = plaintextDec->GetCKKSPackedValue();
        //        std::cout << "Actual summmm\n\t" << finalResult << std::endl << std::endl;

        std::cout << "weights[" << iter << "]: ";
        for (int i = 0; i < finalResult.size() - 1; i++)
        {
            weights[i] = finalResult[i].real();
            std::cout << weights[i] << ", ";
        }
        weights[finalResult.size() - 1] = finalResult[finalResult.size() - 1].real();
        std::cout << weights[finalResult.size() - 1];
        encrypted_weights = train_weight;
        double train_accuracy = ComputeAccuracy(train_features, labels, weights);
        if (train_accuracy > best_acc)
        {
            best_acc = train_accuracy;
            WriteWeightsToCSV("/home/im5hry/Project_Crypto/weights/best_weights.csv", weights);
        }
        cout << "\n\n\n\n\nTrain accuracy: " << train_accuracy << endl;
    }

    /*Result*/
    weights = ReadWeightsFromCSV("/home/im5hry/Project_Crypto/weights/best_weights.csv");
    cout << "\n\n\nFinal result: ";
    cout << "Best accuracy: " << best_acc << "\n";
    cout << "Best weight: ";
    for (int j = 0; j < weights.size(); j++)
        cout << weights[j] << ", ";
    cout << weights[weights.size() - 1];

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\n\n\nTraining time: " << duration.count() / 1000000 << " seconds" << std::endl;
}
