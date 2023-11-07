#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "openfhe.h"
// #include "homomorphic.hpp"
#include "io.hpp"
#include "plain_algo.hpp"
using namespace std;
#define MAX_ITER 20
using namespace lbcrypto;
CCParams<CryptoContextBGVRNS> parameters;
CryptoContext<DCRTPoly> cryptoContext;
uint32_t polyDegree = 59;
KeyPair<DCRTPoly> keyPair;
uint32_t multDepth = 7;
uint32_t scaleModSize = 59;
uint32_t batchSize = 8;
usint firstModSize = 60;

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypt(std::vector<int64_t> x)
{
    lbcrypto::Plaintext plain = cryptoContext->MakePackedPlaintext(x);
    auto cipher = cryptoContext->Encrypt(keyPair.publicKey, plain);
    return cipher;
}
Ciphertext<DCRTPoly> Sigmoid(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly> &encrypted_product)
{
    auto x2 = cc->EvalSquare(encrypted_product);
    auto x4 = cc->EvalSquare(x2);
    auto x5 = cc->EvalMult(x4, encrypted_product);

    vector<int64_t> coef={20};
    auto haimuoi = encrypt(coef);
    coef.pop_back();
    x5 = cc->EvalMult(x5, haimuoi);

    coef.push_back(210);
    auto x3 = cc->EvalMult(x2, encrypted_product);
    auto haitram10 = encrypt(coef);
    x3 = cc->EvalMult(x3, haitram10);
    coef.pop_back();
    coef.push_back(2500);
    auto hainghin500 = encrypt(coef);
    auto x1 = cc->EvalMult(encrypted_product, hainghin500);

    auto final_result = cc->EvalSub(x5, x3);

    coef.pop_back();
    coef.push_back(5000);
    auto namnghin = encrypt(coef);
    final_result = cc->EvalAdd(x1, final_result);
    final_result = cc->EvalAdd(final_result, namnghin);

    return final_result;
}


Ciphertext<DCRTPoly> PartialDerivative(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly> &sigmoid,
                                       const Ciphertext<DCRTPoly> &x_encrypted,
                                       const Ciphertext<DCRTPoly> &y_encrypted)
{
    auto result = cc->EvalSub(y_encrypted, sigmoid);
    // cout << "loi o day\n\n";
    result = cc->EvalMult(result, x_encrypted);
    // cout << "loi khong o day\n\n";
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
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 10);
    parameters.SetPlaintextModulus(1017348097);
    parameters.SetMaxRelinSkDeg(3);
    cryptoContext = GenCryptoContext(parameters);
    
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    
    cryptoContext->Enable(ADVANCEDSHE);
    //Gen key 
    keyPair = cryptoContext->KeyGen();

    // Generate the relinearization key
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);


    auto train_features = ReadDatasetBGVFromCSV("/home/im5hry/Project_Crypto/dataset/diabetes_normalized.csv");
    //    std::cout << train_features.size();
    if (train_features.back().size() == 0)
    {
        train_features.pop_back();
    }

    auto labels = ExtractLabelBGV(train_features, 8);
    int64_t learning_rate = 1;

    /// Set up weights
    vector<int64_t> weights = {1, 1, 1, 1, 1, 1, 1, 1};
    // std::cout << "ok1\n";

    // for (int i = 0; i < weights.size(); i++) {
    //         cout << weights[i] << " ";
    //     cout << endl;
    // }
    std::cout << "Parameters: \n\n";
    std::cout << "Secret key: " << keyPair.secretKey << "\n\n";
    std::cout << "Public key: " << keyPair.publicKey << "\n\n";
    std::cout << "MultiDepth: " << parameters.GetMultiplicativeDepth() << "\n\n";
    std::cout << "BatchSize: " << parameters.GetBatchSize() << "\n\n";
    std::cout << "Ring Dimensions: " << parameters.GetRingDim() << "\n\n";
    std::cout << "Scaling Mod size: " << parameters.GetScalingModSize() << "\n\n";

    // Set up learning rate

    vector<int64_t> vector_lr;
    for (int i = 0; i < train_features[0].size(); i++)
    {
        vector_lr.push_back(learning_rate);
    }
    auto encrypted_learning_rate = encrypt(vector_lr);
    std::cout << "Encrypting features...\n\n\n";
    
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_features;
    for (int i = 0; i < train_features.size(); ++i)
    {
        // cout <<
        auto encrypted_feature = encrypt(train_features[i]);
        encrypted_features.push_back((encrypted_feature));
        // std::cout << i << "\n";
    }

    // Set up training_labels
    std::cout << "Encrypting labels...\n\n\n";
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_labels;
    for (int i = 0; i < labels.size(); ++i)
    {
        vector<int64_t> outputs;
        for (int j = 0; j < train_features[0].size(); ++j)
            outputs.push_back(labels[i]);
        auto encrypted_label = encrypt(outputs);

        encrypted_labels.push_back(encrypted_label);
    }

    // Homomorphic training
    double best_acc = 0;
    auto encrypted_weights = encrypt(weights);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        cout << "Iteration " << iter << " is running...\n";
        
        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/

        /*Hospital works*/
        // encrypt product
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_products;
        vector<int64_t> products;
        vector<vector<int64_t>> set_products;
        for (int i = 0; i < train_features.size(); ++i)
        {

            int64_t product = PlainVectorMultiplicationBGV(train_features[i], weights);
            // cout << "loi day\n\n";
            // if(iter == 1) {
            //     cout << "product: "<<product << "\n\n";
            // }
            for (int j = 0; j < train_features[0].size(); ++j)
                products.push_back(product);
            set_products.push_back(products);

            // std::cout << product << std::endl;
            // r.clear();
            auto encrypted_product = encrypt(products);
            products.clear();
            encrypted_products.push_back(encrypted_product);
        }

        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/

        /*Cloud works*/
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> partial_derivatives;
        // std::cout << train_features[0].size() << "hehehehe\n\n";
        for (size_t i = 0; i < train_features.size(); ++i)
        {
            // cout << "sample " << i << "\n\n\n";
            // ----------------------------------------------------------------- //
            // std::cout << encrypted_products[i] << std::endl;
            // encrypted_sample_x_weights -> Level 5
            // ----------------------------------------------------------------- //
            // Perform sigmoid function
            //        auto sigmoid = cryptoContext->EvalLogistic(encrypted_products[i], 0, set_products[i][0], polyDegree);
            // cout << "loi o day\n\n";

            auto sigmoid = Sigmoid(cryptoContext, encrypted_products[i]);
            // cout << "loi khong o day\n\n";

            // sigmoid.scale() = scale;
            // sigmoid -> Level 2

            // ----------------------------------------------------------------- //
            // Compute the partial derivative of the weighted sample
            auto partial_derivative =
                PartialDerivative(cryptoContext, sigmoid, encrypted_features[i], encrypted_labels[i]);
            partial_derivatives.push_back(partial_derivative);
        }

        auto encrypted_derivatives_sum = SumPartialDerivative(cryptoContext, partial_derivatives);

        Plaintext plaintextDec;
        std::vector<int64_t> finalResult;
        auto encrypted_train_weight = cryptoContext->EvalMult(encrypted_learning_rate, encrypted_derivatives_sum);

        cryptoContext->Decrypt(keyPair.secretKey, encrypted_train_weight, &plaintextDec);

        // finalResult = plaintextDec->GetRealPackedValue();
        // std::cout << "here is final: " << finalResult << "\n\n";
        auto train_weight = encrypted_weights;
        train_weight = cryptoContext->EvalAdd(train_weight, encrypted_train_weight);

        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/
        /*------------------------------------------------------------------------------------------------*/

        /*Hospital works*/

        cryptoContext->Decrypt(keyPair.secretKey, train_weight, &plaintextDec);

        finalResult = plaintextDec->GetPackedValue();
        //        std::cout << "Actual summmm\n\t" << finalResult << std::endl << std::endl;

        std::cout << "weights[" << iter << "]: ";
        cout << plaintextDec << "\n\n";
        for (int i = 0; i < 7; i++)
        {
            weights[i] = (int64_t)(finalResult[i]/100);
            std::cout << weights[i] << ", ";
        }
        weights[7] = (int64_t)(finalResult[7]/100);
        std::cout << weights[7];
        encrypted_weights = train_weight;
        double train_accuracy = ComputeAccuracy(train_features, labels, weights);
        if (train_accuracy > best_acc)
        {
            best_acc = train_accuracy;
            WriteWeightsBGVToCSV("/home/im5hry/Project_Crypto/weights/best_weights.csv", weights);
        }
        cout << "\n\n\n\n\nTrain accuracy: " << train_accuracy << endl;
    }

    /*Result*/
    weights = ReadWeightsBGVFromCSV("/home/im5hry/Project_Crypto/weights/best_weights.csv");
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
