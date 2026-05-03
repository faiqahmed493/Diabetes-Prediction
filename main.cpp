#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <memory>
#include <numeric>
#include <iomanip>
#include <set>
#include <functional>
#include <string>

using namespace std;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct DataPoint {
    vector<double> features;
    int label;
};

struct TreeNode {
    int idx;
    int label;
    int featureIndex;
    double threshold;
    shared_ptr<TreeNode> left;
    shared_ptr<TreeNode> right;
    
    TreeNode(int lbl, int i) : label(lbl), idx(i), featureIndex(-1), 
                                threshold(0.0), left(nullptr), right(nullptr) {}
};

struct Metrics {
    double accuracy;
    double precision;
    double recall;
    double f1Score;
    double rocAuc;
    int tp, tn, fp, fn;
};

struct Config {
    string splitCriterion;
    int maxDepth;
    int minSamplesSplit;
    int minSamplesLeaf;
    double ccpAlpha;
    int numTrees;
    string maxFeatures;
    bool bootstrap;
    bool useSMOTE;
    int validationFolds;
};

// ============================================================================
// DATASET LOADER
// ============================================================================

class DatasetLoader {
public:
    static vector<DataPoint> loadCSV(const string& filename, bool hasLabel = true) {
        vector<DataPoint> data;
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return data;
        }
        
        string line;
        bool firstLine = true;
        
        while (getline(file, line)) {
            if (firstLine) {
                firstLine = false;
                continue;
            }
            
            stringstream ss(line);
            DataPoint point;
            string value;
            
            try {
                while (getline(ss, value, ',')) {
                    if (!hasLabel || ss.peek() != EOF) {
                        point.features.push_back(value.empty() ? NAN : stod(value));
                    } else {
                        point.label = stoi(value);
                    }
                }
                
                if (hasLabel && point.features.size() > 0) {
                    data.push_back(point);
                }
            } catch (const exception& e) {
                cerr << "Warning: Skipping invalid line" << endl;
            }
        }
        
        cout << "Loaded " << data.size() << " samples" << endl;
        return data;
    }
    
    static pair<vector<DataPoint>, vector<DataPoint>> stratifiedSplit(
        const vector<DataPoint>& data, double trainRatio = 0.8, int seed = 42) {
        
        map<int, vector<DataPoint>> labelGroups;
        for (const auto& point : data) {
            labelGroups[point.label].push_back(point);
        }
        
        vector<DataPoint> trainData, testData;
        mt19937 gen(seed);
        
        for (auto& it : labelGroups) {
            auto label = it.first; 
            auto points = it.second; 
            shuffle(points.begin(), points.end(), gen);
            int trainSize = static_cast<int>(points.size() * trainRatio);
            
            trainData.insert(trainData.end(), points.begin(), points.begin() + trainSize);
            testData.insert(testData.end(), points.begin() + trainSize, points.end());
        }
        
        shuffle(trainData.begin(), trainData.end(), gen);
        shuffle(testData.begin(), testData.end(), gen);
        
        cout << "Training samples: " << trainData.size() << endl;
        cout << "Testing samples: " << testData.size() << endl;
        
        return {trainData, testData};
    }
};

// ============================================================================
// PREPROCESSOR
// ============================================================================

class Preprocessor {
private:
    vector<double> medians;
    vector<pair<double, double>> iqrBounds;
    bool fitted;
    
public:
    Preprocessor() : fitted(false) {}
    
    void fit(vector<DataPoint>& data) {
        if (data.empty()) return;
        
        int numFeatures = data[0].features.size();
        medians.resize(numFeatures);
        iqrBounds.resize(numFeatures);
        
        // 1. Median Imputation
        for (int f = 0; f < numFeatures; ++f) {
            vector<double> validValues;
            for (const auto& point : data) {
                if (!isnan(point.features[f])) {
                    validValues.push_back(point.features[f]);
                }
            }
            
            if (!validValues.empty()) {
                sort(validValues.begin(), validValues.end());
                medians[f] = validValues[validValues.size() / 2];
            }
        }
        
        // Apply imputation
        for (auto& point : data) {
            for (int f = 0; f < numFeatures; ++f) {
                if (isnan(point.features[f])) {
                    point.features[f] = medians[f];
                }
            }
        }
        
        // 2. IQR Outlier Detection
        for (int f = 0; f < numFeatures; ++f) {
            vector<double> values;
            for (const auto& point : data) {
                values.push_back(point.features[f]);
            }
            
            sort(values.begin(), values.end());
            double q1 = values[values.size() / 4];
            double q3 = values[3 * values.size() / 4];
            double iqr = q3 - q1;
            
            iqrBounds[f] = {q1 - 1.5 * iqr, q3 + 1.5 * iqr};
        }
        
        // Replace outliers with median
        for (auto& point : data) {
            for (int f = 0; f < numFeatures; ++f) {
                if (point.features[f] < iqrBounds[f].first || 
                    point.features[f] > iqrBounds[f].second) {
                    point.features[f] = medians[f];
                }
            }
        }
        
        fitted = true;
        cout << "Preprocessing fitted on training data" << endl;
    }
    
    void transform(vector<DataPoint>& data) {
        if (!fitted) {
            cerr << "Error: Preprocessor not fitted!" << endl;
            return;
        }
        
        int numFeatures = data[0].features.size();
        
        // Apply median imputation
        for (auto& point : data) {
            for (int f = 0; f < numFeatures; ++f) {
                if (isnan(point.features[f])) {
                    point.features[f] = medians[f];
                }
            }
        }
        
        // Apply IQR bounds
        for (auto& point : data) {
            for (int f = 0; f < numFeatures; ++f) {
                if (point.features[f] < iqrBounds[f].first || 
                    point.features[f] > iqrBounds[f].second) {
                    point.features[f] = medians[f];
                }
            }
        }
        
        cout << "Preprocessing applied to data" << endl;
    }
};

// ============================================================================
// SMOTE GENERATOR
// ============================================================================

class SMOTEGenerator {
public:
    static vector<DataPoint> applySMOTE(const vector<DataPoint>& data, int k = 5, int seed = 42) {
        map<int, vector<DataPoint>> labelGroups;
        for (const auto& point : data) {
            labelGroups[point.label].push_back(point);
        }
        
        int maxCount = 0;
        for (const auto& it: labelGroups) {
            auto label = it.first; 
            auto points = it.second; 
            maxCount = max(maxCount, (int)points.size());
        }
        
        vector<DataPoint> balancedData = data;
        mt19937 gen(seed);
        uniform_real_distribution<> dis(0.0, 1.0);
        
        for (auto&  it : labelGroups) {
            auto label = it.first; 
            auto points = it.second; 
            int needed = maxCount - points.size();
            
            for (int i = 0; i < needed; ++i) {
                int idx = uniform_int_distribution<>(0, points.size() - 1)(gen);
                const DataPoint& sample = points[idx];
                
                // Find k nearest neighbors
                vector<pair<double, int>> distances;
                for (int j = 0; j < points.size(); ++j) {
                    if (j != idx) {
                        double dist = 0.0;
                        for (size_t f = 0; f < sample.features.size(); ++f) {
                            dist += pow(sample.features[f] - points[j].features[f], 2);
                        }
                        distances.push_back({sqrt(dist), j});
                    }
                }
                
                sort(distances.begin(), distances.end());
                int neighborIdx = distances[min(k-1, (int)distances.size()-1)].second;
                
                // Generate synthetic sample
                DataPoint synthetic;
                synthetic.label = label;
                double lambda = dis(gen);
                
                for (size_t f = 0; f < sample.features.size(); ++f) {
                    synthetic.features.push_back(
                        sample.features[f] + lambda * 
                        (points[neighborIdx].features[f] - sample.features[f])
                    );
                }
                
                balancedData.push_back(synthetic);
            }
        }
        
        shuffle(balancedData.begin(), balancedData.end(), gen);
        cout << "SMOTE applied: " << data.size() << " -> " << balancedData.size() << " samples" << endl;
        
        return balancedData;
    }
};

// ============================================================================
// DECISION TREE
// ============================================================================

class DecisionTree {
private:
    Config config;
    int nodeIdx;
    
    double calculateGini(const vector<DataPoint>& data) {
        map<int, int> labelCount;
        for (const auto& point : data) {
            labelCount[point.label]++;
        }
        
        double gini = 1.0;
        for (const auto& it: labelCount) {
             auto label = it.first; 
            auto count = it.second; 
            double prob = static_cast<double>(count) / data.size();
            gini -= prob * prob;
        }
        return gini;
    }
    
    double calculateEntropy(const vector<DataPoint>& data) {
        map<int, int> labelCount;
        for (const auto& point : data) {
            labelCount[point.label]++;
        }
        
        double ent = 0.0;
        for (const auto& it: labelCount) {
            auto label = it.first; 
            auto count = it.second; 
            double prob = static_cast<double>(count) / data.size();
            if (prob > 0) {
                ent -= prob * log2(prob);
            }
        }
        return ent;
    }
    
    pair<vector<DataPoint>, vector<DataPoint>> splitData(
        const vector<DataPoint>& data, int featureIndex, double threshold) {
        
        vector<DataPoint> leftSubset, rightSubset;
        for (const auto& point : data) {
            if (point.features[featureIndex] < threshold) {
                leftSubset.push_back(point);
            } else {
                rightSubset.push_back(point);
            }
        }
        return {leftSubset, rightSubset};
    }
    
    pair<int, double> findBestSplit(const vector<DataPoint>& data) {
        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestGain = 0.0;
        
        double parentScore = (config.splitCriterion == "gini") ? 
                            calculateGini(data) : calculateEntropy(data);
        
        // If parent is already pure, don't split
        if (parentScore < 1e-7) {
            return {-1, 0.0};
        }
        
        for (size_t f = 0; f < data[0].features.size(); ++f) {
            vector<double> values;
            for (const auto& point : data) {
                values.push_back(point.features[f]);
            }
            
            sort(values.begin(), values.end());
            values.erase(unique(values.begin(), values.end()), values.end());
            
            // Need at least 2 unique values to split
            if (values.size() < 2) continue;
            
            for (size_t i = 1; i < values.size(); ++i) {
                double threshold = (values[i-1] + values[i]) / 2.0;
                

                auto it = splitData(data, f, threshold);
                auto leftData = it.first; 
                auto rightData = it.second; 
                
                if (leftData.size() < config.minSamplesLeaf || 
                    rightData.size() < config.minSamplesLeaf) {
                    continue;
                }
                
                // Check if both children would have same label
                int leftLabel = getMajorityLabel(leftData);
                int rightLabel = getMajorityLabel(rightData);
                
                // Count labels in each subset
                map<int, int> leftLabelCount, rightLabelCount;
                for (const auto& p : leftData) leftLabelCount[p.label]++;
                for (const auto& p : rightData) rightLabelCount[p.label]++;
                
                // If both subsets are pure and have same label, skip this split
                if (leftLabelCount.size() == 1 && rightLabelCount.size() == 1 && 
                    leftLabel == rightLabel) {
                    continue;
                }
                
                double leftScore = (config.splitCriterion == "gini") ? 
                                  calculateGini(leftData) : calculateEntropy(leftData);
                double rightScore = (config.splitCriterion == "gini") ? 
                                   calculateGini(rightData) : calculateEntropy(rightData);
                
                double weightedScore = (leftData.size() * leftScore + 
                                       rightData.size() * rightScore) / data.size();
                
                double gain = parentScore - weightedScore;
                
                // Only accept split if it provides meaningful gain
                if (gain > bestGain && gain > 1e-7) {
                    bestGain = gain;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }
        
        return {bestFeature, bestThreshold};
    }
    
    int getMajorityLabel(const vector<DataPoint>& data) {
        map<int, int> labelCount;
        for (const auto& point : data) {
            labelCount[point.label]++;
        }
        return max_element(labelCount.begin(), labelCount.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
    
    shared_ptr<TreeNode> buildTreeRecursive(const vector<DataPoint>& data, int depth) {
        map<int, int> labelCount;
        for (const auto& point : data) {
            labelCount[point.label]++;
        }
        
        // Stopping criteria
        if (labelCount.size() == 1 || 
            data.size() < config.minSamplesSplit ||
            depth >= config.maxDepth) {
            
            int majorityLabel = getMajorityLabel(data);
            return make_shared<TreeNode>(majorityLabel, nodeIdx++);
        }
        
        auto it = findBestSplit(data);
        auto bestFeature = it.first; 
        auto bestThreshold = it.second; 
        
        if (bestFeature == -1) {
            int majorityLabel = getMajorityLabel(data);
            return make_shared<TreeNode>(majorityLabel, nodeIdx++);
        }
        
        auto result = splitData(data, bestFeature, bestThreshold);
        vector<DataPoint>  leftData = result.first; 
        vector<DataPoint> rightData= result.second;
        
        // Check if split is empty on either side
        if (leftData.empty() || rightData.empty()) {
            int majorityLabel = getMajorityLabel(data);
            return make_shared<TreeNode>(majorityLabel, nodeIdx++);
        }
        
        // Build child nodes
        auto leftChild = buildTreeRecursive(leftData, depth + 1);
        auto rightChild = buildTreeRecursive(rightData, depth + 1);
        
        // If both children are leaves with the same label, 
        // don't create an internal node - just return a leaf
        if (leftChild->label != -1 && rightChild->label != -1 && 
            leftChild->label == rightChild->label) {
            return make_shared<TreeNode>(leftChild->label, nodeIdx++);
        }
        
        auto node = make_shared<TreeNode>(-1, nodeIdx++);
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = leftChild;
        node->right = rightChild;
        
        return node;
    }
    
public:
    shared_ptr<TreeNode> root;
    
    DecisionTree(const Config& cfg) : config(cfg), nodeIdx(1), root(nullptr) {}
    
    void train(const vector<DataPoint>& data) {
        nodeIdx = 1;
        root = buildTreeRecursive(data, 0);
        cout << "Decision tree trained" << endl;
    }
    
    int predict(const DataPoint& point) const {
        auto node = root;
        while (node->left && node->right) {
            if (point.features[node->featureIndex] < node->threshold) {
                node = node->left;
            } else {
                node = node->right;
            }
        }
        return node->label;
    }
    
    void saveToJSON(const string& filename) const {
        ofstream file(filename);
        function<void(shared_ptr<TreeNode>, ostream&)> serialize = 
            [&](shared_ptr<TreeNode> node, ostream& os) {
            if (!node) {
                os << "null";
                return;
            }
            os << "{\"nodeIndex\":" << node->idx 
               << ",\"featureIndex\":" << node->featureIndex
               << ",\"threshold\":" << node->threshold
               << ",\"label\":" << (node->label == -1 ? "null" : to_string(node->label))
               << ",\"left\":";
            serialize(node->left, os);
            os << ",\"right\":";
            serialize(node->right, os);
            os << "}";
        };
        
        serialize(root, file);
        file.close();
    }
};

// ============================================================================
// RANDOM FOREST
// ============================================================================

class RandomForest {
private:
    Config config;
    vector<shared_ptr<DecisionTree>> trees;
    
public:
    RandomForest(const Config& cfg) : config(cfg) {}
    
    void train(const vector<DataPoint>& data) {
        mt19937 gen(42);
        uniform_int_distribution<> dis(0, data.size() - 1);
        
        for (int i = 0; i < config.numTrees; ++i) {
            cout << "Building tree " << (i + 1) << "/" << config.numTrees << "..." << endl;
            
            vector<DataPoint> sampleData;
            
            if (config.bootstrap) {
                for (size_t j = 0; j < data.size(); ++j) {
                    sampleData.push_back(data[dis(gen)]);
                }
            } else {
                sampleData = data;
            }
            
            auto tree = make_shared<DecisionTree>(config);
            tree->train(sampleData);
            trees.push_back(tree);
        }
        
        cout << "Random Forest trained with " << config.numTrees << " trees" << endl;
    }
    
    int predict(const DataPoint& point) const {
        map<int, int> votes;
        for (const auto& tree : trees) {
            votes[tree->predict(point)]++;
        }
        
        return max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
    
    void saveAllTrees(const string& prefix) const {
        for (size_t i = 0; i < trees.size(); ++i) {
            trees[i]->saveToJSON(prefix + to_string(i + 1) + ".json");
        }
    }
};

// ============================================================================
// METRICS CALCULATOR
// ============================================================================

class MetricsCalculator {
public:
    template<typename Model>
    static Metrics evaluate(const Model& model, const vector<DataPoint>& data) {
        Metrics m = {0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0};
        
        for (const auto& point : data) {
            int pred = model.predict(point);
            int actual = point.label;
            
            if (pred == 1 && actual == 1) m.tp++;
            else if (pred == 0 && actual == 0) m.tn++;
            else if (pred == 1 && actual == 0) m.fp++;
            else if (pred == 0 && actual == 1) m.fn++;
        }
        
        m.accuracy = (m.tp + m.tn) / static_cast<double>(data.size()) * 100.0;
        m.precision = (m.tp + m.fp > 0) ? m.tp / static_cast<double>(m.tp + m.fp) : 0.0;
        m.recall = (m.tp + m.fn > 0) ? m.tp / static_cast<double>(m.tp + m.fn) : 0.0;
        m.f1Score = (m.precision + m.recall > 0) ? 
                    2 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
        
        // Simplified ROC-AUC approximation
        m.rocAuc = (m.recall + (m.tn / static_cast<double>(m.tn + m.fp))) / 2.0;
        
        return m;
    }
    
    static void printMetrics(const Metrics& m, const string& title = "") {
        if (!title.empty()) {
            cout << "\n========== " << title << " ==========" << endl;
        }
        
        cout << fixed << setprecision(2);
        cout << "\nConfusion Matrix:" << endl;
        cout << "                Predicted" << endl;
        cout << "                0      1" << endl;
        cout << "Actual   0   [" << setw(4) << m.tn << "] [" << setw(4) << m.fp << "]" << endl;
        cout << "         1   [" << setw(4) << m.fn << "] [" << setw(4) << m.tp << "]" << endl;
        
        cout << "\nPerformance Metrics:" << endl;
        cout << "  Accuracy:  " << m.accuracy << "%" << endl;
        cout << "  Precision: " << m.precision << endl;
        cout << "  Recall:    " << m.recall << endl;
        cout << "  F1-Score:  " << m.f1Score << endl;
        cout << "  ROC-AUC:   " << m.rocAuc << endl;
    }
};

// ============================================================================
// CROSS VALIDATOR
// ============================================================================

class CrossValidator {
public:
    template<typename ModelBuilder>
    static void validate(const vector<DataPoint>& data, const Config& config, 
                        ModelBuilder buildModel, int folds) {
        
        cout << "\n========== " << folds << "-Fold Cross-Validation ==========" << endl;
        
        map<int, vector<DataPoint>> labelGroups;
        for (const auto& point : data) {
            labelGroups[point.label].push_back(point);
        }
        
        vector<Metrics> foldMetrics;
        
        for (int fold = 0; fold < folds; ++fold) {
            cout << "\n--- Fold " << (fold + 1) << "/" << folds << " ---" << endl;
            
            vector<DataPoint> trainFold, valFold;
            
            for (auto& it : labelGroups) {
                int label = it.first;
                vector<DataPoint> points = it.second;
                int foldSize = points.size() / folds;
                int start = fold * foldSize;
                int end = (fold == folds - 1) ? points.size() : start + foldSize;
                
                for (int i = 0; i < points.size(); ++i) {
                    if (i >= start && i < end) {
                        valFold.push_back(points[i]);
                    } else {
                        trainFold.push_back(points[i]);
                    }
                }
            }
            
            auto model = buildModel(trainFold);
            Metrics m = MetricsCalculator::evaluate(model, valFold);
            foldMetrics.push_back(m);
            
            cout << "  Accuracy: " << m.accuracy << "%" << endl;
        }
        
        // Average metrics
        Metrics avg = {0};
        for (const auto& m : foldMetrics) {
            avg.accuracy += m.accuracy;
            avg.precision += m.precision;
            avg.recall += m.recall;
            avg.f1Score += m.f1Score;
            avg.rocAuc += m.rocAuc;
        }
        
        avg.accuracy /= folds;
        avg.precision /= folds;
        avg.recall /= folds;
        avg.f1Score /= folds;
        avg.rocAuc /= folds;
        
        cout << "\n========== Average Cross-Validation Results ==========" << endl;
        cout << fixed << setprecision(2);
        cout << "  Accuracy:  " << avg.accuracy << "%" << endl;
        cout << "  Precision: " << avg.precision << endl;
        cout << "  Recall:    " << avg.recall << endl;
        cout << "  F1-Score:  " << avg.f1Score << endl;
        cout << "  ROC-AUC:   " << avg.rocAuc << endl;
    }
};

// ============================================================================
// CLI INTERACTION
// ============================================================================

class CLIInteraction {
public:
    static Config getUserConfig() {
        Config config;
        
        cout << "\n|===============================|" << endl;
        cout << "|   Diabetes Prediction System  |" << endl;
        cout << "|===============================|" << endl;
        
        int modelChoice = -1;
        while(modelChoice != 1 && modelChoice != 2) {
            cout << "\n[1] Choose Model:" << endl;
            cout << "    1. Decision Tree" << endl;
            cout << "    2. Random Forest" << endl;
            cout << "Enter choice (1 or 2): ";
            cin >> modelChoice;
        }
        
        int criterionChoice = -1;
        while(criterionChoice != 1 && criterionChoice != 2) {
            cout << "\n[2] Choose Split Criterion:" << endl;
            cout << "    1. Gini Impurity" << endl;
            cout << "    2. Entropy (Information Gain)" << endl;
            cout << "Enter choice (1 or 2): ";
            cin >> criterionChoice;
            config.splitCriterion = (criterionChoice == 1) ? "gini" : "entropy";
        }
        
        string smoteChoice = "";
        while(smoteChoice != "yes" && smoteChoice != "y" && smoteChoice != "no" && smoteChoice != "n") {
            cout << "\n[3] Enable SMOTE for class balancing? (yes/no): ";
            cin >> smoteChoice;
            config.useSMOTE = (smoteChoice == "yes" || smoteChoice == "y");
        }
        
        cout << "\n[4] Decision Tree Parameters:" << endl;
        do {
            cout << "    Enter max_depth (recommended: 15): ";
            cin >> config.maxDepth;
        } while(config.maxDepth <= 0);
        

        do {
            cout << "    Enter min_samples_split (recommended: 5-10): ";
            cin >> config.minSamplesSplit;
        } while(config.minSamplesSplit <= 1); 
        
        do {
            cout << "    Enter min_samples_leaf (recommended: 2-5): ";
            cin >> config.minSamplesLeaf;
        } while(config.minSamplesLeaf <= 0);
        
        do {
            cout << "    Enter pruning alpha (0 for no pruning): ";
            cin >> config.ccpAlpha;
        } while(config.ccpAlpha < 0);
        
        if (modelChoice == 2) {
            cout << "\n[5] Random Forest Parameters:" << endl;
            do {
                cout << "    Enter number of trees (recommended: 10-20): ";
                cin >> config.numTrees;
            } while(config.numTrees <= 0);
            
        
            do {
                cout << "    Enter max_features (sqrt/log2/all): ";
                cin >> config.maxFeatures;
            } while(config.maxFeatures != "sqrt" && config.maxFeatures != "log2" && config.maxFeatures != "all");
            
            
            string bootstrapChoice = "";
            while(bootstrapChoice != "yes" && bootstrapChoice != "y" && bootstrapChoice != "no" && bootstrapChoice != "n") {
                cout << "    Enable bootstrap sampling? (yes/no): ";
                cin >> bootstrapChoice;
                config.bootstrap = (bootstrapChoice == "yes" || bootstrapChoice == "y");
            }
        } else {
            config.numTrees = 1;
            config.maxFeatures = "all";
            config.bootstrap = false;
        }
        
        int validationChoice = -1;
        while(validationChoice != 1 && validationChoice != 2 && validationChoice != 3) {
            cout << "\n[6] Choose Validation Strategy:" << endl;
            cout << "    1. No validation" << endl;
            cout << "    2. 5-fold Cross-Validation" << endl;
            cout << "    3. 10-fold Cross-Validation" << endl;
            cout << "Enter choice (1, 2, or 3): ";
            cin >> validationChoice;
        }
        
        if (validationChoice == 2) config.validationFolds = 5;
        else if (validationChoice == 3) config.validationFolds = 10;
        else config.validationFolds = 0;
        
        return config;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    try {
        // Get user configuration
        Config config = CLIInteraction::getUserConfig();
        
        cout << "\n========== Loading Dataset ==========" << endl;
        string dataFile = "../data/diabetes_train.csv";
        vector<DataPoint> allData = DatasetLoader::loadCSV(dataFile);
        
        if (allData.empty()) {
            cerr << "Error: No data loaded!" << endl;
            return 1;
        }
        
        cout << "\n========== Stratified Split (80/20) ==========" << endl;
        auto result = DatasetLoader::stratifiedSplit(allData, 0.8);
        vector<DataPoint> trainData = result.first;
        vector<DataPoint> testData = result.second;
        
        cout << "\n========== Preprocessing Pipeline ==========" << endl;
        Preprocessor preprocessor;
        preprocessor.fit(trainData);
        preprocessor.transform(testData);
        
        if (config.useSMOTE) {
            cout << "\n========== Applying SMOTE ==========" << endl;
            trainData = SMOTEGenerator::applySMOTE(trainData);
        }
        
        // Cross-validation if requested
        if (config.validationFolds > 0) {
            if (config.numTrees > 1) {
                auto buildRF = [&config](const vector<DataPoint>& data) {
                    RandomForest rf(config);
                    rf.train(data);
                    return rf;
                };
                CrossValidator::validate(trainData, config, buildRF, config.validationFolds);
            } else {
                auto buildDT = [&config](const vector<DataPoint>& data) {
                    DecisionTree dt(config);
                    dt.train(data);
                    return dt;
                };
                CrossValidator::validate(trainData, config, buildDT, config.validationFolds);
            }
        }
        
        cout << "\n========== Training Final Model ==========" << endl;
        
        // Clean JSON directory first
        system("rm -f JSON/tree_*.json 2>/dev/null || del /f /q JSON\\tree_*.json 2>nul");

        if (config.numTrees > 1) {
            // Random Forest
            RandomForest forest(config);
            forest.train(trainData);
            
            cout << "\n========== Evaluation on Test Set ==========" << endl;
            Metrics testMetrics = MetricsCalculator::evaluate(forest, testData);
            MetricsCalculator::printMetrics(testMetrics, "Random Forest Test Results");

            cout << "\n========== Saving Trees to JSON ==========" << endl;
            forest.saveAllTrees("JSON/tree_");
            cout << "All trees saved to JSON/ directory" << endl;
            
            // Save predictions
            ofstream predFile("../data/diabetes_predictions.csv");
            predFile << "Prediction\n";
            for (const auto& point : testData) {
                predFile << forest.predict(point) << "\n";
            }
            predFile.close();
            cout << "Predictions saved to diabetes_predictions.csv" << endl;
            
        } else {
            // Decision Tree
            DecisionTree tree(config);
            tree.train(trainData);
            
            cout << "\n========== Evaluation on Test Set ==========" << endl;
            Metrics testMetrics = MetricsCalculator::evaluate(tree, testData);
            MetricsCalculator::printMetrics(testMetrics, "Decision Tree Test Results");
            
            cout << "\n========== Saving Tree to JSON ==========" << endl;
            tree.saveToJSON("JSON/tree_1.json");
            cout << "Decision tree saved to JSON/decision_tree.json" << endl;
            
            // Save predictions
            ofstream predFile("../data/diabetes_predictions.csv");
            predFile << "Prediction\n";
            for (const auto& point : testData) {
                predFile << tree.predict(point) << "\n";
            }
            predFile.close();
            cout << "Predictions saved to diabetes_predictions.csv" << endl;
        }
        
        cout << "\n|=============================================================|" << endl;
        cout << "|              Processing Completed Successfully!             |" << endl;
        cout << "|=============================================================|" << endl;
        
        cout << "\nPress Enter to exit...";
        cin.ignore();
        cin.get();
        
    } catch (const exception& e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}