# Neural_Network_Charity_Analysis

## Background

Beks is a data scientist who has been working with machine learning and neural networks for five years. She recently completed a boot camp to enhance her skills and is now ready to use her knowledge to help a foundation predict where to make investments. She has been provided with a dataset from Alphabet Soup's business team, which contains information on more than 34,000 applicants. Beks's goal is to use the features in this dataset to create a binary classifier that can predict whether applicants will be successful if Alphabet Soup funds them. This classifier will be an essential tool for the foundation to make informed decisions about where to allocate resources and support to maximize the impact of their investments.

## Results
### Data Preprocessing

We first preprocess our data set charity_data.csv by reading our data and noting the following target, feature, and identification variables:
 - Target Variable: IS_SUCCESSFUL
 - Feature Variables: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
 - Identification Variables (to be removed): EIN, NAME
 
 We then encode categorical variables using sklearn.preprocessing.OneHotEncoder after bucketing noisy features APPLICATION_TYPE and CLASSIFICATION with many unique values. After one hot encoding, we split our data into the target and features, split further into training and testing sets, and scaled our training and testing data using sklearn.preprocessing.StandardScaler.
 
 #### Compiling, trianing & Evaluating the model
 
 With our data preprocessed, we build the base model defined using tensorflow.keras.models.Sequential and tensorflow.keras.layers.Dense with the following parameters:
 - Number of hidden layers: 2 (Deep Neural Learning)
 - Architecture (hidden_nodes1, hidden_nodes2) : (80,30) neurons
 - Hidden Layer Activation Function: relu
 - Number of ouput nodes: 1
 - Output layer Activation Function: Sigmoid Function

We then compile and train the model using the binary_crossentropy loss function, adam optimizer, and accuracy metric to obtain the training results. Verifying with the testing set, we obtain the following results:
 - Loss: 0.554
 - Accuracy: 0.728
 
 We next optimize the previous model by adjusting the parameters shown above and more in the optimized model initially making the following single changes:
Reduction in input Variables: Dropped the variable "Special considerations"
Increased the number of hidden layers to 3: (80, 50, 30)
Changed the Activation fumction: relu to tanh
We do not see a significant increase in performance from the initial model and do not meet the target 75% accuracy criteria.
 
 ### Summary
 
 Additionally, one could iteratively tune the parameters above and keep optimal values when moving to the following parameters instead of reverting to the base setting and combining after completion. This requires careful consideration of how one adjusts parameters to arrive at an optimized model.

An alternative to this project's deep learning classification model could be a more traditional Random Forest Classifier. This model is also appropriate for this binary classification problem and can often perform comparably to deep learning models with just two hidden layers. It is also advantageous because there are fewer parameters to optimize, and those requiring attention are more intuitive than those in a neural network.
