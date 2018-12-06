# Classification Modeling : Logistic Regression, RandomForest, Deep Neural Networks Using Keras and Tensorflow as backend.

The requirement is to analyze a confidential data set with 160,000 records and 50 columns/features to predict y - which takes on the value 0 (no) or 1 (yes). There was no information provided on the dataset on what each variables refers to or how the features are related to the outcome etc. Column names were provided as x0,x1,x2....x49 and atarget variable y (0/1).As the report was created, the focus was on creating a model that would be highly predictive of y (has the greatest accuracy), while providing interpretation where possible. Refer to python notebook for full commentary and detail explanation of the models used and their respective results.


Model Details
	
Logistic Regression

The team started the modeling process by using a Logistic Regression algorithm. The results from this model were not very good with default parameters. Then, a grid search method was performed to obtain the best Logistic Regression model parameters (c = 0.1 and penalty = l2). The grid search took around 9 mins to complete. Based on these best grid search parameters, the trained Logistic Regression model had an accuracy of 70% on the 10% test data split, which seemed low; therefore, the team decided to try other algorithms.

Random Forest

The second algorithm the team chose was a Random Forest Classifier. To optimize this model, a randomized search on various parameters were executed. The metric used to perform the randomized search was ‘accuracy’. The model was then trained using the best parameters generated from the randomized search. This model showed significant improvement on the predictive accuracy over the Logistic Regression model. The prediction accuracy on this Random Forest model was 91.53% on the 10% test data split. The important parameters used in this model was ( criterion = 'entropy' and n_estimators=10 ).

Naïve Bayes

For this Naive Bayes model, the algorithm used was Gaussian Naive Bayes algorithm for classification. The prediction accuracy on this Naive Bayes model was 69.21% on the 10% test data split. Since it was lower than the previous two models (Logistic Regression and Random Forest) the team decided to move on to test a different algorithm as the accuracy was not as high.

Boosted Trees

In Boosted Trees modeling, the model was trained using Gradient Boosting for classification. Important features used in the model were ( criterion='friedman_mse',
learning_rate=0.1, loss='deviance' ). The best accuracy score obtained for this Boosted Trees model was 83.33 % on the 10% test data split which was better than Naive Bayes and Logistic Regression models but significantly less than the Random Forest model. Hence the team decided not to further tune this model and proceed with advance deep learning techniques.


Deep Neural Network Model (Using Keras)

As Deep Learning is considered as one of the best approaches in tackling classification problems in today’s world, our team decided to build a deep neural net model for this project. Keras with Tensorflow as the backend was used to build the Neural Network architecture. The neural network designed by the team has 3 hidden layers plus input / output layers.  Optimizer used in this model is ‘Nadam’ with a learning rate of 0.001. Loss function used is ‘binary_crossentropy’ and metrics for prediction is ‘accuracy’. The kernel initializer used is ‘RandomNormal’ and the activation function used is PReLU which is an advanced activation function. While training the neural network model the team encountered the issue of overfitting e.g. a training accuracy of 99.89 was obtained while training the model and the prediction accuracy on the test set was 96.8. Hence the the dropout value was increased from .1 to .3 and retrained on multiple architectures. On increasing the dropout value the overfitting issue was contained. Based on a number of trials, the optimal batch size observed was 250 and the optimal number of epochs was found to be 250. The architecture has 230 neurons in each of its layers. The training time for this final neural network model is 35 mins on a CPU. The final prediction accuracy on this model was 97.96% and is the top classifier model for this project. The architecture chosen for the deep neural network is shown below.


Interpretation: Deep Neural Network Model

Deep Neural Networks are often known as being black boxes where inputs are fed in and predictions are mysteriously given as output. To provide insight to the business partner, we utilized a model explainer Python package (called “Skater”) to plot the relative column/feature importance for the deep neural network model. Then, to go further, we created partial dependence plots to visualize the effect of each top influential feature on the model’s predictions of y. 
First, in the appendix, a plot of the “Deep Neural Network Feature Importance” is provided. This plot provides relative feature importance, where higher importance means the variable had more impact to predict y. The top features with the highest relative feature importance, in decreasing order, were: x37, x7, x28, x23, x40, x20, x49, x12, x46, x42, x41, x38, x48, x6, x27, and x2. 
The partial dependence plots (PDPs) may be useful to the business partner if he has some control to change the related feature values and if there is a marginal difference in profit for y = 0 vs. y = 1.  For the top two features, x37 and x7, the team created a partial dependence plot (PDP) with their combined impact to predict y. The resulting PDP is shown below. Below, a feature surface is plotted, with the color representing the combination of x37 and x7 values (shown with the “Gradient of PDP” legend). For the range of values for x37 and x7, these were the values within approximately 2 standard deviations from the mean value for each. Above and below this feature surface is the range of predicted variance (in light blue). Then, the y-axis (“0”), represents the probability that 0 is predicted by the model. Conversely, the probability 1 is predicted is 1 - the probability that 0 is predicted. Therefore, if the probability of predicting y = 0 is 0.2 (20%), there is a 0.8 (80%) probability that y = 1 is predicted. Other PDPs for the top 16 features are provided in the appendix and can be interpreted similarly.

