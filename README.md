# T04_Cyberbullying_Detection

**Team Name**: The Predictive Squad

**Team Members**: Vedant Dave (202418014), Kashish Patel (202418044), Sujal Dhrangdhariya (202418017), Jatin Sindhi (202418055)

## Documentation

## 1. Data Preparation :-
### Loading Data :
**->** Read twitter data from CSV files got from DataSet Link given below. A complete description of dataset is given in the link below

Dataset Link :- https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

<hr>

## 2. Data Preprocessing :-
**->** Converted the tweet text to lower case.

**->** Removed all the noise (punctuations, numbers, special characters, hyperlinks, urls, emojis) from the tweet text.

**->** Then the text is lemmantized and only those words were kept in text which were english.

**->** A function is defined to get top unigrams and bigrams of the text according to cyberbullying type.


<hr>

## 3. Visualization :-
1.  A bar chart is plotted to show the top 10 unigrams and bigrams of each cyberbullying type.
2.  A word cloud is plotted to show the top word used for each cyberbullying type.


<hr>

## 4. Training / Testing / Splitting & Define Basic How Model Work :-
1. Splitting Our data in 80:10:10 (Train Data : Validation Data : Test Data)
2. Label encoding all the 6 classes present in our target variable.
3. Term Frequency-Inverse Document Frequency Vectorizer with maximum 1000 features.
4. Make A pipeline.
5. Use GridSearch Cross Validation for feature selections.
6. Train 3 different models and fit on training data.
7. After model run over test data and check accuracy score based on different matrics.

<hr>

## 5. Model Training & Psuedocode :-

**1. Naive bayes model :-**

  **Psuedocode:-**

Input: X (features), Y (labels), α (smoothing)
Train:
  - Compute P(Y=c) for each class c
  - For each feature f, class c, estimate P(X_f | Y=c) with smoothing
Predict:
  - For each class c, compute P(Y=c | X) = P(Y=c) * Π P(X_f | Y=c)
  - Return class with highest posterior



**2. SVM Model :-**
  
  **Psuedocode:-**

Input: X (features), Y (labels), C (regularization), K (kernel)
Train:
  - Initialize α_i = 0
  - While not converged:
    - For each i, compute decision function f(X_i) = Σ α_j Y_j K(X_j, X_i) + b
    - Update α_i and α_j using optimization
  - Compute bias b from support vectors
Predict:
  - For X, compute f(X) = Σ α_j Y_j K(X_j, X) + b
  - Return sign(f(X))



**3. Gradient Boosting Model (Ensamble Method) :-**

  **Psuedocode:-**
  
Input: X (features), Y (labels), T (iterations), η (learning rate)
Train:
  - Initialize F_0(X) = log(P(Y=1) / P(Y=0))
  - For t = 1 to T:
    - Compute residuals r_i = Y_i - P(Y_i | X_i)
    - Fit weak learner h_t(X) to residuals
    - Update model F_{t+1}(X) = F_t(X) + η h_t(X)
Predict:
  - Compute F_T(X) = F_0(X) + Σ η h_t(X)
  - Return 1 if P(Y=1 | X) > 0.5, else 0

<hr>

## 6. Contributions / Novelty :-
**->**  This project addresses cyberbullying detection on Twitter using a multi-model approach, applying Naive Bayes, Support Vector Machines (SVM), and Gradient Boosting ensemble methods. After extensive preprocessing, including cleaning, tokenization, and transforming tweets using TF-IDF for text representation. These features help in identifying potential patterns of abusive behavior.

**->** Naive Bayes was used for its computational efficiency and simplicity, while SVM was chosen for its strong performance with high-dimensional data. Gradient Boosting, an ensemble method, was implemented to combine weak learners into a more powerful model, improving overall prediction accuracy. Each model was rigorously evaluated using metrics such as accuracy, precision, recall, and F1-score.

**->** The novelty of this approach lies in the combination of traditional text classification models with advanced ensemble methods, along with the incorporation of social behavior features that go beyond simple text analysis. By comparing the models, we identified the strengths and limitations of each in the context of cyberbullying detection. This comprehensive, multi-dimensional approach offers a more accurate and nuanced understanding of online bullying behaviors, capturing subtle, complex forms of aggression like sarcasm or indirect insults.

<hr>

## 7. Citations :-

* For Dataset :- https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

* For Data Visulizations :-  https://matplotlib.org/ , https://seaborn.pydata.org,  https://plotly.com/


* For Dataset Training/Testing/Splitting :-  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

* For GridSearch Feature Selections :-   https://scikit-learn.org/stable/modules/grid_search.html

* For Naive bayes Model :-  https://www.datacamp.com/tutorial/naive-bayes-scikit-learn

* For SVM Model :-  https://medium.com/@57fdaditya/the-complete-guide-to-support-vector-machines-svms-with-intuition-78697c347200

* For Gradient Boosting Model :- https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm
