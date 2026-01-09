# Midterm

Assignment 1 – Finding Pearson’s Correlation Coefficient

The data points were collected manually from the interactive graph available at max.ge/aiml_midterm/69428_html by hovering over each blue dot and recording the displayed (x,y)coordinates. I wrote the (x,y)coordinates into a CSV file from which the Python code will read them.

To measure the relationship between the two variables, Pearson’s correlation coefficient was used. Pearson’s correlation coefficient quantifies the strength and direction of a linear relationship between two numerical variables.The solution is provided in Python code in the "Task 1 - https://github.com/trokhvadze25/aimlmid2026_t_rokhvadze25/blob/5f86f9ad86057235f75eb769d46c89821c42dd30/Task%201/Correlation.py " folder.

The calculation process is as follows:
	First, the mean value of all x values and the mean value of all y values are calculated.
	For each data point, the deviation of x from its mean and the deviation of y from its mean are computed.
	The covariance between xand y is obtained by summing the products of these deviations.
	This covariance is then normalized by dividing it by the product of the standard deviations of x and y.
	The resulting value is Pearson’s correlation coefficient r, which ranges between -1 and +1.
Using this method, the calculated Pearson correlation coefficient for the given data is: r≈0.9997

This value indicates a very strong positive linear correlation. As the value of x increases, the value of y also increases in an almost perfectly linear manner. This conclusion is further supported by the scatter plot, which shows the data points forming an almost straight line with a positive slope.

Visualisation :


![Scatter Plot](scatter_plot.png)



# aimlmid2026_t_rokhvadze25

**Assignment 1 – Finding Pearson’s Correlation Coefficient**

The data points were collected manually from the interactive graph available at max.ge/aiml_midterm/69428_html by hovering over each blue dot and recording the displayed (x,y)coordinates. I wrote the (x,y)coordinates into a CSV file from which the Python code will read them.

To measure the relationship between the two variables, Pearson’s correlation coefficient was used. Pearson’s correlation coefficient quantifies the strength and direction of a linear relationship between two numerical variables.The solution is provided in Python code in the "Task 1 - https://github.com/trokhvadze25/aimlmid2026_t_rokhvadze25/blob/5f86f9ad86057235f75eb769d46c89821c42dd30/Task%201/Correlation.py " folder.

The calculation process is as follows:
	First, the mean value of all x values and the mean value of all y values are calculated.
	For each data point, the deviation of x from its mean and the deviation of y from its mean are computed.
	The covariance between xand y is obtained by summing the products of these deviations.
	This covariance is then normalized by dividing it by the product of the standard deviations of x and y.
	The resulting value is Pearson’s correlation coefficient r, which ranges between -1 and +1.
Using this method, the calculated Pearson correlation coefficient for the given data is: r≈0.9997

This value indicates a very strong positive linear correlation. As the value of x increases, the value of y also increases in an almost perfectly linear manner. This conclusion is further supported by the scatter plot, which shows the data points forming an almost straight line with a positive slope.

Visualisation :


![Scatter Plot](scatter_plot.png)




**Assignment 2 - Spam email detection**

1) Training data link in the repo: https://github.com/trokhvadze25/aimlmid2026_t_rokhvadze25/blob/3cfa218417cb4fca2b173f44bbc43ab3266f9567/Task%202/t_rokhvadze25_69428.csv

The dataset contains numerical email features along with their corresponding class labels, indicating whether an email is spam or legitimate.


**Data Loading and Processing**
   
The dataset is loaded using the pandas library. After loading, the data is divided into two main parts:

Features (X): numerical attributes describing email characteristics

Target (y): the email class (spam or legitimate)

To evaluate the model fairly, the dataset is split into training and testing subsets.
Seventy percent (70%) of the data is used for training the model, while the remaining thirty percent (30%) is reserved for validation.
This approach allows the model to be evaluated on data it has not seen during training.

**code : **
data = pd.read_csv("data/t_rokhvadze25_69428.csv")

X = data.drop(columns=["class"])
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
"

**Logistic Regression Model**
Logistic Regression is used as the classification algorithm because it is well suited for binary classification problems, such as distinguishing between spam and legitimate emails.

The model learns a linear combination of the input features and applies a logistic (sigmoid) function to estimate the probability that an email belongs to the spam class.

"
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
"


**Model Coefficients**

After training, the coefficients of the Logistic Regression model are extracted. Each coefficient represents the influence of a specific feature on the probability of an email being classified as spam.

Positive coefficient: increases the likelihood of spam

Negative coefficient: decreases the likelihood of spam

"
for feature, coef in zip(X.columns, model.coef_[0]):
    print(feature, coef)

print("Intercept:", model.intercept_)
"
These coefficients provide insight into which features are most important for classification.


**Model Validation**
The trained model is validated using data that was not used for training. Two evaluation metrics are calculated:

Confusion Matrix: shows correct and incorrect classifications

Accuracy: measures the proportion of correctly classified emails

"
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
"

The confusion matrix includes:

True Positives (correct spam predictions)

True Negatives (correct legitimate predictions)

False Positives (legitimate emails classified as spam)

False Negatives (spam emails classified as legitimate)

Accuracy provides an overall measure of the model’s performance.


**Email Text Classification Capability**

   The application is capable of classifying new email text provided by the user.
The email text is first parsed, and the same features used in the dataset are extracted from the raw text.

"
def extract_features(text):
    text = text.lower()
    return [
        text.count("free"),
        text.count("win"),
        text.count("http"),
        len(text.split())
    ]
	"
	
These extracted features are then passed to the trained model to determine whether the email is spam or legitimate.

"
def classify_email(text):
    features = extract_features(text)
    prediction = model.predict([features])[0]
    return "SPAM" if prediction == 1 else "LEGITIMATE"
"

**Manually Composed Spam Email**

Subject: FREE MONEY – Act Now!

You won free money! Click the link below to claim your prize:
http://spam-link.example

This email was intentionally designed to resemble spam by including promotional language, urgency, and a suspicious hyperlink. These characteristics align with features commonly associated with spam emails in the dataset.


**Manually Composed Legitimate Email**


Subject: Meeting Reminder

Hello team,
This is a reminder that our meeting is scheduled for tomorrow at 10 AM.



**Visualization - Class Distribution**
A bar chart is generated to show the number of spam and legitimate emails in the dataset.
This visualization helps identify whether the dataset is balanced or imbalanced.

![Class Distribution](images/class_distribution.png)


The chart reveals the relative proportion of spam and legitimate emails, which is important for understanding how class imbalance may affect model performance.


**Visualization – Confusion Matrix Heatmap**

A heatmap visualization of the confusion matrix is generated to provide a clearer view of model performance.



![Confusion Matrix](images/confusion_matrix.png)

The heatmap visually highlights correct and incorrect classifications, making it easier to analyze which types of errors the model makes.



This assignment demonstrates the complete development of a Logistic Regression–based email spam classifier, including data preprocessing, model training, validation, email text classification, and visualization. All required components of the midterm task have been successfully implemented and documented.
