# Midterm

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





**Assignment 2 – Spam Email Detection**


1. Training Dataset

The dataset used in this assignment is available in the GitHub repository at the following link:

https://github.com/trokhvadze25/aimlmid2026_t_rokhvadze25/blob/e57e2f8b9f629218ef9ad130355b16fc4c4a00f2/Task%202/data/t_rokhvadze25_69428.csv

The dataset contains numerical email features along with a class label indicating whether an email is spam (1) or legitimate (0).

The features included in the dataset are:

words – total number of words in the email

links – number of hyperlinks

capital_words – number of fully capitalized words

spam_word_count – number of common spam-related words

is_spam – target label (spam or legitimate)

Uploading the dataset to the repository ensures reproducibility and transparency of the experiment.





























