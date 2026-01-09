# aimlmid2026_t_rokhvadze25

Assignment 1 – Finding Pearson’s Correlation Coefficient

The data points were collected manually from the interactive graph available at max.ge/aiml_midterm/69428_html by hovering over each blue dot and recording the displayed (x,y)coordinates. I wrote the (x,y)coordinates into a CSV file from which the Python code will read them.

To measure the relationship between the two variables, Pearson’s correlation coefficient was used. Pearson’s correlation coefficient quantifies the strength and direction of a linear relationship between two numerical variables.

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

