#Jehad Hamayel 1200348

import pandas as pand
import matplotlib.pyplot as plot
import numpy as np
#Read the data file
try:
    DataPath = 'C:\\Users\\hp\\PycharmProjects\\MLProject\\cars.csv'
    DataFrame = pand.read_csv(DataPath)
    print("Reading completed successfully")
except Exception as e:
    print("Error The reading was not successful", str(e))
#Print The number of features and examples
print("The number of features are:", len(DataFrame.columns))
print("The number of examples are:", len(DataFrame))
#Check for missing values and print how many missing values are found
MissingValues = DataFrame.isnull().sum()
if any(MissingValues > 0):
    print("Yes there are missing values in each feature:\n", MissingValues)
else:
    print("No there are not missing values in each feature")

#Filling in missing values using the median and mode
intFloatColumns = DataFrame.select_dtypes(include=['int', 'float']).columns
StringColumns = DataFrame.select_dtypes(include=['object']).columns
DataFrame[intFloatColumns] = DataFrame[intFloatColumns].fillna(DataFrame[intFloatColumns].median())
DataFrame[StringColumns] = DataFrame[StringColumns].fillna(DataFrame[StringColumns].mode().iloc[0])

#plot the box plot that shows the mpg for each country so we know which country produces cars with better fuel economy
DataFrame.boxplot(column='mpg', by='origin')
plot.title("Box Plot shows the mpg for each Country")
plot.xlabel('Countries')
plot.ylabel('mpg')
plot.show()
#Know the median mpg for each country to know which country is better at consuming fuel for cars
medianForMPGofTheCountries = DataFrame.groupby('origin')['mpg'].median()
bestFuelCountry = medianForMPGofTheCountries.idxmax()
bestFuelCountryMedian = medianForMPGofTheCountries.max()
print(f"The Better Fuel Economy in the {bestFuelCountry} country with median {bestFuelCountryMedian}")

#plot the histogram of each feature so that it is known which features has a distribution that is most similar to a Gaussian
DataFrame[['mpg', 'horsepower', 'acceleration']].hist()
plot.suptitle('Histograms for Features')
plot.show()

#Find out which features has a distribution that is most similar to a Gaussian by quantitatively measuring skewness
features = ['acceleration', 'horsepower', 'mpg']
S= 100
bestFeat=""
for feature in features:
    skewness_S = 3*((np.mean(DataFrame[feature])-np.median(DataFrame[feature]))/np.std(DataFrame[feature]))
    print(f'The skewness Value for the feature({feature}) = {skewness_S}')
    if abs(skewness_S) < abs(S):
        S = skewness_S
        bestFeat = feature
print(f'The data for {bestFeat} looks Gaussian distributed because:\nit is approximately symmetric with skewness = {S} ')

# Plot a scatter plot that shows the ‘horsepower’ on the x-axis and ‘mpg’ on the y-axis.
plot.scatter(DataFrame['horsepower'], DataFrame['mpg'])
plot.title('Scatter Plot Of Horsepower X MPG')
plot.xlabel('horsepower')
plot.ylabel('mpg')
plot.show()


X_Matrix = np.column_stack((np.ones(len(DataFrame)), DataFrame['horsepower']))
Y_Target = DataFrame['mpg'].values
#Finding the Parameters of the object functionn, then find the simple linear regression of horsepower
W_Parameters = np.linalg.inv(X_Matrix.T @ X_Matrix) @ X_Matrix.T @ Y_Target
print(f'W_Parameters by simple linear regression: {W_Parameters}')
plot.scatter(DataFrame['horsepower'], DataFrame['mpg'])
plot.plot(DataFrame['horsepower'], X_Matrix @ W_Parameters, color='yellow')
plot.title('Scatter Plot for Horsepower and MPG with linear regression')
plot.xlabel('Horsepower')
plot.ylabel('MPG')
plot.show()

#Finding the Parameters of the object functionn, then find the Quadratic linear regression of horsepower
X_MatrixOfQuadratic = np.column_stack((np.ones(len(DataFrame)), DataFrame['horsepower'], DataFrame['horsepower'] ** 2))
W_ParametersOfQuadratic = np.linalg.inv(X_MatrixOfQuadratic.T @ X_MatrixOfQuadratic) @ X_MatrixOfQuadratic.T @ Y_Target
x = DataFrame['horsepower'].sort_values()
QuadraticFun = W_ParametersOfQuadratic[0] + W_ParametersOfQuadratic[1]*x + W_ParametersOfQuadratic[2] * x **2
plot.scatter(DataFrame['horsepower'], DataFrame['mpg'])
plot.plot(x,QuadraticFun, color='green')
plot.title('Scatter Plot for Horsepower and MPG with quadratic function')
plot.xlabel('Horsepower')
plot.ylabel('MPG')
plot.show()

#Gradient Descent Function
def GradientDescent(X_Matrix, Y_Target, W_Parameters, learning_rate, iterations):
    n = float(len(X_Matrix))
    for i in range(iterations):
        gradientDescent = np.zeros(X_Matrix.shape[1])
        print(W_Parameters)
        for x in range(len(X_Matrix)):
              gradientDescent += ((X_Matrix[x] @ W_Parameters) - Y_Target[x]) * X_Matrix[x] * (2 / n)

        W_Parameters -=  (learning_rate * gradientDescent)

    return W_Parameters

#Preparing data for the partitions using linear regression using Gradient Descent
X_Matrix_GradientDescent = np.column_stack((np.ones(len(DataFrame)), DataFrame['horsepower']))
Y_Target_GradientDescent = DataFrame['mpg'].values
W_ParametersInitial = np.random.rand(X_Matrix_GradientDescent.shape[1])
learningRate = 0.05
iterations = 1500

standard=np.std(X_Matrix_GradientDescent)
mean=np.mean(X_Matrix_GradientDescent)
#Doing the scaling process
X_Matrix_GradientDescent = (X_Matrix_GradientDescent - np.mean(X_Matrix_GradientDescent)) / np.std(X_Matrix_GradientDescent)

W_Parameters_GradientDescent = GradientDescent(X_Matrix_GradientDescent, Y_Target_GradientDescent, W_ParametersInitial, learningRate, iterations)
print(W_Parameters_GradientDescent)

#Extraction parameters before scaling
W1=(W_Parameters_GradientDescent[1]/standard)
W0=(((W_Parameters_GradientDescent[0]/standard)-(W_Parameters_GradientDescent[0]*mean)/standard)-((W_Parameters_GradientDescent[1]*mean)/standard))
print(f'the Parameter:(W0={W0},W1={W1})')

plot.scatter(DataFrame['horsepower'], DataFrame['mpg'])
plot.plot(DataFrame['horsepower'], X_Matrix_GradientDescent @ W_Parameters_GradientDescent, color='red', linewidth=2)
plot.title('Scatter Plot for Horsepower and MPG with Linear Regression with Gradient Descent')
plot.xlabel('Horsepower')
plot.ylabel('MPG')
plot.show()

