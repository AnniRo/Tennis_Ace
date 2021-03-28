# import modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from itertools import combinations

# load and investigate the data
df = pd.read_csv('tennis_stats.csv')
print(df.head())

# perform exploratory analysis

# set up a list of colors for the graphs
colors=['r','g','b','k', 'c', 'm', 'y', 'lime', 'royalblue', 'darkorange', 'tomato', 'indigo', 'yellow', 'chocolate', 'darkred', 'crimson', 'teal', 'deeppink']

# iterate through the columns of features 
for i in range(2, len(df.columns[2:20])):
  # create a figure at each iteration  
  plt.figure()
  # create a scatter plot for each feature against the 'Winnings' outcome variable
  plt.scatter(df[[df.columns[i]]], df[['Winnings']], color=colors[i], alpha=0.4)
  plt.xlabel(df.columns[i], fontsize=16)
  plt.ylabel('Winnings')

  plt.show()

plt.clf()


# create list with features that seem to have linear relationship with the 'Winnings' variable
lst = ['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities' , 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed']

# build a linear regression function
def linear_regression(x, y, col_name_x):

    # split data into training and test datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=4)

    # initialize and train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # predict winnings
    y_predict = model.predict(x_test)

    # plot actual winnings vs predicted winnings
    plt.scatter(y_test, y_predict, alpha=0.4)
    plt.xlabel('Actual Winnings')
    plt.ylabel('Predicted Winnings')
    plt.title(f'Actual vs predicted winnings - {col_name_x}')
    plt.show()

    # evaluate model's accuracy on the test set
    print('Model\'s Test Accuracy:')
    return model.score(x_test, y_test)


# perform single feature linear regression
print(f'{lst[-1]} vs Winnings\n')
print(linear_regression(df[[lst[-1]]], df[['Winnings']], lst[-1]))
print('-' * 30)


# perform multiple feature linear regressions
for i in range(len(lst[:-1])):
    print(f'{lst[i]} vs Winnings\n')
    # create a figure at each iteration
    plt.figure()
    print(linear_regression(df[[lst[i]]], df[['Winnings']], lst[i]))
    print('-' * 30)

# The model that uses the feature 'ServiceGamesPlayed' to predict the outcome 'Winnings' is the best



######## create linear regression models that use two features to predict yearly earnings ########



# get all possible pairs from lst using combinations()
pairs = list(combinations(lst, 2))

# perform linear regression
for i in range(len(pairs)):
    print(f'{pairs[i][0]}, {pairs[i][1]} vs Winnings\n')
    # create a figure at each iteration
    plt.figure()
    print(linear_regression(df[[pairs[i][0], pairs[i][1]]], df[['Winnings']], pairs[i]))
    print('-' * 30)

# The set of features 'DoubleFaults' and 'ServiceGamesPlayed' results in the best model



######## create multiple linear regression models that use multiple features to predict yearly earnings ########



# get all possible 5-feature combinations from lst
feature_comb = list(combinations(lst, 5))

# perform linear regression
for i in range(len(feature_comb)):
    print(f'{feature_comb[i][0]}, {feature_comb[i][1]}, {feature_comb[i][2]}, {feature_comb[i][3]}, {feature_comb[i][4]} vs Winnings\n')
    # create a figure at each iteration
    plt.figure()
    print(linear_regression(df[[feature_comb[i][0], feature_comb[i][1], feature_comb[i][2], feature_comb[i][3], feature_comb[i][4]]], df[['Winnings']], feature_comb[i]))
    print('-' * 30)

# The best model results from the features 'Aces', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed' and 'ServiceGamesPlayed'