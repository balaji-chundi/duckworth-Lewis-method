##Importing libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize 
import math
import matplotlib.pyplot as plt


def preprocessing(data):
    '''
    Extracts the relevant columns for our task from the dataframe and return them.
    All the preprocessing steps are included in this function itself.
    '''
    #step 0:- drop innings 2 data points
    data.drop(data.index[data['Innings'] == 2], inplace=True)

    #step 1:- dropping interrupted matches
    # matchData = data.groupby('Match')
    # interruptedMatchIDs = []
    # for matchID, group in matchData:
    #     lastRow = group.tail(1)
    #     if(int(lastRow['Wickets.in.Hand'].iloc[0:1]) != 0 and int(lastRow['Over'].iloc[0:1] != 50)):
    #         interruptedMatchIDs.append(matchID)
    # #print(interruptedMatchIDs)
    # for item in interruptedMatchIDs:
    #     data.drop(data.index[data['Match'] == item], inplace=True)


    #step 1:- oversRemaining will ranging from 0-49, But there should be atleast a row which corressponds to oversRemaining '50'
    matchData_1 = data.groupby('Over')

    for over, group in matchData_1:
        if(over==49):
            dataNeedsAppending = group
            #changing over columns to 50
            dataNeedsAppending['Over'] = dataNeedsAppending['Over'].replace([49], 50)
    
    data = pd.concat([data, dataNeedsAppending])

    #step 2:- dropping errorenous data
    data.drop(data.index[data['Error.In.Data'] == 1], inplace=True)

    #step 3:- removing oversRemaining = 0 data points
    oversRemaining    = data['Total.Overs'].values-data['Over'].values
    data['oversRemaining'] = pd.Series(oversRemaining)
    data.drop(data.index[data['oversRemaining'] == 0], inplace=True)

    #step 4:- removing datapoints specifying 0 wickets left.
    data.drop(data.index[data['Wickets.in.Hand'] == 0], inplace=True)

    runsRemaining    = data['Innings.Total.Runs'].values - data['Total.Runs'].values
    # oversRemaining    = data['oversRemaining'].values
    oversRemaining    = data['Total.Overs'].values-data['Over'].values
    wicketsInHand    = data['Wickets.in.Hand'].values
    
    return runsRemaining, oversRemaining, wicketsInHand


def lossFunction(params, args):
    '''
    This Funtion calculates the total normalized sum of squared error loss over all data points which are of innings 1.
    input : 'params' - List of 11 elements (Z1, Z2, Z3,..., Z10, L)
    args : List with 4 elements [innings, runs, oversRemaining, wicketsInHand]
    
    returns : total normalized squared error of the function.
    '''
    error = 0
    L = params[10]
    runs = args[0]
    oversRemaining = args[1]
    wicketsInHand = args[2]
    for i in range(len(wicketsInHand)):
        runsMadeTillNow = runs[i]
        oversRemainingNow = oversRemaining[i]
        wicketsInHandNow = wicketsInHand[i]
        Z_ = params[wicketsInHandNow - 1]
        if runsMadeTillNow > 0:
            runsPredicted =  Z_ * (1 - np.exp(-1*L * oversRemainingNow / Z_))
            error = error + (math.pow(runsPredicted - runsMadeTillNow, 2))
    return error / len(wicketsInHand)

def Minimize(runs, oversRemaining, wicketsInHand, Method):
    '''
    This function helps in minimizing the given scalar loss function against all the 11 parameters using different methods of scipy.optimize.minimize like 'CG', 'BFGS',
    'L-BFGS-B' and 'SLSQP'
    Input : innings - innings number of that data point
            runs - runs scored till that point in the game
            oversRemaining - Number of overs remaining at that particular point in the game
            wicketsInHand - Number of wickets that the batting side still has
            Method - List of methods in scipy.optimize.minimize
    
    Output : output - is a list consisting of all the parameters and the corresponding total normalized loss for all the methods specified in input 'Method' 
    '''
    output = []
    for i in range(len(Method)):
        #initializing parameters close to mean
        dataframe = pd.DataFrame()
        dataframe['runs'] = pd.Series(runs)
        dataframe['overRemaining'] = pd.Series(oversRemaining)
        dataframe['wickets'] = pd.Series(wicketsInHand)

        mean_values = dataframe.groupby('wickets').mean()
        means = list(mean_values['runs'])
        means.append(10)
        parameters = means
        # print(parameters)

        #parameters are initialized to some arbitrary values(initial guess)
        parameters = [10, 30, 40, 65, 85, 130, 155, 170, 185, 200, 10]

        result = minimize(lossFunction, parameters, args=[runs, oversRemaining, wicketsInHand], method=Method[i])
        output.append(result)

        resourceVSovers(result['x'], Method[i])

    return output

def resourceVSovers(params, Method):
    '''
        This function helps in plotting the graph between 'Percentage of Resources Remaining' and 'Overs Remaining'
    '''
    plt.figure(1)
    plt.title("Percentage of Resources Remaining vs Overs Used")
    plt.xlim((0, 50))
    plt.ylim((0, 100))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Overs Remaining')
    plt.ylabel('Percentage Of Resources Remaining')
    color = ['b', 'c', 'g', 'm', 'r', 'y', 'k', 'lime', 'teal', 'seagreen']
    x = np.zeros((51))
    for i in range(51):
        x[i] = i
    L = params[10]
    Z = params[9] * (1 - np.exp(-L * 50 /params[9]))
    for i in range(len(params)-1):
        y = params[i] * (1 - np.exp(-L * x /params[i]))
        plt.plot(x, (y / Z) * 100, c = color[i], label='Z' + str(i + 1))
        plt.legend()
    plt.savefig(f'resourceVSovers_{Method}.png')
    #plt.show()
    plt.close()

    return

if __name__ == "__main__":

    data = pd.read_csv('./data/04_cricket_1999to2011.csv')
    #some methods of scipy.optimize.minimize
    # Method = ['CG', 'BFGS', 'L-BFGS-B', 'SLSQP'] 
    Method = ['L-BFGS-B', 'SLSQP']
    runs, oversRemaining, wicketsInHand = preprocessing(data)
    
    #optimizing using scipy.optimize.minimize
    output = Minimize(runs, oversRemaining, wicketsInHand, Method)
    print(output)