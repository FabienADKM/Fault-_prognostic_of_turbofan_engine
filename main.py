import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.backend import set_floatx
from pca import *
from NeuralNetwork import *
from scipy import stats

######################################
###         Program Parameters     ###
######################################

display_pca_result = 0
new_Network_Training = 1



###########################################################
#####              LOAD DATA                          #####
###########################################################

fname_train = "./CMAPSSData/Text Data/train_FD001.txt"
fname_xtest = "./CMAPSSData/Text Data/test_FD001.txt"
fname_ytest = "./CMAPSSData/Text Data/RUL_FD001.txt"
data = np.loadtxt(fname_train)
data_test = np.loadtxt(fname_xtest)
RUL_test = np.loadtxt(fname_ytest)

############################################################
###########     Extract table for each reactor   ###########
############################################################

nbr_reactor = int(data[-1,0])
nbr_reactor_test = int(data_test[-1,0])
Reactor = []
Reactor_test = []

#Extrac table for training Data
index_0 = 0
index_1 = 0

for reactor_index in range(1,nbr_reactor+1):
    while(data[index_1,0] == reactor_index):
        index_1 += 1
        if(index_1 == data.shape[0]):
            break

    Reactor.append(np.copy(data[index_0:index_1,1:]))
    index_0 = index_1


#Extract table for test data
index_0 = 0
index_1 = 0

for reactor_index in range(1,nbr_reactor_test+1):
    while(data_test[index_1,0] == reactor_index):
        index_1 += 1
        if(index_1 == data_test.shape[0]):
            break

    Reactor_test.append(np.copy(data_test[index_0:index_1,1:]))
    index_0 = index_1

##############################################################################
#####       Remove from  reactor variable with variance inferior to eps     ##
##############################################################################

eps = 1e-15
var = np.var(Reactor[0],axis = 0)
to_delete = []
for j in range(len(var)):
    if(var[j] <= eps):
        to_delete.append(j)

#Remove variable in training dataset
for i in range(nbr_reactor):
    Reactor[i] = np.delete(Reactor[i],to_delete,axis = 1)     

#Remove the same variable in test dataset
for i in range(nbr_reactor_test):
    Reactor_test[i] = np.delete(Reactor_test[i],to_delete,axis = 1) 




##############################################################################
##########              APPLY PCA                                   ##########
##############################################################################

Reactor_reduce = []
Reactor_test_reduce = []

for i in range(nbr_reactor):
    Reactor_reduce.append(myPCA(Reactor[i],8))

for i in range(nbr_reactor_test):
    Reactor_test_reduce.append(myPCA(Reactor_test[i],8))

if( display_pca_result == 1 ):

    nbr_of_composante_to_plot = 5
    reactor_to_plot = 0
    fig, axs = plt.subplots(nbr_of_composante_to_plot,1)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(nbr_of_composante_to_plot):
        axs[i].plot(Reactor_reduce[reactor_to_plot][:,0],Reactor_reduce[reactor_to_plot][:,i+1])
        axs[i].set_title("sensor "+str(i))
    
    plt.show()


##############################################################################
######      Now PCA has been applyed, prepare dataset for Neural Network   ###
##############################################################################

def split_sequences(reactor_list,windows_size):
    X,y = list(),list()
    for reactor in reactor_list:
        UL = reactor[-1,0] #Useful Life of the reactor
        end = reactor.shape[0]
        for i in range(end):
            end_ix = end - windows_size - i
            #check if we are beyond the dataset
            if(end_ix < 0):
                break
            X.append(reactor[end_ix:end-i,:])
            y.append(UL - reactor[end-i-1,0])  #y is the RUL of the reactor
    
    return np.array(X),np.array(y)

def split_sequences_test(reactor_test_reduce,RUL_test,window_size):
    X_test = []
    y_test = []
    for reactor_index in range(len(reactor_test_reduce)):
        r_x = []
        r_y = []
        RUL = RUL_test[reactor_index]
        end = reactor_test_reduce[reactor_index].shape[0]
        for i in range(end):
            end_ix = end -window_size - i
            #check if we are beyond teh dataset
            if(end_ix < 0):
                break
            r_x.append(reactor_test_reduce[reactor_index][end_ix:end-i,:])
            r_y.append(RUL+i)
        X_test.append(np.array(r_x[::-1]))
        y_test.append(np.array(r_y[::-1]))

    return X_test,y_test

windows_size = 15 
X, y = split_sequences(Reactor_reduce,windows_size)
X_test , y_test = split_sequences_test(Reactor_test_reduce,RUL_test,windows_size)

# flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

for i in range(len(X_test)):
    X_test[i] = X_test[i].reshape((X_test[i].shape[0], n_input))

##Split Dataset into train set and evaluation set
#First shuffle the dataset
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
y = y[s]

rate_dataTrain = 0.8
nbr_dataTrain = int(0.8*X.shape[0])

X_train = np.copy(X[:nbr_dataTrain,:])
y_train = np.copy(y[:nbr_dataTrain])
X_val  = np.copy(X[nbr_dataTrain:,:])
y_val  = np.copy(y[nbr_dataTrain:])

##############################################################################
########            NEURAL NETWORK                                  ##########
##############################################################################

model = Sequential()

#######   Parameters of the Network
# Probablement mettre ça dans un fichier .yml 
num_hidden_layers = 3 
architecture = [50,20,10]
epochs_array = [100,100,100]
batch_array = [128,128,128]
act_func = 'relu'
loss = 'mae'
successive_fit_number = 3
trainModel_history = []

#Force Keras to work with float64
set_floatx('float64')


if(new_Network_Training == 1):
    buildModel(model,num_hidden_layers,architecture,act_func,n_input,loss_func =loss )
    trainModel(model,X_train,y_train,X_val,y_val,successive_fit_number,epochs_array,batch_array,trainModel_history)
    mean = CrossValidation(5,model,X,y,epochs_array[0])
    plotTrainningHistory(successive_fit_number,trainModel_history)

else:
    if os.path.isfile('last_model.h5'):
        print("-------Loading model.h5------------")
        model = load_model('last_model.h5')
    else:
        print("Error loading file : last_model.h5 doesn't exist. Please check for it.")



###############################################################################
########                PREDICTION                              ###############
###############################################################################
num_reactor_topredict = 15 
y_pred = model.predict(X_test[num_reactor_topredict])
print("X_test[0].shape = ",X_test[0].shape)
print("y_pred.shape = ",y_pred.shape)
#Compute linear regression
t = np.arange(X_test[num_reactor_topredict].shape[0])
print("t_shape=",t.shape)
slope,intercept, r_value,p_value,std_err = stats.linregress(t,y_pred.reshape((y_pred.shape[0],)))

#Display result
plt.figure()
plt.plot(t,y_pred,c='r',label='Network prediction')
plt.plot(t,y_test[num_reactor_topredict],c='g',label='True RUL')
plt.plot(t,slope*t+intercept,c = 'b', label = 'Predicted RUL')
plt.xlabel('time')
plt.ylabel('RUL')
plt.legend()
plt.show()











