import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import KFold
import pdb

def customLoss(y_true,y_pred):
    '''
	The loss function for this algortihm is asymmetric around the true
	time of failure such that late predictions is more heavily
	penalized than early predictions.
    The shape of the final loss is (batch_size,).
    '''
    
    loss   = y_pred - y_true
    N = loss.shape[0]
    
    a1 = 10
    a2 = 13
    
    loss_p = loss[loss >= 0]
    loss_n = loss[loss < 0]
    
    loss_p = K.exp(loss_p/a2)  - 1
    loss_n = K.exp(-loss_n/a1) - 1
    
    mean_p = K.mean(loss_p)
    mean_n = K.mean(loss_n)
    
    print("loss_n",loss_n)
    print("loss_n.mean",mean_n)

    return 0.5*( mean_p + mean_n )




def buildModel(model,num_hidden_layers,architecture,act_func,n_input,output_class=1,optimizers = 'adam',loss_func = 'mse'):

        """
        Build a densely connected neural network model with user input parameters
        num_hidden_layers : Number of hidden layers 
        *architecture : List containing the number of unit for each layers 
        *act_func :Activation function. 'relu', 'sigmoid', 'tanh',...
        *in_shape : Dimension of the input vector
        *optimizers : SGD, RMSprop, Adam,...
        *loss_func : mse,mae,msle,... or custom loss function
        *output_class : Number of classes in the ouput vector
        """
        #First hidden layer
        model.add(Dense(architecture[0],
                                          activation=act_func,
                                          input_shape= (n_input,),
                                          kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01),
                                          bias_regularizer=regularizers.l2(0.01),
                                          activity_regularizer=regularizers.l2(0.01))  )

        #Hidden Layers
        for i in range(num_hidden_layers):
            model.add(Dense(architecture[i], activation=act_func,
                            kernel_regularizer= regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.l2(1e-4),
                            activity_regularizer=regularizers.l2(1e-5)))

        #Output Layer
        model.add(Dense(output_class))
        model.compile(optimizer= optimizers, loss = loss_func)

def trainModel(model,X_train,y_train,X_val,y_val,successive_fit_numb,epochs_array,batch_array,trainModel_history):
    """
    Train the model in regards to the user input
    At the ends of the training, the best model in regards to monitor function is saved in "best.h5"
    and the model at the end of training is save in "model.h5"
    successive_fit_numb : Number of successive fit during the training
    epochs_array : list of integer representing the number of epochs by fit. Size of epochs must be equal to successive_fit_num
    batch_array : list of integer representing the number of batch by fit. Size of batch must be equal to successive_fit_num
    train_data :
    train_targets :
    test_data :
    test_targets :
    weights : array of numbers that specify how much weight each sample in a batch should have in computing the total loss
     """
    verb = 1 
    #A simpler check-point strategy is to save the model weights to the same file, if and only if the validation accuracy improves
    #The best model is saved in file "bestb.h5"

    checkpoint = ModelCheckpoint( monitor='val_loss', filepath='weights.best.hdf5', save_best_only=True,verbose=verb)
    earlystop = EarlyStopping( monitor="val_loss",min_delta= 0.01,patience=40,verbose=2,mode="min",baseline=None,restore_best_weights=True)
    callbacks_list = [checkpoint,earlystop]

    print("----------Start of model Training-----------------------")
    for i in range(successive_fit_numb):
        seqM = model.fit(     X_train, y_train,
                              epochs = epochs_array[i] ,
                              batch_size = batch_array[i],
                              validation_data = (X_val, y_val),
                              verbose = verb,
                              callbacks = callbacks_list,
                              shuffle = True)
                              

        trainModel_history.append(seqM)
        print("----------End of model Training-------------------------")

        print("----------Loading best weights--------------------------")
        model.load_weights("weights.best.hdf5")

        print("----------Saving model as 'last _model.h5'--------------")
        model.save('last_model.h5', include_optimizer = True)


def CrossValidation(n_split,model,X,Y,epochs):
    mean_eval = []
    for train_index,test_index in KFold(n_split).split(X):
      X_train,X_eval=X[train_index],X[test_index]
      Y_train,Y_eval=Y[train_index],Y[test_index]
      
      model.fit(X_train, Y_train,epochs)
      
      mean_eval.append(model.evaluate(X_eval,Y_eval))
     
    return np.mean(mean_eval)


    

def plotTrainningHistory(successive_fit_numb,trainModel_history):

    """
    Allows to visualize model training history
    This function produce severals charts : a plot of loss and 'mea' on the training dataset over
    training epochs, for each successive fit.
    """

    col = successive_fit_numb
    row = 1 
    fig, axs = plt.subplots(row,col,figsize=(20,20))
    axs = axs.flatten()

    key = list(trainModel_history[0].history.keys())
    print("key ::",key)
    loss = key[0]
    val_loss = key[1]
    
    for i in range(successive_fit_numb):

        axs[i].plot(trainModel_history[i].history[loss])
        axs[i].plot(trainModel_history[i].history[val_loss])
        axs[i].set_title("Fit number " + str(i+1) + " of training" )
        axs[i].set_ylabel('loss')
        axs[i].set_xlabel('epoch')
        axs[i].legend(['train', 'eval'], loc='upper left')
        
    plt.savefig("Loss evolution during training.png") 
    plt.show()

