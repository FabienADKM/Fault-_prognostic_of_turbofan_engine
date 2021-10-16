import numpy as np
import matplotlib.pyplot as plt



def myPCA(Data,nbr_axes,verbose=0):
    
    """
    Return a matrix of dimension time*nbr_axes where each line is the 
    the projection of Data on principales axes. Returned matrix is 
    normalized (reduction and centering).

    *Data : Data corresponding to a single reactor 
    *nbr_axes : Number of principales axes to keep
    *verbose : Display informations if verbose=1. (Default 0).
    """

    X = np.copy(Data[:,1:])
   
    #Centering the data
    N = np.shape(X)[0]
    W = (1/N)*np.eye(N)
    mu = np.mean(X,axis = 0)
    var = np.var(X,axis = 0)
    
    X_centered = X - np.dot(np.ones((N,1)),np.reshape(mu,(np.shape(mu)[0],1)).T)
    Z = np.dot(X_centered,np.diag(np.sqrt(1/var)))
    R = np.dot(np.dot(Z.T,W),Z)
    
    #Save correlation matrix to a .csv file for deeper analysis
    np.savetxt("corr_matrix.csv", R, delimiter=",")
    
    Ig = np.trace(R)
    w,v = np.linalg.eig(R)
    index_sort = np.argsort(w)[::-1]
    w = np.sort(w)[::-1]
    
    #Plot eigenvalue diagramme if verbose==1.
    if (verbose==1):
        plt.figure()
        plt.bar(np.arange(len(w)),w)
        plt.title('Diagramme des valeurs propres')
        plt.show()
    
    #Compute inertie of each axes and plot it if verbose==1.
    iner = 0
    j = 0
    for j in range(nbr_axes):
        iner += w[j]/Ig
        j += 1
    
    if(verbose == 1):
        print("Inertie des ",nbr_axes," premiers axes conservés = ",iner,"%")
    
    #Projection on the nbr_axes first  principales axes 
    out  = np.dot(Z,v[:,:j])
    out = np.concatenate((Data[:,0].reshape(np.shape(Data[:,0])[0],1),out),axis =1) 
    
    return out 
