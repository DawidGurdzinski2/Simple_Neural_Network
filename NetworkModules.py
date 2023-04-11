import numpy as np



### Layers module ###

class Linear():
    def __init__(self,inputs,outputs):
        self.weights=0.01*np.random.randn(outputs,inputs)
        self.biases=np.zeros((1,outputs))

    def forward(self,inputs):
        self.inputs=inputs
        self.outputs=np.dot(inputs,self.weights.T)+self.biases
        return self.outputs
    
    def backward(self,dnextinputs):
        self.dweights=np.dot(dnextinputs.T,self.inputs)
        self.dbiases=np.sum(dnextinputs,axis=0,keepdims=True)
        self.dinputs=np.dot(dnextinputs,self.weights)
        return self.dinputs
    
### Activation functions ###

class Relu():
    def __init__(self) -> None:
        pass
    
    def forward(self,inputs):
        self.inputs=inputs
        self.outputs=np.maximum(0,inputs)
        return self.outputs


    def backward(self,dnextinputs):
        self.dinputs=dnextinputs
        self.dinputs[self.inputs<=0]=0
        return self.dinputs


class Softmax():
    def __init__(self) -> None:
        pass

    def forward(self,inputs):
        self.inputs=inputs
        exp=np.exp(inputs -np.max(inputs, axis=1,keepdims=True))
        exp_sum=np.sum(exp,axis=1,keepdims=True)
        self.outputs=exp/exp_sum
        return self.outputs

    def backwards(self,dnextinputs):
        self.dinputs = np.zeros_like(dnextinputs)
        i=0
        for output, dnextinput in zip(self.outputs, dnextinputs):
            output = output.reshape(-1, 1)
            Jacobian= np.diagflat(output) - np.dot(output,output.T)
            self.dinputs[i] = np.dot(Jacobian, dnextinput)
            i+=1
        return self.dinputs
    

### Loss Functions ###
class MSELoss():
    def __init__(self) -> None:
        pass
    
    def forward(self,inputs,labels):
        self.difference=inputs-labels
        self.output=np.mean((inputs-labels)**2)
        return self.output
    
    def backward(self):
        self.dinputs=(2*self.difference)/self.difference.size
        return self.dinputs
        

class CrossEntropyLoss():
    def __init__(self) -> None:
        pass
        
    def forward(self,inputs,labels):
        self.labels=labels
        self.inputs=inputs
        epsilon=1e-7
        inputs=np.clip(inputs,epsilon,1-epsilon)
        one_hot=np.eye(len(inputs[0]))[labels]
        self.output=-np.sum(one_hot*np.log(inputs))/one_hot.shape[0]
        return self.output
    
    def backward(self):
        one_hot=np.eye(len(self.inputs[0]))[self.labels]
        self.dinputs=(-one_hot/self.inputs)/one_hot.shape[0]
        return self.dinputs
### Optimaizers ###