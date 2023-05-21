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

    def backward(self,dnextinputs):
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




class  Optimizer_Adam():
    def __init__ (self, learning_rate = 0.001,decay =0.,epsilon =1e-7 ,beta_1 =0.9 ,beta_2 =0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon  = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def start_update_params(self ):
        if  self.decay:
            self.current_learning_rate = self.learning_rate* ( 1./(1.+ self.decay*self.iterations))
  

    def update_params (self , layer ):
        if not hasattr (layer, 'weight_second_momentums' ):
            layer.weight_first_momentums = np.zeros_like(layer.weights)
            layer.weight_second_momentums = np.zeros_like(layer.weights)
            layer.bias_first_momentums = np.zeros_like(layer.biases)
            layer.bias_second_momentums = np.zeros_like(layer.biases)


        layer.weight_first_momentums = self.beta_1 *layer.weight_first_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_first_momentums = self.beta_1 * layer.bias_first_momentums  +(1 - self.beta_1) * layer.dbiases
        weight_first_momentums_corrected = layer.weight_first_momentums  /  (1 -  self.beta_1  ** (self.iterations  + 1))
        bias_first_momentums_corrected = layer.bias_first_momentums  / (1 - self.beta_1 ** (self.iterations  +  1))
        layer.weight_second_momentums = self.beta_2 * layer.weight_second_momentums  + (1 - self.beta_2) * layer.dweights ** 2 
        layer.bias_second_momentums = self.beta_2 * layer.bias_second_momentums  + (1 - self.beta_2) * layer.dbiases **2
        weight_second_momentums_corrected = layer.weight_second_momentums  / (1- self.beta_2 ** (self.iterations + 1))
        bias_second_momentums_corrected = layer.bias_second_momentums  / (1-self.beta_2 ** (self.iterations + 1))
        layer.weights += -self.current_learning_rate * weight_first_momentums_corrected /(np.sqrt(weight_second_momentums_corrected)+ self.epsilon)
        layer.biases += -self.current_learning_rate * bias_first_momentums_corrected /(np.sqrt(bias_second_momentums_corrected)+ self.epsilon)


    def stop_update_params( self ):
        self.iterations  +=  1





















