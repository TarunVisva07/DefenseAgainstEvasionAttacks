import torch, torchvision
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms,datasets
from glob import glob
import pickle
import platform
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import numpy as np

#glob module can be used for file name matching

class ConvNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ConvNet,self).__init__()

        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)

        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
        output=output.view(-1,32*75*75)

        output=self.fc(output)
            
        return output

    def initialise_parameters(self,result_parameters):

        #print('Initialize parameters called ...')
        for params in self.parameters():
            params.data = params.to(torch.device('mps'))
        with torch.no_grad():
            #Adding the obtained gradients to the model
            self.conv1.weight += torch.nn.Parameter(result_parameters['conv1.weight'].to(torch.device('mps')))
            self.conv1.bias += torch.nn.Parameter(result_parameters["conv1.bias"].to(torch.device('mps')))
            self.bn1.weight += torch.nn.Parameter(result_parameters["bn1.weight"].to(torch.device('mps')))
            self.bn1.bias += torch.nn.Parameter(result_parameters["bn1.bias"].to(torch.device('mps')))
            self.conv2.weight += torch.nn.Parameter(result_parameters["conv2.weight"].to(torch.device('mps')))
            self.conv2.bias += torch.nn.Parameter(result_parameters["conv2.bias"].to(torch.device('mps')))
            self.conv3.weight += torch.nn.Parameter(result_parameters["conv3.weight"].to(torch.device('mps')))
            self.conv3.bias += torch.nn.Parameter(result_parameters["conv3.bias"].to(torch.device('mps')))
            self.bn3.weight += torch.nn.Parameter(result_parameters["bn3.weight"].to(torch.device('mps')))
            self.bn3.bias += torch.nn.Parameter(result_parameters["bn3.bias"].to(torch.device('mps')))
            self.fc.weight += torch.nn.Parameter(result_parameters["fc.weight"].to(torch.device('mps')))
            self.fc.bias += torch.nn.Parameter(result_parameters["fc.bias"].to(torch.device('mps')))


class HWRModel:
    
    def __init__(self,data_path):
        self.train_path = data_path+'/Train'
        self.test_path = data_path +'/Test'
        self.model = ConvNet(num_classes = 2)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.CrossEntropyLoss()
        self.dest_file = 'best_checkpoint_server.model'
        self.batch_size = 20
        self.local_data_percentage = 70
        self.device = torch.device('mps')
        self.classifier = PyTorchClassifier(
            model = self.model,
            loss = self.loss_func,
            optimizer = self.optimizer,
            input_shape=(3,150,150),
            nb_classes=2,
            channels_first=False
        )

    def user_instance(self,user_id,batch_size,local_data_percentage):
        self.user_id = user_id
        self.dest_file = 'best_checkpoint_{}.model'.format(user_id)
        self.batch_size = batch_size
        self.local_data_percentage = local_data_percentage

    def initialise_parameters(self,result):
        self.model.initialise_parameters(result)
    
    def preprocess(self,resize=150):
        transformer = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ]
        )
        return transformer   


    def get_dataset(self,path):
        #Extracts a certain portion of dataset and returns
        img_dataset = datasets.ImageFolder(path,transform = self.preprocess())
        total_data_count = len(glob(path+"/**/*.png"))
        shared_data_count = int((self.local_data_percentage/100)*total_data_count)
        local_dataset,rem_dataset = random_split(img_dataset,(shared_data_count,total_data_count-shared_data_count))

        print("{}/{} images taken".format(shared_data_count,total_data_count))
        return img_dataset


    def load_train_dataset(self):
        #Creates train_loader and test_loader from the extracted dataset
        train_loader = DataLoader(self.get_dataset(self.train_path),
    batch_size=self.batch_size, shuffle=True)

        return self.get_dataset(self.train_path)

    def load_test_dataset(self):

        test_loader = DataLoader(self.get_dataset(self.test_path),
    batch_size=self.batch_size, shuffle=True)

        return self.get_dataset(self.test_path)


        
    def train(self,num_epochs=10):
        self.model.to(self.device)
        best_accuracy = 0.0
        train_dataset = self.load_train_dataset()
        train_count=len(glob(self.train_path+'/**/*.png'))

        classifier.fit()


        
        return best_accuracy

    def test(self):
        test_loader = self.load_test_dataset()
        test_count=len(glob(self.test_path+'/**/*.png'))
        self.model.eval()
        test_accuracy = 0.0

        for images,labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _,predictions = torch.max(outputs.data,1)
            test_accuracy += int(torch.sum(predictions==labels.data))
        test_accuracy /= test_count
        print("Test accuracy =  ",str(test_accuracy))

    def get_best_parameters(self):
        loaded_model = torch.load(self.dest_file)
        params = dict()
        for name,parameters in loaded_model.named_parameters():
            params[name] = parameters
            
        return params
    
    def get_model_parameters(self):
        #Returns model parameters asa dictionary 
        result = dict()
        for name,params in self.model.named_parameters():
            result[name] = params
        return result



if __name__ == '__main__':

    data_path = '/Users/tarunvisvar/Downloads/Dataset/Handwriting/Handwriting-subset'
    batch_size = 64
    local_data_percentage = 40
    parameter_list = [] # For testing aggregator function developed by Shasaank
    for i in range(5):
        mymodel = HWRModel(data_path)
        #mymodel.user_instance(i,batch_size,local_data_percentage)
        train_dataset = datasets.ImageFolder(mymodel.train_path,transform = mymodel.preprocess())
        test_dataset = datasets.ImageFolder(mymodel.test_path,transform = mymodel.preprocess())
        #classifier = mymodel.classifier
        #classifier.fit(img_dataset)
        train_count,test_count = len(train_dataset),len(test_dataset)
        #train_count,test_count = 5,5
        test_loader = DataLoader(test_dataset,batch_size=test_count)
        train_loader = DataLoader(train_dataset,batch_size=train_count)
        x_train,y_train = next(iter(train_loader))
        x_test,y_test = next(iter(test_loader))
        print(type(x_train))
        mymodel.classifier.fit(x_train,y_train,nb_epochs=5)
  
        print("--Training complete--")
        predictions = mymodel.classifier.predict(x_test)

        print("Prediction : ",np.argmax(predictions, axis=1))
        print("Prediction : ",np.argmax(predictions, axis=1))
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))


        '''mymodel.train(num_epochs = 5)
        mymodel.test()
        parameters = mymodel.get_best_parameters()
        for name,params in mymodel.model.named_parameters():
            print(name,params.shape)
        parameter_list.append(parameters)'''
        break
    with open('parameter_list.bin','wb') as f:
        pickle.dump(parameter_list,f)


    



   

        

