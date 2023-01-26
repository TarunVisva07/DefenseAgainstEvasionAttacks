import torch, torchvision
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms,datasets
from glob import glob
import pickle
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")

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
        self.batch_size = 1
        self.local_data_percentage = 100
        self.device = torch.device('mps')


    
    def preprocess(self):
        transformer = transforms.Compose(
            [
                transforms.Resize(150),
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
        return local_dataset


    def load_train_dataset(self):
        #Creates train_loader and test_loader from the extracted dataset
        train_loader = DataLoader(self.get_dataset(self.train_path),
    batch_size=self.batch_size, shuffle=True)

        return train_loader

    def load_test_dataset(self):

        test_loader = DataLoader(self.get_dataset(self.test_path),
    batch_size=self.batch_size, shuffle=True)

        return test_loader


        
    def train(self,num_epochs=10):
        self.model.to(self.device)
        best_accuracy = 0.0
        train_loader = self.load_train_dataset()
        train_count=len(glob(self.train_path+'/**/*.png'))
        

        self.model.train()
        for epoch in range(num_epochs):
            #Model will be in training mode and takes place on training dataset
            train_loss = 0.0
            train_accuracy = 0.0
            c = 0
            for images,labels in train_loader:
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images) 
                c += 1
                if c==1: print("Inside model = ",outputs.shape,images.shape,type(outputs),type(images),outputs,labels)
                loss = self.loss_func(outputs,labels)
                loss.backward() # backpropagation
                self.optimizer.step() # Updates the weights

                train_loss += loss.data*self.batch_size
                _,predictions = torch.max(outputs.data,1)
                train_accuracy+=int(torch.sum(predictions==labels.data))
            train_accuracy /= train_count
            train_loss /= train_count
            print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy))


            if train_accuracy>best_accuracy:
                
                torch.save(self.model,self.dest_file)
                best_accuracy=train_accuracy
        self.model = torch.load(self.dest_file)
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

    def predict(self,filename):
        self.model.to(torch.device('cpu'))
        transformer = self.preprocess()
        image=Image.open(filename).convert('RGB')
        image_tensor=transformer(image)
        image_tensor=image_tensor.unsqueeze_(0)
        output=self.model(image_tensor)
        print("Prediction = ",image_tensor.shape) #torch.Size([1, 3, 150, 150])
        index = output.data.numpy().argmax()
        print('index = ',index)
        return output


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
    local_data_percentage = 100
    parameter_list = [] # For testing aggregator function developed by Shasaank
    for i in range(5):
        mymodel = HWRModel(data_path)
        #mymodel.user_instance(i,batch_size,local_data_percentage)
        mymodel.train(num_epochs = 1)
        mymodel.test()
        print(mymodel.predict('/Users/tarunvisvar/Downloads/Dataset/Handwriting/Handwriting-subset/Test/Normal/A-56.png'))
        parameters = mymodel.get_best_parameters()
        #for name,params in mymodel.model.named_parameters():
        #    print(name,params.shape)
        parameter_list.append(parameters)
        break
    with open('parameter_list.bin','wb') as f:
        pickle.dump(parameter_list,f)


    



   

        

