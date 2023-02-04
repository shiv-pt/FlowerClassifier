import argparse
import torch
import sys
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('Directory name')
    parser.add_argument('--arch', type=str, help='Choose either densenet121 or vgg13 (str)',required=False)
    parser.add_argument('--save_dir', type=str, help='Give path of directory to save checkpoint, else it will be lost (str)',required=False)
    parser.add_argument('--learning_rate', type=float, help='Provide learning rate for gradient descent (float)',required=False)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units for the model (int)',required=False)
    parser.add_argument('--epochs', type=int, help='Training iterations (int)',required=False)
    parser.add_argument('--gpu', action="store_true", help='Use GPU + Cuda for calculations',required=False)
    args = parser.parse_args()
    return args

def train_transform(train_dir):
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
   train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_data

def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
def load_data(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


def model_loader(architecture="vgg16"):
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    elif (architecture == "vgg13"): 
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
        print("Network architecture specified as vgg13.")
    elif (architecture == "densenet121"): 
        model = models.densenet121(pretrained=True)
        model.name = "densenet121"
        print("Network architecture specified as densenet121.")
    else:
        print("Only densenet121 or vgg13 can be chosen, given model does not match with any hence going with default vgg16")
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    #model.name = architecture    
    for param in model.parameters():
        param.requires_grad = False 
    return model

def initial_classifier(model, hidden_units):
    if type(hidden_units) == type(None): 
        hidden_units = 4096 
        print("Number of Hidden Layers specificed as 4096.")
    start_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(start_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier


def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def trainer(model, Trainloader, validloader, Device, 
                  criterion, Optimizer, Epochs, print_every, Steps):
    if type(Epochs) == type(None):
        Epochs = 5
        print("Number of Epochs specificed as 5.")     
    print("Training process initializing .....\n")

    for e in range(Epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            inputs, labels = inputs.to(Device), labels.to(Device)
            Optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
        
            running_loss += loss.item()
        
            if Steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, Device)
            
                print("Epoch: {}/{} | ".format(e+1, Epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))
            
                running_loss = 0
                model.train()

    return model

def validator(Model, Testloader, Device):
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))


def checkpoint(Model, Save_Dir, Train_data):
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            Model.class_to_idx = Train_data.class_to_idx
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            torch.save(checkpoint, Save_dir+'my_checkpoint.pth')
        else: 
            print("Directory not found, model will not be saved.")
 
def main():
    if(len(sys.argv)<2):
        print("Incorrect usage")
    else:
        data_dir = sys.argv[1]
        print("Directory: ",sys.argv[1])
        if(isdir(data_dir)):
            print("Using directory for data:",data_dir)
        else:
            print("No such Directory exist, using default flowers")
            data_dir = "flowers"
    args = arg_parser()
    print("Data dir: ",data_dir)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_data = train_transform(train_dir)
    valid_data = test_transform(valid_dir)
    test_data = test_transform(test_dir)
    
    trainloader = load_data(train_data)
    validloader = load_data(valid_data, train=False)
    testloader = load_data(test_data, train=False)
    
    model = model_loader(architecture=args.arch)
    
    model.classifier = initial_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print("Device = ",device)
    print_every = 30
    steps = 0
    trained_model = trainer(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    validator(trained_model, testloader, device)
    
    checkpoint(trained_model, args.save_dir, train_data)


if __name__ == '__main__': main()