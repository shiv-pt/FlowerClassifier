import argparse
import json
import PIL
import torch
import sys
import numpy as np
from os.path import isfile
from math import ceil
from train import check_gpu
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('Input')
    parser.add_argument('Checkpoint')
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.',required=False)
    parser.add_argument('--category_names', type=str, help='Mapping from categories to real names.',required=False)
    parser.add_argument('--gpu', action="store_true", help='Use GPU + Cuda for calculations',required=False)
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load("my_checkpoint.pth")
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    elif (checkpoint['architecture'] == "vgg13"): 
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
    elif (checkpoint['architecture'] == "densenet121"): 
        model = models.vgg13(pretrained=True)
        model.name = "densenet121"
    else:
        print("Only densenet121 or vgg13 can be chosen, given model does not match with any hence going with default vgg16")
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    
    
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    test_image = PIL.Image.open(image_path)
    orig_width, orig_height = test_image.size

    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    np_image = np.array(test_image)/255 
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def predict(image_tensor, model, device, cat_to_name, top_k):    
    if type(top_k) == type(None):
        top_k = 5
        print("Top K not specified, assuming K=5.")
    model.eval();
    torch_image = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)

    model=model.cpu()
    log_probs = model.forward(torch_image)
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(top_k)
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))

def main():
    if(len(sys.argv)<3):
        print("Incorrect usage")
    else:
        image = sys.argv[1]
        checkpoint = sys.argv[2]
        print("Img and Checkpoint: ",sys.argv[0],sys.argv[1])
        if(isfile(image) and isfile(checkpoint)):
            print("Using given image and checkpoint for data")
        else:
            print("No such image or checkpoint exist",isfile(image),isfile(checkpoint))
    args = arg_parser()
    if(type(args.category_names) == type(None)):
        print("Using default category names")
        args.category_names = "cat_to_name.json"
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    model = load_checkpoint(checkpoint)    
    image_tensor = process_image(image)    
    device = check_gpu(gpu_arg=args.gpu);    
    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)
    print_probability(top_flowers, top_probs)

if __name__ == '__main__': main()