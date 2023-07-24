import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
import time

import warnings
warnings.filterwarnings(action='ignore')

HYP = {
    'EPOCHS': 2,
    'LEARNING_RATE':1e-2,
    'SEED':42
}

def preprocess(data):
    
    #Time Stamp 따로 저장
    data_time = pd.DataFrame(data['TimeStamp'])
    
    # 필요없는 컬럼 삭제
    X_drop = data.drop(['_id','NGmark','Hopper_Temperature','Mold_Temperature_1','Mold_Temperature_2',
                                      'Mold_Temperature_3','Mold_Temperature_4','Mold_Temperature_5','Mold_Temperature_6',
                                      'Mold_Temperature_7','Mold_Temperature_8','Mold_Temperature_9','Mold_Temperature_10',
                                      'Mold_Temperature_11','Mold_Temperature_12','Clamp_open_time','Cavity',
                                      'idx','Machine_Name', 'Additional_Info_1', 'Additional_Info_2','TimeStamp','Shot_Number'],axis=1)
    
    
    # 결측치 처리
    if X_drop.isnull().any().any():
        X_drop = X_drop.fillna(0)
        
    else:
        X_drop = X_drop


    #MinMax Scaling
    X_pre = MinMaxScaler().fit_transform(X_drop)
    X_pre = pd.DataFrame(X_pre,columns= X_drop.columns)


    return X_pre, data_time

class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded


#Denoising AutoEncoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.ReLU()
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
    
    
def validation_teacher(model, val_loader, criterion):
    model.eval()

    val_mae = []
    
    with torch.no_grad():
        for X in tqdm(val_loader):
            X = X.float()
            
            model_pred = model(X)
            
            loss = criterion(model_pred, X)
            val_mae.append(loss.item())      
      
    return val_mae

def train(model, optimizer, train_loader, val_loader, scheduler):
    criterion = nn.L1Loss()


    for epoch in range(HYP["EPOCHS"]):  
        train_mae = []
  
        model.train()
    
        for X in tqdm(train_loader):
            X = X.float()
        
            optimizer.zero_grad() #옵티마이저의 gradient 초기화
            
            teacher_pred = model(X) #teacher_pred로 변경
            
            loss = criterion(teacher_pred, X)
            loss.backward()
            
            optimizer.step()

            train_mae.append(loss.item())
            #training mae


        val_mae = validation_teacher(model, val_loader, criterion)
        print(f'Epoch [{epoch}], Train Loss : [{np.mean(train_mae) :.5f}] Val Loss : [{np.mean(val_mae) :.5f}]')
        
           
    return train_mae, val_mae

def distillation(student_logits, labels, teacher_logits, alpha):

    distillation_loss = nn.L1Loss()(student_logits,teacher_logits)
    student_loss = nn.L1Loss()(student_logits, labels)
    
    return alpha * student_loss + (1-alpha) * distillation_loss



def distill_loss(output, target, teacher_output, loss_fn=distillation, opt=None):
    loss_b = loss_fn(output, target, teacher_output, alpha=0.1)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    return loss_b.item()

def validation_student(s_model, t_model, val_loader, criterion):
    s_model.eval()
    t_model.eval()

    val_loss = []
    threshold = 0.1
    
    with torch.no_grad():
        for X in tqdm(val_loader):
            X = X.float()

            model_pred = s_model(X)
            teacher_output = t_model(X)
            
            loss_b = distill_loss(model_pred, X, teacher_output, loss_fn=distillation, opt=None)
            val_loss.append(loss_b)
        
    return val_loss


def student_train(s_model, t_model, optimizer, train_loader, val_loader, scheduler):

    for epoch in range(HYP["EPOCHS"]):
        train_loss = []
        val_loss = []
        
        s_model.train()
        t_model.eval()
        
        for X in tqdm(train_loader):
            X = X.float()

            optimizer.zero_grad()

            output = s_model(X)
                  
            with torch.no_grad():
                teacher_output = t_model(X)
                
            loss_b = distill_loss(output, X, teacher_output, loss_fn=distillation, opt=optimizer)
            # print(loss_b)
            train_loss.append(loss_b)

        val_loss = validation_student(s_model, t_model, val_loader, distill_loss)
        print(f'Epoch [{epoch}], Distill Loss: {np.mean(train_loss):.5f}, Val Loss: {np.mean(val_loss):.5f}')
        
    return train_loss, val_loss

def inference(s_model, t_model, test_loader, timeStamp):
    s_model.eval()
    t_model.eval()

    result = pd.DataFrame()
    threshold = 0.1
    
    result['TimeStamp'] = timeStamp
    loss_list = []  
    
    with torch.no_grad():
        for X in tqdm(test_loader):
            X = X.float()

            model_pred = s_model(X)
            teacher_output = t_model(X)
            
            loss_b = distill_loss(model_pred, X, teacher_output, loss_fn=distillation, opt=None)
            
            if not isinstance(loss_b, torch.Tensor): 
                loss_b = torch.tensor(loss_b)
            
            loss_list.append(loss_b.item())  
            
    result['Loss MAE'] = loss_list  
    
    anomaly_list = np.where(np.array(loss_list) > threshold, 1, 0)
    result['Anomaly'] =  anomaly_list
    
    return result

