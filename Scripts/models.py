import torch
import numpy as np
from torch import nn
from util_layers import *
from train_utils import *
from torch.nn import TransformerEncoder,TransformerEncoderLayer
numgpu=3
class NaiveNet(nn.Module):
    """
        CNN only
    """

    def __init__(self, input_size=None):
        super(NaiveNet, self).__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=7, stride=2, padding=0),  # [bs, 8, 72]
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),  # [bs 32 72]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0),  # [bs 32 36]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),  # [bs 128 36]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)  # [bs 128 18]
        )
        # self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        in_features_1 = (input_size - 7) // 2 + 1
        in_features_2 = (in_features_1 - 2) // 2 + 1
        in_features_3 = (in_features_2 - 2) // 2 + 1
        self.Flatten = nn.Flatten()
        self.SharedFC = nn.Sequential(nn.Linear(in_features=128 * in_features_3, out_features=512),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )

    def forward(self, x):
        x = self.NaiveCNN(x)
        output = self.Flatten(x)  # flatten output
        outs = self.SharedFC(output)
        return outs

class NaiveNet_v1(nn.Module):
    """
        CNN + LSTM + Attention
    """
    def __init__(self,input_size=None,num_task=None):
        self.num_task = num_task
        super(NaiveNet_v1, self).__init__()
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=1),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=1)
                        )
        self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        self.Attention = BahdanauAttention(in_features=256,hidden_units=10,num_task=num_task)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=256,out_features=64),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=64,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):
        x = self.NaiveCNN(x)
        batch_size, features, seq_len = x.size()
        x = x.view(batch_size,seq_len, features) # parepare input for LSTM
        output, (h_n, c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1]) # pareprae input for Attention
        context_vector,attention_weights = self.Attention(h_n,output) # Attention (batch_size, num_task, unit)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(context_vector[:,i,:])
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs

class NaiveNet_v2(nn.Module):
    """
        CNN + LSTM
    """
    def __init__(self,input_size=None,num_task=None):
        self.num_task = num_task
        super(NaiveNet_v2, self).__init__()
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0)
                        )
        in_features_1 = (input_size - 7) // 2 + 1
        in_features_2 = (in_features_1 - 2) // 2 + 1
        in_features_3 = (in_features_2 - 2) // 2 + 1
        self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        self.Flatten = nn.Flatten()
        self.SharedFC = nn.Sequential(nn.Linear(in_features=in_features_3*256,out_features=1024),
                                    nn.ReLU(),
                                    nn.Dropout()
                                    )
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                      nn.Linear(in_features=1024,out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                       nn.Linear(in_features=256,out_features=64),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=64,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):
        x = self.NaiveCNN(x)
        batch_size, features, seq_len = x.size()
        x = x.view(batch_size,seq_len, features) # parepare input for LSTM
        output, (h_n, c_n) = self.NaiveBiLSTM(x)
        output = self.Flatten(output) # flatten output
        shared_layer = self.SharedFC(output)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(shared_layer)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs


class model_v3(nn.Module):

    def __init__(self,num_task,use_embedding):
        super(model_v3,self).__init__()

        self.num_task = num_task
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embed = EmbeddingSeq('../Embeddings/embeddings_12RM.pkl') # Word2Vec
            # self.embed = EmbeddingHmm(t=3,out_dims=300) # hmm
            self.NaiveBiLSTM = nn.LSTM(input_size=300,hidden_size=256,batch_first=True,bidirectional=True)
        else:
            self.NaiveBiLSTM = nn.LSTM(input_size=4,hidden_size=256,batch_first=True,bidirectional=True)

        self.Attention = BahdanauAttention(in_features=512,hidden_units=100,num_task=num_task)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):

        if self.use_embedding:
            x = self.embed(x)
        else:
            x = torch.transpose(x,1,2)
        batch_size = x.size()[0]
        # x = torch.transpose(x,1,2)

        output,(h_n,c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1])
        context_vector,attention_weights = self.Attention(h_n,output)
        # print(attention_weights.shape)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(context_vector[:,i,:])
            y = torch.squeeze(y, dim=-1)
            outs.append(y)

        return outs

# *************************************************************
class model_v4(nn.Module):

    def __init__(self,num_task):
        super(model_v4,self).__init__()

        self.num_task = num_task
        # self.linear = nn.Linear(4, 512)
        # self.embed = EmbeddingSeq('../Embeddings/embeddings_12RM.pkl') 
        self.transfomer= MyTransformerModel(embedding_dim=300, p_drop=0.1, h=2, output_size=512)
        # self.transfomerlayer=TransformerEncoderLayer(d_model=4,nhead=2,dim_feedforward=128)
        # self.transfomer2=TransformerEncoder(self.transfomerlayer,num_layers=6)
        self.cnn=NaiveNet(input_size=51)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=1024,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))
    def forward(self,x,fp,fn):
        x=x.permute(1,0)
        output1=self.transfomer(three_mer(x,fp).cuda(numgpu)).cuda(numgpu)
        output1=torch.reshape(output1,(-1,512))
        out=NCP(x,fn)
        out=torch.tensor(out,dtype=torch.float32).permute(0, 2, 1)
        output2=self.cnn(out.cuda(numgpu)).cuda(numgpu)
        output=torch.cat((output1,output2),-1)
        file_handle=open('TS.txt',mode='a')
        #将tensor变量转化为numpy类型
        x = output.cpu().detach().numpy()
        #将numpy类型转化为list类型
        x=x.tolist()
        #将list转化为string类型
        strNums=[str(x_i) for x_i in x]
        str1=",".join(strNums)
        #将str类型数据存入本地文件1.txt中
        file_handle.write(str1)

        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(output)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs

class model_v4_test(nn.Module):

    def __init__(self,num_task):
        super(model_v4_test,self).__init__()

        self.num_task = num_task
        self.transfomer= MyTransformerModel(embedding_dim=300, p_drop=0.1, h=2, output_size=512)
        self.cnn=NaiveNet(input_size=51)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                    #    nn.Linear(in_features=1024,out_features=512),
                                    #    nn.ReLU(),
                                    #    nn.Dropout(),
                                       nn.Linear(in_features=1024,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))
    def forward(self,x1,x2):
        output1=self.transfomer(x1.cuda(numgpu)).cuda(numgpu)
        output1=torch.reshape(output1,(-1,512))
        out=x2
        out=torch.tensor(out,dtype=torch.float32).permute(0, 2, 1)
        output2=self.cnn(out.cuda(numgpu)).cuda(numgpu)
        output=torch.cat((output1,output2),-1)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(output)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs
class model_v4_test_onlyseq(nn.Module):

    def __init__(self,num_task):
        super(model_v4_test_onlyseq,self).__init__()

        self.num_task = num_task
        self.transfomer= MyTransformerModel(embedding_dim=300, p_drop=0.1, h=2, output_size=512)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                    #    nn.Linear(in_features=1024,out_features=512),
                                    #    nn.ReLU(),
                                    #    nn.Dropout(),
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))
    def forward(self,x1):
        output1=self.transfomer(x1.cuda(numgpu)).cuda(numgpu)
        output1=torch.reshape(output1,(-1,512))
        output=output1
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(output)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs
class model_v5(nn.Module):

    def __init__(self,num_task):
        super(model_v5,self).__init__()

        self.num_task = num_task
        self.linear = nn.Linear(4, 512)
        self.transfomer1= MyTransformerModel(embedding_dim=300, p_drop=0.2, h=6, output_size=512)
        # self.embed = EmbeddingSeq('../Embeddings/embeddings_12RM.pkl') 
        # self.transfomerlayer=TransformerEncoderLayer(d_model=300,nhead=4,dim_feedforward=128)
        # self.transfomer1=TransformerEncoder(self.transfomerlayer,num_layers=6)
        # self.cnn=NaiveNet(input_size=1001)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))
    def forward(self,x,fp):
        x=x.permute(1,0)
        output1=self.transfomer1(three_mer(x,fp).cuda(numgpu)).cuda(numgpu)
        # input1=three_mer(x,fp)
        # input1=self.embed(input1)
        # output1=self.transfomer(input1)
        # output1=output1.sum(1) / (output1.shape[1] + 1e-5)
        # output1=self.linear(output1).squeeze()
        output=torch.reshape(output1,(-1,512))
        # output=output1
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(output)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs

class model_v5_test(nn.Module):

    def __init__(self,num_task):
        super(model_v4_test_onlyseq,self).__init__()

        self.num_task = num_task
        self.transfomer= MyTransformerModel(embedding_dim=300, p_drop=0.1, h=2, output_size=512)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                    #    nn.Linear(in_features=1024,out_features=512),
                                    #    nn.ReLU(),
                                    #    nn.Dropout(),
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))
    def forward(self,x1):
        output1=self.transfomer(x1.cuda(numgpu)).cuda(numgpu)
        output1=torch.reshape(output1,(-1,512))
        output=output1
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(output)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs

class model_v6(nn.Module):

    def __init__(self,num_task):
        super(model_v6,self).__init__()

        self.num_task = num_task
        # self.linear = nn.Linear(4, 512)
        # self.embed = EmbeddingSeq('../Embeddings/embeddings_12RM.pkl') 
        # self.transfomer= MyTransformerModel(embedding_dim=300, p_drop=0.1, h=2, output_size=512)
        # self.transfomerlayer=TransformerEncoderLayer(d_model=4,nhead=2,dim_feedforward=128)
        # self.transfomer2=TransformerEncoder(self.transfomerlayer,num_layers=6)
        self.cnn=NaiveNet(input_size=51)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))
    def forward(self,x,fp,fn):
        x=x.permute(1,0)
        # output1=self.transfomer(three_mer(x,fp).cuda(numgpu)).cuda(numgpu)
        # output1=torch.reshape(output1,(-1,512))
        out=NCP(x,fn)
        out=torch.tensor(out,dtype=torch.float32).permute(0, 2, 1)
        output2=self.cnn(out.cuda(numgpu)).cuda(numgpu)
        # output=torch.cat((output1,output2),-1)
        output=output2
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(output)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs
