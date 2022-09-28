from cProfile import label
from concurrent.futures import process
import imp
from logging.handlers import DatagramHandler
from re import I
from unittest import result
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import pickle
from scipy import stats
import sys
import h5py
import copy
sys.path.append("../Scripts/")
from models import model_v4_test,model_v3,model_v4_test_onlyseq
from util_funs import seq2index, cutseqs, highest_x, index2word_, word2index_
from util_att import cal_attention
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, sampler
from newlayer import *
from newlayer import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from torchvision import transforms
from sklearn.metrics import recall_score,precision_score,roc_auc_score,roc_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
threshold_list = [0.012895,0.016116,0.044296,0.090964,0.046770,0.120424, 0.171631, 0.051592,0.029826,0.065932, 0.138664, 0.073055] #my
# threshold_list = [0.002887,0.004897,0.001442,0.010347,0.036834,0.028677,0.009135,0.095019,0.001394,0.007883,0.113931,0.125591] #his
my_dict = pickle.load(open('../Embeddings/embeddings_12RM.pkl','rb'))  # 3-mers dict
num_task = 12
numgpu=3
chemical_property = {
                        'A': [1, 1, 1],
                        'C': [0, 1, 0],
                        'G': [1, 0, 0],
                        'T': [0, 0, 1],
                        'U': [0, 0, 1],
                        'N': [0, 0, 0],
                    }
def precision_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based precision

    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not (y_pred[i] == 0).all():
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_pred_posi = np.sum(y_pred[i])
            result += n_true_posi / n_pred_posi
    return result / n_samples

def recall_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based recall

    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not (y_true[i] == 0).all():
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_ground_true = np.sum(y_true[i])
            result += n_true_posi / n_ground_true
    return result / n_samples

def f1_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based recall
    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not ((y_true[i] == 0).all() and (y_pred[i] == 0).all()):
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_ground_true = np.sum(y_true[i])
            n_pred_posi = np.sum(y_pred[i])
            f1 = 2*(n_true_posi) / (n_ground_true+n_pred_posi)
            result += f1
    return result / n_samples
def plotprc(fpr_2,tpr_2,precisions_m,recalls_m,metrics,class_names=["Cm","Gm","Um","m1A","m5C","m5U","m6A","m7G","Y"]):
    colors = [(44, 160, 44), (152, 223, 138), (174, 199, 232),
            (255, 127, 14), (255, 187, 120),(214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213)]
    key=[1,2,3,4,5,6,7,9,10]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)

    # modifying parameters for plot
    from math import sqrt
    golden_mean = (sqrt(5)-1.0)/2.0
    fig_width = 6 # fig width in inches
    fig_height = fig_width*golden_mean # fig height in inches
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    # params = {'axes.labelsize': 10, # fontsize for x and y labels (was 10)
    #       'axes.titlesize': 10,
    #       'font.size': 10,
    #       'legend.fontsize': 10,
    #       'xtick.labelsize': 8,
    #       'ytick.labelsize': 8,
    #       'text.usetex': False,
    #       'font.family': 'serif'
    #       }
    lw = 3
    #fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(13,4),gridspec_kw={'width_ratios': [1, 2.2]})

    # roc curve
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(fig_width*2+0.7,fig_height+0.1))

    # PR curve
    fig_2, axes_2 = plt.subplots(nrows=1,ncols=2,figsize=(fig_width*2+0.7,fig_height+0.1))
    # matplotlib.rcParams.update(params)
    # set color palettes

    for i, class_name in zip(range(9), class_names):

        axes[0].plot(fpr_2[key[i]], tpr_2[key[i]], color=colors[i],lw=lw,
                    label ='%s ($AUC_{m}$ = %.3f)'%(class_name,
                            metrics['auc'][i]))
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.0])
        axes[0].tick_params(axis='x',which='both',top=False)
        axes[0].tick_params(axis='y',which='both',right=False,left=False,labelleft=False)
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC curves (multiple)')

        axes_2[0].plot(recalls_m[key[i]], precisions_m[key[i]], color=colors[i],lw=lw,
                    label ='%s ($AP_{m}$ = %.3f)'%(class_name,
                            metrics['ap'][i]))
        axes_2[0].set_xlim([0.0, 1.0])
        axes_2[0].set_ylim([0.0, 1.0])
        axes_2[0].tick_params(axis='x',which='both',top=False)
        axes_2[0].tick_params(axis='y',which='both',right=False,left=False,labelleft=True)
        xmin, xmax = axes_2[0].get_xlim()
        ymin, ymax = axes_2[0].get_ylim()
        axes_2[0].set_aspect(abs((xmax-xmin)/(ymax-ymin)), adjustable='box')
        axes_2[0].set_xlabel('Recall')
        axes_2[0].set_ylabel('Precision')
        axes_2[0].set_title('PR curves (multiple)')

    # Shrink current axis by 20%
    # box = axes[1].get_position()
    # print(box)
    # axes[1].set_position([box.x0, box.y0, box.x1-box.width * 0.5, box.height])
    # print(axes[1].get_position())
    axes[0].plot([0, 1], [0, 1], 'k--', lw=lw, label='no skill')
    axes_2[0].plot([0, 1], [0.04, 0.04], 'k--', lw=lw, label = 'no skill')

    # Put a legend to the right of the current axis
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1),borderaxespad=0.,frameon=False)
    axes_2[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1),borderaxespad=0.,frameon=False)

    fig.tight_layout()
    fig_2.tight_layout()

    fig.savefig('../Figs/mouse_roc_curve.pdf')
    fig_2.savefig('../Figs/mouse_precision_recall_curve.pdf')
def hamming_loss(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: hamming loss
    """
    n_samples = y_true.shape[0]
    n_classes = y_true.shape[1]
    loss = 0
    for i in range(n_samples):
        xor = np.sum((y_true[i] + y_pred[i]) % 2)
        loss += xor / n_classes
    return loss / n_samples

def chemical(seq):
    length=len(seq[0])
    out=np.zeros((seq.shape[0],length,4))
    for j in range(seq.shape[0]):
        for i in range(length):
            temp=chemical_property.get(seq[j][i]).copy()
            temp.append(list(seq[j][:i+1]).count(seq[j][i])/(i+1))
            # temp.append(EIIP[j][i])
            for k in range(4):
                out[j][i][k]=temp[k]
    return out

def evaluate(model, input_x):
    """
    Calculate the attention weights and predicted probabilities
    """
    model.eval()
    y_pred = model(input_x)
    y_score=[float(i) for i in y_pred]
    for i in range(12):
        y_pred = np.array([0 if instance < threshold_list[i] else 1 for instance in y_pred])
    return y_pred,y_score
def evaluatemy(model, input_x1,input_x2):
    """
    Calculate the attention weights and predicted probabilities
    """
    model.eval()
    y_pred = model(input_x1,input_x2)
    y_score=[float(i) for i in y_pred]
    for i in range(12):
        y_pred = np.array([0 if instance < threshold_list[i] else 1 for instance in y_pred])
    return y_pred,y_score
    
def evaluateseq(model, input_x1):
    """
    Calculate the attention weights and predicted probabilities
    """
    model.eval()
    y_pred = model(input_x1)
    y_score=[float(i) for i in y_pred]
    for i in range(12):
        y_pred = np.array([0 if instance < threshold_list[i] else 1 for instance in y_pred])
    return y_pred,y_score

def word2index_(my_dict):
    word2index = dict()
    for index, ele in enumerate(list(my_dict.keys())):
        word2index[ele] = index

    return word2index
def mapfun(x,my_dict):
    if x not in list(my_dict.keys()):
        return None
    else:
        return word2index_(my_dict)[x]
def seq2index(seqs,my_dict,window=3,save_data=False):
    """
    Convert single RNA sequences to k-mers representation.
        Inputs: ['ACAUG','CAACC',...] of equal length RNA seqs
        Example: 'ACAUG' ----> [ACA,CAU,AUG] ---->[21,34,31]
    """
    num_samples = len(seqs)
    temp = []
    for k in range(num_samples):
        length = len(seqs[k])
        seqs_kmers = [seqs[k][i:i+window].upper() for i in range(0,length-window+1)]
        temp.append(seqs_kmers)
    seq_kmers = pd.DataFrame(data = np.concatenate(temp,axis=0))
    # load pretained word2vec embeddings
    seq_kmers_index = seq_kmers.applymap(lambda x: mapfun(x,my_dict))
    return seq_kmers_index.to_numpy()
def three_mer(seq,embedding_path = '../Embeddings/embeddings_12RM.pkl'):
    res=[]
    for i in range(len(seq)):
        # print(seq2index([i], pickle.load(open(embedding_path, 'rb'))).transpose(1, 0))
        # print('ok')
        res.append(seq2index([seq[i]], pickle.load(open(embedding_path, 'rb'))).transpose(1,0)[0])
        if i%10000==1000:
            print('3-mer has finished {}%'.format(i*100/len(seq)))
    return np.array(res)
def gendata_orignal():
    data=pickle.load(open('./data.pkl','rb'))

    model = model_v3(num_task,use_embedding=True).cuda(numgpu)
    model_path = '../Weights/MultiRM/trained_model_51seqs.pkl'
    model.load_state_dict(torch.load(model_path))
    res=[]
    score=[]
    data=[i[476:527] for i in data]
    for i in range(len(data)):
        y_pred,y_score=evaluate(model,torch.tensor(three_mer([data[i]])))
        res.append(y_pred)    
        score.append(y_score)      
    # pickle.dump(res,open('./mouse_res.txt','wb'))
    # pickle.dump(score,open('./mouse_score.txt','wb'))
def evaluate_orignal():
    from sklearn.metrics import confusion_matrix,roc_auc_score
    # y_pred=np.array(pickle.load(open('./mouse_res.txt','rb')))
    y_true=np.array(pickle.load(open('./label.pkl','rb')))
    y_score=np.array(pickle.load(open('./mouse_score.txt','rb')))
    acc_l=[]
    sn_l=[]
    sp_l=[]
    mcc_l=[]
    auc_l=[]
    ap_2_l=[]
    Y_pred = np.zeros(y_true.shape)
    fpr,tpr = dict(), dict()
    fpr_2,tpr_2 = dict(), dict()
    precisions, recalls = dict(), dict()
    precisions_m, recalls_m = dict(), dict()
    dic={}
    metrics=['acc','mcc','sn','sp','auc','ap']
    for i in range(12):
        if i!=0 and i!=8 and i!=11:
            
            fpr_2[i], tpr_2[i], thresholds_2 = roc_curve(y_true[:,i], y_score[:,i])
            precisions_m[i], recalls_m[i], _ = precision_recall_curve(y_true[:,i], y_score[:,i])
            gmeans = np.sqrt(tpr_2[i] * (1-fpr_2[i]))
            # gmeans = np.sqrt(tpr[i] * (1-fpr[i]))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            print('Best Threshold=%f, G-Mean=%.3f' % (thresholds_2[ix], gmeans[ix]))

            best_threshold = thresholds_2[ix]
            # best_threshold = thresholds[ix]
            y_pred = np.array([0 if instance < best_threshold else 1 for instance in list(y_score[:,i])])
            Y_pred[:,i] = y_pred
            # tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred[:,i]).ravel()
            tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred).ravel()
            pp = tp+fn
            pn = tn+fp        
            sensitivity = tp / pp
            specificity = tn / pn
            recall = sensitivity
            precision = tp / (tp + fp)
            acc = (tp+tn) / (pp+pn)
            mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            auc = roc_auc_score(y_true[:,i],y_score[:,i])
            mcc_l.append(mcc)
            acc_l.append(acc)
            sn_l.append(sensitivity)
            sp_l.append(specificity)
            auc_l.append(auc)
            ap_2 = average_precision_score(y_true[:,i], y_score[:,i])
            ap_2_l.append(ap_2)
    dic[metrics[0]]=acc_l
    dic[metrics[1]]=mcc_l
    dic[metrics[2]]=sn_l
    dic[metrics[3]]=sp_l
    dic[metrics[4]]=auc_l
    dic[metrics[5]]=ap_2_l
    print(dic)
    precision_multi_ = precision_multi(y_true,Y_pred)
    recall_multi_ = recall_multi(y_true,Y_pred)
    f1_multi_ = f1_multi(y_true,Y_pred)
    hamming_loss_ = hamming_loss(y_true,Y_pred)
    print("precision multi: %f"%(precision_multi_))
    print("recall multi: %f"%(recall_multi_))
    print("f1 multi: %f"%(f1_multi_))
    print("hamming loss: %f"%(hamming_loss_))
    plotprc(fpr_2,tpr_2,precisions_m,recalls_m,dic)
    pickle.dump(dic,open('./mouse_org_metric.txt','wb'))

# gendata_orignal()
evaluate_orignal()
def gendata_my():
    data=pickle.load(open('./data.pkl','rb'))
    model = model_v4_test(num_task).cuda(numgpu)
    # model_path = '../Best_Weights/cnnANFtrained_model_51seqs.pkl'
    model_path = '../Best_Weights/seq+NCP+ANF1001.pkl'
    model.load_state_dict(torch.load(model_path))
    res=[]
    score=[]
    data1=[i[476:527].upper() for i in data]
    for i in range(len(data)):
        y_pred,y_score=evaluatemy(model,torch.tensor(three_mer([data[i]])),torch.tensor(chemical(np.array([data1[i]]))))
        res.append(y_pred) 
        score.append(y_score)    
    pickle.dump(res,open('./mouse_res_my_onlyseq.txt','wb'))
    pickle.dump(score,open('./mouse_score_my.txt','wb'))
def evaluate_my():
    from sklearn.metrics import confusion_matrix,roc_auc_score
    # y_pred=np.array(pickle.load(open('./mouse_res_my.txt','rb')))
    y_true=np.array(pickle.load(open('./label.pkl','rb')))
    y_score=np.array(pickle.load(open('./mouse_score_my.txt','rb')))
    acc_l=[]
    sn_l=[]
    sp_l=[]
    mcc_l=[]
    auc_l=[]
    ap_2_l=[]
    Y_pred = np.zeros(y_true.shape)
    dic={}
    metrics=['acc','mcc','sn','sp','auc','ap']
    fpr,tpr = dict(), dict()
    fpr_2,tpr_2 = dict(), dict()
    
    precisions, recalls = dict(), dict()
    precisions_m, recalls_m = dict(), dict()

    for i in range(12):
        if i!=0 and i!=8 and i!=11:
                      

            fpr_2[i], tpr_2[i], thresholds_2 = roc_curve(y_true[:,i], y_score[:,i])
            precisions_m[i], recalls_m[i], _ = precision_recall_curve(y_true[:,i], y_score[:,i])
            gmeans = np.sqrt(tpr_2[i] * (1-fpr_2[i]))
            # gmeans = np.sqrt(tpr[i] * (1-fpr[i]))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            print('Best Threshold=%f, G-Mean=%.3f' % (thresholds_2[ix], gmeans[ix]))

            best_threshold = thresholds_2[ix]
            # best_threshold = thresholds[ix]
            y_pred = np.array([0 if instance < best_threshold else 1 for instance in list(y_score[:,i])])
            Y_pred[:,i] = y_pred
            # tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred[:,i]).ravel()
            tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred).ravel()  
            pp = tp+fn
            pn = tn+fp        
            sensitivity = tp / pp
            specificity = tn / pn
            recall = sensitivity
            precision = tp / (tp + fp)
            acc = (tp+tn) / (pp+pn)
            mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            auc = roc_auc_score(y_true[:,i],y_score[:,i])
            mcc_l.append(mcc)
            acc_l.append(acc)
            sn_l.append(sensitivity)
            sp_l.append(specificity)
            auc_l.append(auc)   
            ap_2 = average_precision_score(y_true[:,i], y_score[:,i])
            ap_2_l.append(ap_2)


    dic[metrics[0]]=acc_l
    dic[metrics[1]]=mcc_l
    dic[metrics[2]]=sn_l
    dic[metrics[3]]=sp_l
    dic[metrics[4]]=auc_l
    dic[metrics[5]]=ap_2_l
    # print(dic["ap"],ap_2_l,"12345678")
    print(dic)
    precision_multi_ = precision_multi(y_true,Y_pred)
    recall_multi_ = recall_multi(y_true,Y_pred)
    f1_multi_ = f1_multi(y_true,Y_pred)
    hamming_loss_ = hamming_loss(y_true,Y_pred)

    print("precision multi: %f"%(precision_multi_))
    print("recall multi: %f"%(recall_multi_))
    print("f1 multi: %f"%(f1_multi_))
    print("hamming loss: %f"%(hamming_loss_))
    pickle.dump(dic,open('./mouse_my_metric.txt','wb'))
    plotprc(fpr_2,tpr_2,precisions_m,recalls_m,dic)
    

# gendata_my()
# evaluate_my()

def gendata_seq():
    data=pickle.load(open('./data.pkl','rb'))
    model = model_v4_test_onlyseq(num_task).cuda(numgpu)
    model_path = '../Results/seq1001.pkl'
    model.load_state_dict(torch.load(model_path))
    res=[]
    score=[]
    for i in range(len(data)):
        y_pred,y_score=evaluateseq(model,torch.tensor(three_mer([data[i]])))
        res.append(y_pred)   
        score.append(y_score)  
    pickle.dump(res,open('./mouse_res_my_onlyseq.txt','wb'))
    pickle.dump(score,open('./mouse_score_my_nolyseq.txt','wb'))
def evaluate_seq():
    from sklearn.metrics import confusion_matrix,roc_auc_score
    y_pred=np.array(pickle.load(open('./mouse_res_my_onlyseq.txt','rb')))
    y_true=np.array(pickle.load(open('./label.pkl','rb')))
    y_score=np.array(pickle.load(open('./mouse_score_my_nolyseq.txt','rb')))
    acc_l=[]
    sn_l=[]
    sp_l=[]
    mcc_l=[]
    auc_l=[]
    dic={}
    metrics=['acc','mcc','sn','sp','auc']
    for i in range(12):
        if i!=0 and i!=8 and i!=11:
            tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred[:,i]).ravel()
            pp = tp+fn
            pn = tn+fp        
            sensitivity = tp / pp
            specificity = tn / pn
            recall = sensitivity
            precision = tp / (tp + fp)
            acc = (tp+tn) / (pp+pn)
            mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            auc = roc_auc_score(y_true[:,i],y_score[:,i])
            mcc_l.append(mcc)
            acc_l.append(acc)
            sn_l.append(sensitivity)
            sp_l.append(specificity)
            auc_l.append(auc)
    dic[metrics[0]]=acc_l
    dic[metrics[1]]=mcc_l
    dic[metrics[2]]=sn_l
    dic[metrics[3]]=sp_l
    dic[metrics[4]]=auc_l
    print(dic)


    # pickle.dump(dic,open('./mouse_seq_metric.txt','wb'))
# gendata_seq()
# evaluate_seq()
