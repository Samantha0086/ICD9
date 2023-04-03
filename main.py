

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from model import Word2Vec_neg_sampling
from utils_modified import count_parameters
from datasets import word2vec_dataset

from helper import evaluate,data_loader

import pandas as pd

from sklearn.metrics import roc_auc_score



outcome = "MORTALITY_30_DAY"
predictor =["PROCEDURE_ICD", "DIAGNOSIS_ICD", "PROCEDURE_AND_DIAGNOSIS_ICD"][2]



X = pd.read_csv("admit_modified.csv")[[outcome,'LOS', 'AGE', 'GENDER_M', "ETHNICITY_Asian", 
     "ETHNICITY_Black", "ETHNICITY_Hispanic", "ETHNICITY_Native_Hawaiian", "ETHNICITY_Other", 
     "ETHNICITY_White", predictor]]
X=X.dropna()
X = X.reset_index().drop(columns = ["index"])
y = X[outcome].values
X = X.drop(columns = outcome)    



for i in range(len(X)): 
    X[predictor][i] = X[predictor][i].replace("'", "")[1:-1].split(", ")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 1, stratify = y) #stratify
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)
train_index = X_train.index
test_index = X_test.index




# Skip-Gram

NEGATIVE_SAMPLES = 20
LR                    = 0.001


BATCH_SIZE            = 256
batch_size = 2**5

NUM_EPOCHS            = 30 #int(1e+3)  

weight_cnn = 0.8
EMBEDDING_DIM = 200
DEVICE = 'cpu'

# add other variables with ICD code together
class_weights = torch.tensor(compute_class_weight( class_weight ="balanced", classes =  np.unique(y_train),y =  y_train ), dtype = torch.float)
criterion_cnn = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')




dataset = word2vec_dataset(predictor, X, train_index, test_index)



X_train = X_train.drop(columns = predictor)
X_test = X_test.drop(columns = predictor)




# takes time
vocab = dataset.vocab

word_to_ix = dataset.word_to_ix


vocab_size = len(word_to_ix.keys())
embedding_dict = word_to_ix


train_dataloader, val_dataloader =  data_loader( dataset.code_same_len[train_index],  np.array(X_train), dataset.code_same_len[test_index], np.array(X_test), y_train, y_test, batch_size=BATCH_SIZE)

train_loader_sg = torch.utils.data.DataLoader(dataset.training_data, batch_size = batch_size, shuffle = not True)
test_loader_sg = torch.utils.data.DataLoader(dataset.testing_data, batch_size = batch_size, shuffle = not True)

print('len(dataset): ', len(dataset))
print('len(train_loader_sg): ', len(train_loader_sg))
print('len(train_dataloader): ', len(train_dataloader))
print('len(vocab): ', len(vocab), '\n')


# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab))
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))


losses = []

word2vec = Word2Vec_neg_sampling(EMBEDDING_DIM, len(vocab), DEVICE, noise_dist, NEGATIVE_SAMPLES).to(DEVICE)
print('\nWe have {} Million trainable parameters here in the word2vec'.format(count_parameters(word2vec)))


optimizer = optim.Adam(word2vec.parameters(), lr = LR)

for epoch in range(NUM_EPOCHS):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))    
    print('\nTRAINING...')

    train_accuracy = [ ]
    train_auroc = []
    
    train_loss = []
    word2vec.train()
    
    for item1, item2 in zip(train_loader_sg, train_dataloader): 
        x_batch = item1[:,0]
        y_batch = item1[:,1]
        
        # X is the input ids and X1 is other features
        X, X1, y = item2

        optimizer.zero_grad()
        
        
        loss_word2vec, logits = word2vec(x_batch, y_batch, X, X1)

        
        loss_cnn = criterion_cnn(logits,y)
        loss =( 1-weight_cnn)*loss_word2vec + weight_cnn* loss_cnn  # * weight
        train_loss.append(loss.item())


        loss.backward(retain_graph=True)
        optimizer.step()    
        
                
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        proba =logits[:,1].detach().numpy()


        accuracy = (preds == y).cpu().numpy().mean() * 100
        
        
        auroc =  roc_auc_score(y, proba)
        
        train_accuracy.append(accuracy)
        train_auroc.append(auroc)
    


    print("average train loss", np.mean(train_loss), '\n')
    print("average train accuracy", np.mean(train_accuracy), '\n')
    print("train auroc", np.mean(train_auroc)*100, '\n')
    
   
    print("VALIDATION... \n")
    val_loss, val_accuracy, val_auroc = evaluate(word2vec, val_dataloader, test_loader_sg, word2vec, criterion_cnn)
    print("val loss", val_loss, '\n')
    print("val accuracy", val_accuracy, '\n')
    print("val auroc", np.mean(val_auroc)*100, '\n')



# weight = 0.6, embedding_dim = 100
# ===== EPOCH 1/100 =====
# 
# TRAINING...
# average train loss 3.9155079454183577 
# 
# average train accuracy 83.22412109375 
# 
# train auroc 91.91028331074543 
# 
# VALIDATION... 
# 
# val loss 3.7634656809270384 
# 
# val accuracy 82.27201021634616 
# 
# val auroc 91.96821294658534 
# 
# 
# ===== EPOCH 2/100 =====
# 
# TRAINING...
# average train loss 2.367410108726472 
# 
# average train accuracy 84.38468424479166 
# 
# train auroc 92.87879754215513 
# 
# VALIDATION... 
# 
# val loss 2.767479309439659 
# 
# val accuracy 83.77178485576924 
# 
# val auroc 92.35547953694191 
# 
# 
# ===== EPOCH 3/100 =====
# 
# TRAINING...
# average train loss 1.7579727657139301 
# 
# average train accuracy 85.508056640625 
# 
# train auroc 93.7140587962458 
# 
# VALIDATION... 
# 
# val loss 2.200912281870842 
# 
# val accuracy 83.48031850961539 
# 
# val auroc 92.42236972575364 
# 
# 
# ===== EPOCH 4/100 =====
# 
# TRAINING...
# average train loss 1.4303127264603972 
# 
# average train accuracy 85.78499348958334 
# 
# train auroc 94.09725057340121 
# 
# VALIDATION... 
# 
# val loss 1.876555996388197 
# 
# val accuracy 85.80416165865384 
# 
# val auroc 92.60608240155592 
# 
# 
# ===== EPOCH 5/100 =====
# 
# TRAINING...
# average train loss 1.2393908779136837 
# 
# average train accuracy 86.40804036458334 
# 
# train auroc 94.49093208956943 
# 
# VALIDATION... 
# 
# val loss 1.6534370858222247 
# 
# val accuracy 83.30679086538461 
# 
# val auroc 92.61719051600984 
# 
# 
# ===== EPOCH 6/100 =====
# 
# TRAINING...
# average train loss 1.1127580680884421 
# 
# average train accuracy 86.73697916666666 
# 
# train auroc 94.74948724631406 
# 
# VALIDATION... 
# 
# val loss 1.4737441089004277 
# 
# val accuracy 84.64806189903847 
# 
# val auroc 92.74669628594555 
# 
# 
# ===== EPOCH 7/100 =====
# 
# TRAINING...
# average train loss 1.0189027469605207 
# 
# average train accuracy 87.12516276041666 
# 
# train auroc 94.99831069630264 
# 
# VALIDATION... 
# 
# val loss 1.3613409351557493 
# 
# val accuracy 84.08541165865384 
# 
# val auroc 92.96671037262449 
# 
# 
# ===== EPOCH 8/100 =====
# 
# TRAINING...
# average train loss 0.9469808918423951 
# 
# average train accuracy 87.27034505208334 
# 
# train auroc 95.24196190314336 
# 
# VALIDATION... 
# 
# val loss 1.2684577543288469 
# 
# val accuracy 85.85561899038461 
# 
# val auroc 92.95718559594987 
# 
# 
# ===== EPOCH 9/100 =====
# 
# TRAINING...
# average train loss 0.903414801787585 
# 
# average train accuracy 87.68082682291666 
# 
# train auroc 95.35076562587035 
# 
# VALIDATION... 
# 
# val loss 1.210795335099101 
# 
# val accuracy 83.29477163461539 
# 
# val auroc 92.84943607874038 
# 
# 
# ===== EPOCH 10/100 =====
# 
# TRAINING...
# average train loss 0.8685935193672776 
# 
# average train accuracy 87.76302083333334 
# 
# train auroc 95.48911446039193 
# 
# VALIDATION... 
# 
# val loss 1.1489767786115408 
# 
# val accuracy 86.07797475961539 
# 
# val auroc 92.93439874035354 
# 
# 
# ===== EPOCH 11/100 =====
# 
# TRAINING...
# average train loss 0.8391619207337498 
# 
# average train accuracy 87.88688151041666 
# 
# train auroc 95.59739569621517 
# 
# VALIDATION... 
# 
# val loss 1.1071446750313043 
# 
# val accuracy 85.13070913461539 
# 
# val auroc 93.04537643560285 
# 
# 
# ===== EPOCH 12/100 =====
# 
# TRAINING...
# average train loss 0.8139713773503899 
# 
# average train accuracy 88.06388346354166 
# 
# train auroc 95.73286186925343 
# 
# VALIDATION... 
# 
# val loss 1.071001411601901 
# 
# val accuracy 85.28771033653847 
# 
# val auroc 92.97938509102306 
# 
# 
# ===== EPOCH 13/100 =====
# 
# TRAINING...
# average train loss 0.7950809302739799 
# 
# average train accuracy 88.28181966145834 
# 
# train auroc 95.84979673821884 
# 
# VALIDATION... 
# 
# val loss 1.075434137135744 
# 
# val accuracy 86.74429086538461 
# 
# val auroc 93.08950936552371 
# 
# 
# ===== EPOCH 14/100 =====
# 
# TRAINING...
# average train loss 0.7829155928455294 
# 
# average train accuracy 88.42154947916666 
# 
# train auroc 95.88855806521742 
# 
# VALIDATION... 
# 
# val loss 1.0276039812713862 
# 
# val accuracy 85.67007211538461 
# 
# val auroc 93.09245684684704 
# 
# 
# ===== EPOCH 15/100 =====
# 
# TRAINING...
# average train loss 0.7706853214651346 
# 
# average train accuracy 88.38435872395834 
# 
# train auroc 95.95159066672942 
# 
# VALIDATION... 
# 
# val loss 1.0183360826224088 
# 
# val accuracy 84.81182391826924 
# 
# val auroc 93.08410321794159 
# 
# 
# ===== EPOCH 16/100 =====
# 
# TRAINING...
# average train loss 0.7574841019697487 
# 
# average train accuracy 88.54459635416666 
# 
# train auroc 96.04880369107906 
# 
# VALIDATION... 
# 
# val loss 1.0186433158814907 
# 
# val accuracy 86.10464242788461 
# 
# val auroc 93.08091843578183 
# 
# 
# ===== EPOCH 17/100 =====
# 
# TRAINING...
# average train loss 0.7539497355930507 
# 
# average train accuracy 88.6318359375 
# 
# train auroc 96.13459379857201 
# 
# VALIDATION... 
# 
# val loss 0.9876274101436138 
# 
# val accuracy 85.41128305288461 
# 
# val auroc 92.99318843926653 
# 
# 
# ===== EPOCH 18/100 =====
# 
# TRAINING...
# average train loss 0.7403068006969988 
# 
# average train accuracy 88.64314778645834 
# 
# train auroc 96.13300434924285 
# 
# VALIDATION... 
# 
# val loss 0.983150252327323 
# 
# val accuracy 84.89595853365384 
# 
# val auroc 92.98536950287962 
# 
# 
# ===== EPOCH 19/100 =====
# 
# TRAINING...
# average train loss 0.7376697989180684 
# 
# average train accuracy 88.832275390625 
# 
# train auroc 96.24670787228588 
# 
# VALIDATION... 
# 
# val loss 0.9844063654541969 
# 
# val accuracy 86.03891225961539 
# 
# val auroc 92.97402134083998 
# 
# 
# ===== EPOCH 20/100 =====
# 
# TRAINING...
# average train loss 0.7355908986181021 
# 
# average train accuracy 88.87508138020834 
# 
# train auroc 96.33114266062715 
# 
# VALIDATION... 
# 
# val loss 0.98568200096488 
# 
# val accuracy 85.66556490384616 
# 
# val auroc 93.03785144662136 
# 
# 
# ===== EPOCH 21/100 =====
# 
# TRAINING...
# average train loss 0.7243727809283882 
# 
# average train accuracy 88.99226888020834 
# 
# train auroc 96.34012057708091 
# 
# VALIDATION... 
# 
# val loss 0.969493393599987 
# 
# val accuracy 85.83045372596153 
# 
# val auroc 93.0954024280458 
# 
# 
# ===== EPOCH 22/100 =====
# 
# TRAINING...
# average train loss 0.7251381291076541 
# 
# average train accuracy 88.872314453125 
# 
# train auroc 96.36843322213295 
# 
# VALIDATION... 
# 
# val loss 0.9638275638222694 
# 
# val accuracy 85.087890625 
# 
# val auroc 92.89290116291625 
# 
# 
# ===== EPOCH 23/100 =====
# 
# TRAINING...
# average train loss 0.7219293764792383 
# 
# average train accuracy 89.0078125 
# 
# train auroc 96.41880057549311 
# 
# VALIDATION... 
# 
# val loss 0.9678030986338854 
# 
# val accuracy 84.90121694711539 
# 
# val auroc 93.05111496572684 
# 
# 
# ===== EPOCH 24/100 =====
# 
# TRAINING...
# average train loss 0.7174126482568681 
# 
# average train accuracy 89.08837890625 
# 
# train auroc 96.46630487520629 
# 
# VALIDATION... 
# 
# val loss 0.9773268196731806 
# 
# val accuracy 86.5673828125 
# 
# val auroc 93.03329567962302 
# 
# 
# ===== EPOCH 25/100 =====
# 
# TRAINING...
# average train loss 0.7128062684088945 
# 
# average train accuracy 88.89241536458334 
# 
# train auroc 96.40321656484457 
# 
# VALIDATION... 
# 
# val loss 0.9563564654439688 
# 
# val accuracy 86.26953125 
# 
# val auroc 93.10257945087865 
# 
# 
# ===== EPOCH 26/100 =====
# 
# TRAINING...
# average train loss 0.713762657251209 
# 
# average train accuracy 89.17862955729166 
# 
# train auroc 96.55371450605031 
# 
# VALIDATION... 
# 
# val loss 0.945347410812974 
# 
# val accuracy 85.1220703125 
# 
# val auroc 93.11676437627811 
# 
# 
# ===== EPOCH 27/100 =====
# 
# TRAINING...
# average train loss 0.7078421466052532 
# 
# average train accuracy 89.18513997395834 
# 
# train auroc 96.57791992063312 
# 
# VALIDATION... 
# 
# val loss 0.9442593943327665 
# 
# val accuracy 86.9140625 
# 
# val auroc 93.00299721514932 

# 
# 
# ===== EPOCH 1/1000 =====
# 
# TRAINING...
# average train loss 3.181918483041227 
# 
# average train accuracy 72.41495768229166 
# 
# train auroc 81.77045463083805 
# 
# Validation: 
# 
# val loss 2.9455404207110405 
# 
# val accuracy 75.49278846153847 
# 
# val auroc 89.55265155944036 
# 
# 
# ===== EPOCH 2/1000 =====
# 
# TRAINING...
# average train loss 2.2722906707786024 
# 
# average train accuracy 79.77351888020834 
# 
# train auroc 89.2077246180028 
# 
# Validation: 
# 
# val loss 2.231369490921497 
# 
# val accuracy 81.88626802884616 
# 
# val auroc 91.3528680664846 
# 
# 
# ===== EPOCH 3/1000 =====
# 
# TRAINING...
# average train loss 1.5231520949862898 
# 
# average train accuracy 82.34488932291666 
# 
# train auroc 91.16963898423226 
# 
# Validation: 
# 
# val loss 1.6231873612850904 
# 
# val accuracy 79.27771935096153 
# 
# val auroc 91.83784077283626 
# 
# 
# ===== EPOCH 4/1000 =====
# 
# TRAINING...
# average train loss 1.054814766254276 
# 
# average train accuracy 84.03800455729166 
# 
# train auroc 92.55807305392413 
# 
# Validation: 
# 
# val loss 1.2231923297047615 
# 
# val accuracy 82.76404747596153 
# 
# val auroc 92.20339161830104 
# 
# 
# ===== EPOCH 5/1000 =====
# 
# TRAINING...
# average train loss 0.8040548083838075 
# 
# average train accuracy 85.154296875 
# 
# train auroc 93.51093411606189 
# 
# Validation: 
# 
# val loss 1.004143350943923 
# 
# val accuracy 82.96536959134616 
# 
# val auroc 92.56264948643306 
# 
# 
# ===== EPOCH 6/1000 =====
# 
# TRAINING...
# average train loss 0.6625897470396012 
# 
# average train accuracy 85.814208984375 
# 
# train auroc 94.0845115775122 
# 
# Validation: 
# 
# val loss 0.8581075862050056 
# 
# val accuracy 85.1953125 
# 
# val auroc 92.71917850406523 
# 
# 
# ===== EPOCH 7/1000 =====
# 
# TRAINING...
# average train loss 0.5825664245523512 
# 
# average train accuracy 86.35555013020834 
# 
# train auroc 94.43436422818195 
# 
# Validation: 
# 
# val loss 0.7590165045112371 
# 
# val accuracy 83.53140024038461 
# 
# val auroc 92.85257582823306 
# 
# 
# ===== EPOCH 8/1000 =====
# 
# TRAINING...
# average train loss 0.5265569990966469 
# 
# average train accuracy 86.63590494791666 
# 
# train auroc 94.73046953487912 
# 
# Validation: 
# 
# val loss 0.6927711300551891 
# 
# val accuracy 84.30401141826924 
# 
# val auroc 92.95101259864481 
# 
# 
# ===== EPOCH 9/1000 =====
# 
# TRAINING...
# average train loss 0.48762516509741544 
# 
# average train accuracy 87.07145182291666 
# 
# train auroc 95.01078513641708 
# 
# Validation: 
# 
# val loss 0.6653395831584931 
# 
# val accuracy 86.22182992788461 
# 
# val auroc 92.96734487732125 
# 
# 
# ===== EPOCH 10/1000 =====
# 
# TRAINING...
# average train loss 0.4598561291582882 
# 
# average train accuracy 87.32568359375 
# 
# train auroc 95.18461865873992 
# 
# Validation: 
# 
# val loss 0.6158743711188436 
# 
# val accuracy 84.521484375 
# 
# val auroc 92.98974332068234 
# 
# 
# ===== EPOCH 11/1000 =====
# 
# TRAINING...
# average train loss 0.4390855747275054 
# 
# average train accuracy 87.44197591145834 
# 
# train auroc 95.32039772778005 
# 
# Validation: 
# 
# val loss 0.5946464248001575 
# 
# val accuracy 86.30746694711539 
# 
# val auroc 93.0005767529498 
# 
# 
# ===== EPOCH 12/1000 =====
# 
# TRAINING...
# average train loss 0.4215274780988693 
# 
# average train accuracy 87.68025716145834 
# 
# train auroc 95.49176191599904 
# 
# Validation: 
# 
# val loss 0.5862127792090177 
# 
# val accuracy 83.72821514423076 
# 
# val auroc 93.08907824523291 
# 
# 
# ===== EPOCH 13/1000 =====
# 
# TRAINING...
# average train loss 0.41015294203534725 
# 
# average train accuracy 87.8701171875 
# 
# train auroc 95.58831999918836 
# 
# Validation: 
# 
# val loss 0.559299198165536 
# 
# val accuracy 85.0341796875 
# 
# val auroc 92.93900318818106 
# 
# 
# ===== EPOCH 14/1000 =====
# 
# TRAINING...
# average train loss 0.39833726214710624 
# 
# average train accuracy 88.10847981770834 
# 
# train auroc 95.76443108375777 
# 
# Validation: 
# 
# val loss 0.5422922195866704 
# 
# val accuracy 84.17029747596153 
# 
# val auroc 93.02862158027499 
# 
# 
# ===== EPOCH 15/1000 =====
# 
# TRAINING...
# average train loss 0.3912767786066979 
# 
# average train accuracy 88.2412109375 
# 
# train auroc 95.86745027481811 
# 
# Validation: 
# 
# val loss 0.5826973799616099 
# 
# val accuracy 85.70763221153847 
# 
# val auroc 92.96095446302637 
# 
# 
# ===== EPOCH 16/1000 =====
# 
# TRAINING...
# average train loss 0.38459239252842964 
# 
# average train accuracy 88.23046875 
# 
# train auroc 95.92409771781642 
# 
# Validation: 
# 
# val loss 0.5501762714236975 
# 
# val accuracy 85.85824819711539 
# 
# val auroc 93.06482094729584 
# 
# 
# ===== EPOCH 17/1000 =====
# 
# TRAINING...
# average train loss 0.37980829291045665 
# 
# average train accuracy 88.33610026041666 
# 
# train auroc 95.94492036089072 
# 
# Validation: 
# 
# val loss 0.5570077972486616 
# 
# val accuracy 85.5908203125 
# 
# val auroc 93.09422619926157 
# 
# 
# ===== EPOCH 18/1000 =====
# 
# TRAINING...
# average train loss 0.37357282494194805 
# 
# average train accuracy 88.52010091145834 
# 
# train auroc 96.11572614892745 
# 
# Validation: 
# 
# val loss 0.5468578346073627 
# 
# val accuracy 86.318359375 
# 
# val auroc 93.13726627907582 
# 
# 
# ===== EPOCH 19/1000 =====
# 
# TRAINING...
# average train loss 0.3706562525359914 
# 
# average train accuracy 88.48933919270834 
# 
# train auroc 96.11131010849728 
# 
# Validation: 
# 
# val loss 0.5493077794089913 
# 
# val accuracy 85.20244891826924 
# 
# val auroc 93.08524263343155 
# 
# 
# ===== EPOCH 20/1000 =====
# 
# TRAINING...
# average train loss 0.3684535296401009 
# 
# average train accuracy 88.630615234375 
# 
# train auroc 96.12277153578471 
# 
# Validation: 
# 
# val loss 0.5276825470849872 
# 
# val accuracy 87.0166015625 
# 
# val auroc 93.07276658691774 
# 
# 
# ===== EPOCH 21/1000 =====
# 
# TRAINING...
# average train loss 0.36458208723925056 
# 
# average train accuracy 88.62418619791666 
# 
# train auroc 96.21292339648375 
# 
# Validation: 
# 
# val loss 0.5546149183064699 
# 
# val accuracy 84.541015625 
# 
# val auroc 93.00481170705439 
# 
# 
# ===== EPOCH 22/1000 =====
# 
# TRAINING...
# average train loss 0.3605312699917704 
# 
# average train accuracy 88.75602213541666 
# 
# train auroc 96.25247412090225 
# 
# Validation: 
# 
# val loss 0.5322434617206454 
# 
# val accuracy 84.931640625 
# 
# val auroc 93.07488059244818 
# 
# 
# ===== EPOCH 23/1000 =====
# 
# TRAINING...
# average train loss 0.35799267566762866 
# 
# average train accuracy 88.811279296875 
# 
# train auroc 96.29777321891578 
# 
# Validation: 
# 
# val loss 0.5365511542186141 
# 
# val accuracy 85.9033203125 
# 
# val auroc 93.10735972156088 
# 
# 
# ===== EPOCH 24/1000 =====
# 
# TRAINING...
# average train loss 0.35595794310793283 
# 
# average train accuracy 88.88151041666666 
# 
# train auroc 96.38758779958299 
# 
# Validation: 
# 
# val loss 0.5506164571270347 
# 
# val accuracy 86.00961538461539 
# 
# val auroc 93.19545442760686 
# 
# 
# ===== EPOCH 25/1000 =====
# 
# TRAINING...
# average train loss 0.35312515338882805 
# 
# average train accuracy 88.92439778645834 
# 
# train auroc 96.40976291965583 
# 
# Validation: 
# 
# val loss 0.5291100081056357 
# 
# val accuracy 86.31573016826924 
# 
# val auroc 93.10437815232234 
# 
# 
# ===== EPOCH 26/1000 =====
# 
# TRAINING...
# average train loss 0.3510746824555099 
# 
# average train accuracy 88.880859375 
# 
# train auroc 96.43318837661077 
# 
# Validation: 
# 
# val loss 0.5443421211093664 
# 
# val accuracy 85.224609375 
# 
# val auroc 93.10966741538742 
# 
# 
# ===== EPOCH 27/1000 =====
# 
# TRAINING...
# average train loss 0.3497388619463891 
# 
# average train accuracy 88.9970703125 
# 
# train auroc 96.4812627647919 
# 
# Validation: 
# 
# val loss 0.5401364017277956 
# 
# val accuracy 86.0205078125 
# 
# val auroc 93.08433076915878 
# 
# 
# ===== EPOCH 28/1000 =====
# 
# TRAINING...
# average train loss 0.3473430474288762 
# 
# average train accuracy 88.98795572916666 
# 
# train auroc 96.5137147944471 
# 
# Validation: 
# 
# val loss 0.5319803100079298 
# 
# val accuracy 85.59457632211539 
# 
# val auroc 93.11803044078879 
# 
# 
# ===== EPOCH 29/1000 =====
# 
# TRAINING...
# average train loss 0.34579292852431537 
# 
# average train accuracy 89.0810546875 
# 
# train auroc 96.5798828687463 
# 
# Validation: 
# 
# val loss 0.5363170266151428 
# 
# val accuracy 86.0791015625 
# 
# val auroc 93.04939773106888 
# 
# 
# ===== EPOCH 30/1000 =====
# 
# TRAINING...
# average train loss 0.3439643070567399 
# 
# average train accuracy 89.18204752604166 
# 
# train auroc 96.6145110891625 
# 
# Validation: 
# 
# val loss 0.5627546571195126 
# 
# val accuracy 84.97671274038461 
# 
# val auroc 93.07163795349595 
# 
# 
# ===== EPOCH 31/1000 =====
# 
# TRAINING...
# average train loss 0.3433305963408202 
# 
# average train accuracy 89.106689453125 
# 
# train auroc 96.64471126351033 
# 
# Validation: 
# 
# val loss 0.544455293007195 
# 
# val accuracy 87.12515024038461 
# 
# val auroc 93.12295044394847 
# 
# 
# ===== EPOCH 32/1000 =====
# 
# TRAINING...
# average train loss 0.3416286462685093 
# 
# average train accuracy 89.22542317708334 
# 
# train auroc 96.64847276416722 
# 
# Validation: 
# 
# val loss 0.546025444380939 
# 
# val accuracy 86.17563100961539 
# 
# val auroc 93.04188477505903 
# 
# 
# ===== EPOCH 33/1000 =====
# 
# TRAINING...
# average train loss 0.3423853023443371 
# 
# average train accuracy 89.26505533854166 
# 
# train auroc 96.70064353079468 
# 
# Validation: 
# 
# val loss 0.5503353421576321 
# 
# val accuracy 86.52193509615384 
# 
# val auroc 93.05819339077395 
# 
# 
# ===== EPOCH 34/1000 =====
# 
# TRAINING...
# average train loss 0.34169039721600714 
# 
# average train accuracy 89.17838541666666 
# 
# train auroc 96.6712074580738 
# 
# Validation: 
# 
# val loss 0.5412569150328637 
# 
# val accuracy 86.62334735576924 
# 
# val auroc 93.02080146784648 
# 
# 
# ===== EPOCH 35/1000 =====
# 
# TRAINING...
# average train loss 0.3394709460204467 
# 
# average train accuracy 89.34684244791666 
# 
# train auroc 96.7011719612077 
# 
# Validation: 
# 
# val loss 0.5637341135181486 
# 
# val accuracy 86.5576171875 
# 
# val auroc 93.02838806793554 
# 
# 
# ===== EPOCH 36/1000 =====
# 
# TRAINING...
# average train loss 0.33863693121820687 
# 
# average train accuracy 89.324951171875 
# 
# train auroc 96.71978662030803 
# 
# Validation: 
# 
# val loss 0.5542684190906584 
# 
# val accuracy 84.73858173076924 
# 
# val auroc 92.96556645139195 
# 
# 
# ===== EPOCH 37/1000 =====
# 
# TRAINING...
# average train loss 0.3355771582573652 
# 
# average train accuracy 89.36702473958334 
# 
# train auroc 96.80852745421447 
# 
# Validation: 
# 
# val loss 0.5441758504137397 
# 
# val accuracy 84.8681640625 
# 
# val auroc 93.02587137726483 
# 
# 
# ===== EPOCH 38/1000 =====
# 
# TRAINING...
# average train loss 0.33352820910513403 
# 
# average train accuracy 89.47932942708334 
# 
# train auroc 96.885451005145 
# 
# Validation: 
# 
# val loss 0.5531829856336117 
# 
# val accuracy 85.9619140625 
# 
# val auroc 93.01401564626323 
# 
# 
# ===== EPOCH 39/1000 =====
# 
# TRAINING...
# average train loss 0.3355723779415712 
# 
# average train accuracy 89.37467447916666 
# 
# train auroc 96.79751979164568 
# 
# Validation: 
# 
# val loss 0.5680099285207689 
# 
# val accuracy 86.14107572115384 
# 
# val auroc 93.09355408660615 
# 
# 
# ===== EPOCH 40/1000 =====
# 
# TRAINING...
# average train loss 0.3329955929191783 
# 
# average train accuracy 89.529052734375 
# 
# train auroc 96.8615685402265 
# 
# Validation: 
# 
# val loss 0.5691265366971493 
# 
# val accuracy 86.89077524038461 
# 
# val auroc 93.09324065500803 
# 
# 
# ===== EPOCH 41/1000 =====
# 
# TRAINING...
# average train loss 0.3325238443445414 
# 
# average train accuracy 89.42252604166666 
# 
# train auroc 96.87056525242059 
# 
# Validation: 
# 
# val loss 0.564812406245619 
# 
# val accuracy 86.50615985576924 
# 
# val auroc 93.06654630961005 
# 
# 
# ===== EPOCH 42/1000 =====
# 
# TRAINING...
# average train loss 0.33304560934193433 
# 
# average train accuracy 89.43717447916666 
# 
# train auroc 96.8860272680719 
# 
# Validation: 
# 
# val loss 0.5565043611451983 
# 
# val accuracy 85.96792367788461 
# 
# val auroc 93.00561708650854 
# 
# 
# ===== EPOCH 43/1000 =====
# 
# TRAINING...
# average train loss 0.33081073996145277 
# 
# average train accuracy 89.63273111979166 
# 
# train auroc 96.92329494928487 
# 
# Validation: 
# 
# val loss 0.5686477957293391 
# 
# val accuracy 86.04116586538461 
# 
# val auroc 93.01856162194811 
# 
# 
# ===== EPOCH 44/1000 =====
# 
# TRAINING...
# average train loss 0.3310036863433197 
# 
# average train accuracy 89.48909505208334 
# 
# train auroc 96.92596992826331 
# 
# Validation: 
# 
# val loss 0.5552703965455293 
# 
# val accuracy 85.75796274038461 
# 
# val auroc 93.04794586467293 
# 
# 
# ===== EPOCH 45/1000 =====
# 
# TRAINING...
# average train loss 0.33157618464902044 
# 
# average train accuracy 89.599853515625 
# 
# train auroc 96.9000033373954 
# 
# Validation: 
# 
# val loss 0.5710386069491505 
# 
# val accuracy 85.8740234375 
# 
# val auroc 93.01221472778953 
# 
# 
# ===== EPOCH 46/1000 =====
# 
# TRAINING...
# average train loss 0.32905763254966586 
# 
# average train accuracy 89.46736653645834 
# 
# train auroc 96.9523988193733 
# 
# Validation: 
# 
# val loss 0.5585635730996727 
# 
# val accuracy 85.19794170673076 
# 
# val auroc 93.0611767473553 
# 
# 
# ===== EPOCH 47/1000 =====
# 
# TRAINING...
# average train loss 0.3277115087024868 
# 
# average train accuracy 89.59407552083334 
# 
# train auroc 97.000636181417 
# 
# Validation: 
# 
# val loss 0.5592231377959251 
# 
# val accuracy 85.7861328125 
# 
# val auroc 92.99166032488657 
# 
# 
# ===== EPOCH 48/1000 =====
# 
# TRAINING...
# average train loss 0.3278312019072473 
# 
# average train accuracy 89.65421549479166 
# 
# train auroc 97.0354748458935 
# 
# Validation: 
# 
# val loss 0.567232848610729 
# 
# val accuracy 85.73016826923076 
# 
# val auroc 92.99840310633375 
# 
# 
# ===== EPOCH 49/1000 =====
# 
# TRAINING...
# average train loss 0.32703209281899037 
# 
# average train accuracy 89.708251953125 
# 
# train auroc 97.00222918541652 
# 
# Validation: 
# 
# val loss 0.5773732451722026 
# 
# val accuracy 85.64227764423076 
# 
# val auroc 92.97951884077177 
# 
# 
# ===== EPOCH 50/1000 =====
# 
# TRAINING...
# average train loss 0.3260296107735485 
# 
# average train accuracy 89.71915690104166 
# 
# train auroc 97.05292719350783 
# 
# Validation: 
# 
# val loss 0.5757073823362588 
# 
# val accuracy 86.29018930288461 
# 
# val auroc 93.07157881934621 
# 
# 
# ===== EPOCH 51/1000 =====
# 
# TRAINING...
# average train loss 0.3270888681989163 
# 
# average train accuracy 89.544921875 
# 
# train auroc 97.02063399378798 
# 
# Validation: 
# 
# val loss 0.5603968623094261 
# 
# val accuracy 86.46221454326924 
# 
# val auroc 93.05187717355776 
# 
# 
# ===== EPOCH 52/1000 =====
# 
# TRAINING...
# average train loss 0.324737157067284 
# 
# average train accuracy 89.70670572916666 
# 
# train auroc 97.08139765732513 
# 
# Validation: 
# 
# val loss 0.5714462288655341 
# 
# val accuracy 85.47438401442307 
# 
# val auroc 93.01909150961183 
# 
# 
# ===== EPOCH 53/1000 =====
# 
# TRAINING...
# average train loss 0.32586775603704154 
# 
# average train accuracy 89.66853841145834 
# 
# train auroc 97.07861662315788 
# 
# Validation: 
# 
# val loss 0.5925700917840004 
# 
# val accuracy 86.05806790865384 
# 
# val auroc 93.01256670190753 
# 
# 
# ===== EPOCH 54/1000 =====
# 
# TRAINING...
# average train loss 0.3252200039336458 
# 
# average train accuracy 89.77986653645834 
# 
# train auroc 97.08409413381752 
# 
# Validation: 
# 
# val loss 0.5734268130734563 
# 
# val accuracy 85.83984375 
# 
# val auroc 92.92089602992728 
# 
# 
# ===== EPOCH 55/1000 =====
# 
# TRAINING...
# average train loss 0.3230402205139399 
# 
# average train accuracy 89.669677734375 
# 
# train auroc 97.1010900724279 
# 
# Validation: 
# 
# val loss 0.5728195735253394 
# 
# val accuracy 86.08022836538461 
# 
# val auroc 93.00157974661651 
# 
# 
# ===== EPOCH 56/1000 =====
# 
# TRAINING...
# average train loss 0.3234122763387859 
# 
# average train accuracy 89.687744140625 
# 
# train auroc 97.10061951701176 
# 
# Validation: 
# 
# val loss 0.5565741027705371 
# 
# val accuracy 85.6689453125 
# 
# val auroc 93.00199718680459 
# 
# 
# ===== EPOCH 57/1000 =====
# 
# TRAINING...
# average train loss 0.32301813950762154 
# 
# average train accuracy 89.70760091145834 
# 
# train auroc 97.10301830852801 
# 
# Validation: 
# 
# val loss 0.5804141668602825 
# 
# val accuracy 86.23497596153847 
# 
# val auroc 92.99108974823604 
# 
# 
# ===== EPOCH 58/1000 =====
# 
# TRAINING...
# average train loss 0.32144541554152967 
# 
# average train accuracy 89.68562825520834 
# 
# train auroc 97.14402994730446 
# 
# Validation: 
# 
# val loss 0.5845622372813523 
# 
# val accuracy 86.44268329326924 
# 
# val auroc 93.0327178790238 
# 
# 
# ===== EPOCH 59/1000 =====
# 
# TRAINING...
# average train loss 0.3220333959441632 
# 
# average train accuracy 89.76529947916666 
# 
# train auroc 97.1327331709029 
# 
# Validation: 
# 
# val loss 0.5917966607026756 
# 
# val accuracy 85.13408954326924 
# 
# val auroc 93.02461047718057 
# 
# 
# ===== EPOCH 60/1000 =====
# 
# TRAINING...
# average train loss 0.32162060725968333 
# 
# average train accuracy 89.756103515625 
# 
# train auroc 97.15302320793941 
# 
# Validation: 
# 
# val loss 0.5987750120460987 
# 
# val accuracy 86.68682391826924 
# 
# val auroc 93.07442805004023 
# 
# 
# ===== EPOCH 61/1000 =====
# 
# TRAINING...
# average train loss 0.3219513637246564 
# 
# average train accuracy 89.80891927083334 
# 
# train auroc 97.16467526167929 
# 
# Validation: 
# 
# val loss 0.5754726411774754 
# 
# val accuracy 86.04604867788461 
# 
# val auroc 93.06434340694894 
# 
# 
# ===== EPOCH 62/1000 =====
# 
# TRAINING...
# average train loss 0.31979463559109716 
# 
# average train accuracy 89.91544596354166 
# 
# train auroc 97.21137049425279 
# 
# Validation: 
# 
# val loss 0.5893440749496222 
# 
# val accuracy 86.02313701923076 
# 
# val auroc 92.99076592504349 
# 
# 
# ===== EPOCH 63/1000 =====
# 
# TRAINING...
# average train loss 0.32160020333249123 
# 
# average train accuracy 89.76123046875 
# 
# train auroc 97.17525534987224 
# 
# Validation: 
# 
# val loss 0.5759924566373229 
# 
# val accuracy 86.18051382211539 
# 
# val auroc 93.05084070260973 

# 
# 
# ===== EPOCH 1/1000 =====
# 
# TRAINING...
# average train loss 9.481754519045353
# average train accuracy 81.16498343552215
# train auroc 0.753458504904206
# val loss 9.4788694024086
# val accuracy 79.63783001077586
# val auroc 0.6150658478496972
# 
# ===== EPOCH 2/1000 =====
# 
# TRAINING...
# average train loss 8.867210291326046
# average train accuracy 83.67824119857595
# train auroc 0.8316177423644291
# val loss 9.08855335712433
# val accuracy 79.62806438577586
# val auroc 0.6071255206144335
# 
# ===== EPOCH 3/1000 =====
# 
# TRAINING...
# average train loss 8.220698601007461
# average train accuracy 85.03170737737341
# train auroc 0.8565567567616771
# val loss 8.558806794881821
# val accuracy 79.68228044181035
# val auroc 0.6049639407962188
# 
# ===== EPOCH 4/1000 =====
# 
# TRAINING...
# average train loss 7.42860289439559
# average train accuracy 85.86132194422468
# train auroc 0.8715864604619469
# val loss 7.811908936500549
# val accuracy 79.44841056034483
# val auroc 0.5963582062359268
# 
# ===== EPOCH 5/1000 =====
# 
# TRAINING...
# average train loss 6.525110354274512
# average train accuracy 86.38285576542721
# train auroc 0.8811247144983552
# val loss 7.023380476236343
# val accuracy 79.56997575431035
# val auroc 0.5855718774637355
# 
# ===== EPOCH 6/1000 =====
# 
# TRAINING...
# average train loss 5.628008701652289
# average train accuracy 87.05458242681962
# train auroc 0.8928177141749808
# val loss 6.204821765422821
# val accuracy 79.47720231681035
# val auroc 0.5753850289136936
# 
# ===== EPOCH 7/1000 =====
# 
# TRAINING...
# average train loss 4.841287352144718
# average train accuracy 87.46225091475475
# train auroc 0.900209986557585
# val loss 5.520470979809761
# val accuracy 79.11536907327586
# val auroc 0.5561028365179552
# 
# ===== EPOCH 8/1000 =====
# 
# TRAINING...
# average train loss 4.188622269779444
# average train accuracy 87.88827630537975
# train auroc 0.9066700218758619
# val loss 4.9845357716083525
# val accuracy 79.62368669181035
# val auroc 0.5250368887663506
# 
# ===== EPOCH 9/1000 =====
# 
# TRAINING...
# average train loss 3.6847474955022337
# average train accuracy 88.29813896855221
# train auroc 0.9108412679866881
# val loss 4.435713562369346
# val accuracy 77.79936691810344
# val auroc 0.5583485980386673
# 
# ===== EPOCH 10/1000 =====
# 
# TRAINING...
# average train loss 3.3083773121237754
# average train accuracy 88.38068322290349
# train auroc 0.9150712394825586
# val loss 4.110553869605065
# val accuracy 78.76767914870689
# val auroc 0.5420875039810463
# 
# ===== EPOCH 11/1000 =====
# 
# TRAINING...
# average train loss 3.012832838669419
# average train accuracy 88.77892602848101
# train auroc 0.9189245528567964
# val loss 3.8833645969629287
# val accuracy 78.31357758620689
# val auroc 0.531293819660552
# 
# ===== EPOCH 12/1000 =====
# 
# TRAINING...
# average train loss 2.7996057186275722
# average train accuracy 88.74718774723101
# train auroc 0.919557541122521
# val loss 3.6805032223463057
# val accuracy 77.14456492456897
# val auroc 0.5573460312195202
# 
# ===== EPOCH 13/1000 =====
# 
# TRAINING...
# average train loss 2.643998497724533
# average train accuracy 88.77884876878956
# train auroc 0.9210747967750088
# val loss 3.555765315890312
# val accuracy 77.33112203663794
# val auroc 0.5614955464829091
# 
# ===== EPOCH 14/1000 =====
# 
# TRAINING...
# average train loss 2.5386668875813485
# average train accuracy 88.95167869857595
# train auroc 0.921383276233426
# val loss 3.4907029062509536
# val accuracy 77.02400996767241
# val auroc 0.5429025988064504
# 
# ===== EPOCH 15/1000 =====
# 
# TRAINING...
# average train loss 2.4639359496533872
# average train accuracy 89.04003288172468
# train auroc 0.9225248804245421
# val loss 3.470598095655441
# val accuracy 78.31256734913794
# val auroc 0.5323486371013996
# 
# ===== EPOCH 16/1000 =====
# 
# TRAINING...
# average train loss 2.4143267307430505
# average train accuracy 89.12287072290349
# train auroc 0.9217560955773199
# val loss 3.4337639272212983
# val accuracy 77.34139278017241
# val auroc 0.5307609669057854
# 
# ===== EPOCH 17/1000 =====
# 
# TRAINING...
# average train loss 2.3754481628537176
# average train accuracy 89.25778159612341
# train auroc 0.923764157744041
# val loss 3.480948919057846
# val accuracy 78.46342941810344
# val auroc 0.5323262030582114
# 
# ===== EPOCH 18/1000 =====
# 
# TRAINING...
# average train loss 2.353608475998044
# average train accuracy 89.16622886174841
# train auroc 0.9248707222629056
# val loss 3.449778100848198
# val accuracy 75.43221309267241
# val auroc 0.5464484234861965
# 
# ===== EPOCH 19/1000 =====
# 
# TRAINING...
# average train loss 2.34249037951231
# average train accuracy 89.35143579410601
# train auroc 0.9226330438674661
# val loss 3.468599873781204
# val accuracy 75.76862203663794
# val auroc 0.5470553791914693
# 
# ===== EPOCH 20/1000 =====
# 
# TRAINING...
# average train loss 2.339990671351552
# average train accuracy 89.39647819422468
# train auroc 0.9233432142084009
# val loss 3.514880421757698
# val accuracy 72.91453394396551
# val auroc 0.5518976948994214
# 
# ===== EPOCH 21/1000 =====
# 
# TRAINING...
# average train loss 2.333827419579029
# average train accuracy 89.40578026107595
# train auroc 0.9244596355433297
# val loss 3.574315017461777
# val accuracy 74.56105199353449
# val auroc 0.5425027304132903
# 
# ===== EPOCH 22/1000 =====
# 
# TRAINING...
# average train loss 2.354543152824044
# average train accuracy 89.55348533920095
# train auroc 0.9247499074748454
# val loss 3.6123420923948286
# val accuracy 76.12944504310344
# val auroc 0.5349073706458475
# 
# ===== EPOCH 23/1000 =====
# 
# TRAINING...
# average train loss 2.3688554599881173
# average train accuracy 89.51256860660601
# train auroc 0.9240606885823197
# val loss 3.6474969744682313
# val accuracy 77.46295797413794
# val auroc 0.5255022814961472
# 
# ===== EPOCH 24/1000 =====
# 
# TRAINING...
# average train loss 2.368288476392627
# average train accuracy 89.55453607100475
# train auroc 0.9234172684521201
# val loss 3.672994527220726
# val accuracy 75.82384832974137
# val auroc 0.5415237015305328
# 
# ===== EPOCH 25/1000 =====
# 
# TRAINING...
# average train loss 2.3967555101960896
# average train accuracy 89.63615320905855
# train auroc 0.9230589772752644
# val loss 3.723589673638344
# val accuracy 75.50596039870689
# val auroc 0.5456524615806776
# 
# ===== EPOCH 26/1000 =====
# 
# TRAINING...
# average train loss 2.4104739874601364
# average train accuracy 89.66120080102849
# train auroc 0.9228850712399075
# val loss 3.7977636605501175
# val accuracy 75.63678609913794
# val auroc 0.5504357115157362
# 
# ===== EPOCH 27/1000 =====
# 
# TRAINING...
# average train loss 2.458819505944848
# average train accuracy 89.58446647547468
# train auroc 0.9215208884634085
# val loss 3.867260479927063
# val accuracy 76.53623383620689
# val auroc 0.5419534585563776
# 
# ===== EPOCH 28/1000 =====
# 
# TRAINING...
# average train loss 2.4953979846090077
# average train accuracy 89.60412134098101
# train auroc 0.9238506176053637
# val loss 3.8869843989610673
# val accuracy 76.31111934267241
# val auroc 0.5406548201135999
# 
# ===== EPOCH 29/1000 =====
# 
# TRAINING...
# average train loss 2.5272481471300123
# average train accuracy 89.77577692345729
# train auroc 0.9244545804032281
# val loss 3.922753873467445
# val accuracy 71.91305226293103
# val auroc 0.5632648544806796
# 
# ===== EPOCH 30/1000 =====
# 
# TRAINING...
# average train loss 2.5545787941664457
# average train accuracy 89.65509728540349
# train auroc 0.9218728340135278
# val loss 3.9473690032958983
# val accuracy 73.22265625
# val auroc 0.5735847544731403
# 
# ===== EPOCH 31/1000 =====
# 
# TRAINING...
# average train loss 2.5818813420832156
# average train accuracy 89.73431937302215
# train auroc 0.9231101985602421
# val loss 4.0294332832098005
# val accuracy 74.29350754310344
# val auroc 0.5648625519892312
# 
# ===== EPOCH 32/1000 =====
# 
# TRAINING...
# average train loss 2.615898463502526
# average train accuracy 89.76920984968355
# train auroc 0.9225580025936854
# val loss 4.060393694043159
# val accuracy 70.95652613146551
# val auroc 0.5618304266976383
# 
# ===== EPOCH 33/1000 =====
# 
# TRAINING...
# average train loss 2.6361778419464827
# average train accuracy 89.77257837223101
# train auroc 0.9220671782744667
# val loss 4.1280857026577
# val accuracy 73.76077586206897
# val auroc 0.5426902999609068
# 
# ===== EPOCH 34/1000 =====
# 
# TRAINING...
# average train loss 2.6650786701589824
# average train accuracy 89.86000543908229
# train auroc 0.9197148671127385
# val loss 4.187872233986854
# val accuracy 77.73151266163794
# val auroc 0.5124468017904489
# 
# ===== EPOCH 35/1000 =====
# 
# TRAINING...
# average train loss 2.6923706149682403
# average train accuracy 89.57482446598101
# train auroc 0.920066788890104
# val loss 4.284627702832222
# val accuracy 65.27427936422414
# val auroc 0.586632711524269
# 
# ===== EPOCH 36/1000 =====
# 
# TRAINING...
# average train loss 2.7118937436491253
# average train accuracy 89.79303673852849
# train auroc 0.9202089927515512
# val loss 4.2930000722408295
# val accuracy 73.23444234913794
# val auroc 0.5656385660634761
# 
# ===== EPOCH 37/1000 =====
# 
# TRAINING...
# average train loss 2.7200013417750597
# average train accuracy 89.95253164556962
# train auroc 0.9193155928966714
# val loss 4.371257311105728
# val accuracy 68.93015894396551
# val auroc 0.5779103438557136
# 
# ===== EPOCH 38/1000 =====
# 
# TRAINING...
# average train loss 2.7621961906552315
# average train accuracy 89.72956017602849
# train auroc 0.9194467063722402
# val loss 4.4228099405765535
# val accuracy 67.80845905172414
# val auroc 0.5828280513519379
# 
# ===== EPOCH 39/1000 =====
# 
# TRAINING...
# average train loss 2.783096028864384
# average train accuracy 89.90685571598101
# train auroc 0.9200922161230028
# val loss 4.444959723949433
# val accuracy 67.19036233836206
# val auroc 0.5820231865529492
# 
# ===== EPOCH 40/1000 =====
# 
# TRAINING...
# average train loss 2.8214302031323313
# average train accuracy 89.81837791732595
# train auroc 0.9178204366718479
# val loss 4.479381507635116
# val accuracy 75.98548626077586
# val auroc 0.5449070538604046
# 
# ===== EPOCH 41/1000 =====
# 
# TRAINING...
# average train loss 2.8578812194988132
# average train accuracy 89.86947747725475
# train auroc 0.9173815399821607
# val loss 4.549116471409798
# val accuracy 76.33704876077586
# val auroc 0.5329803739804242
# 
# ===== EPOCH 42/1000 =====
# 
# TRAINING...
# average train loss 2.892747741006315
# average train accuracy 89.87360314477849
# train auroc 0.917656247957255
# val loss 4.58711573779583
# val accuracy 72.45319234913794
# val auroc 0.5582026225511514
# 
# ===== EPOCH 43/1000 =====
# 
# TRAINING...
# average train loss 2.9306315096095203
# average train accuracy 89.90319360660601
# train auroc 0.9152702949375637
# val loss 4.677428275346756
# val accuracy 73.65537446120689
# val auroc 0.5523844003975601
# 
# ===== EPOCH 44/1000 =====
# 
# TRAINING...
# average train loss 2.9720868250355124
# average train accuracy 89.7642497774921
# train auroc 0.9148857176523795
# val loss 4.691912677884102
# val accuracy 72.12604391163794
# val auroc 0.5489586296952711
# 
# ===== EPOCH 45/1000 =====
# 
# TRAINING...
# average train loss 3.003776434622705
# average train accuracy 89.87941307357595
# train auroc 0.914565265080362
# val loss 4.701113268733025
# val accuracy 72.55084859913794
# val auroc 0.5642371462272904
# 
# ===== EPOCH 46/1000 =====
# 
# TRAINING...
# average train loss 3.0446439858525993
# average train accuracy 89.88214806665349
# train auroc 0.9154117684008435
# val loss 4.724443325400353
# val accuracy 73.3740234375
# val auroc 0.5651856230148936
# 
# ===== EPOCH 47/1000 =====
# 
# TRAINING...
# average train loss 3.068920763023198
# average train accuracy 89.87848595727849
# train auroc 0.9138456272641617
# val loss 4.773574990034104
# val accuracy 75.65092941810344
# val auroc 0.5455049526604789
# 
# ===== EPOCH 48/1000 =====
# 
# TRAINING...
# average train loss 3.1076545625925065
# average train accuracy 89.90912715090981
# train auroc 0.9111624211840299
# val loss 4.820995852351189
# val accuracy 76.18366109913794
# val auroc 0.5326918840187868
# 
# ===== EPOCH 49/1000 =====
# 
# TRAINING...
# average train loss 3.1352929221466184
# average train accuracy 89.81273795984968
# train auroc 0.9085502866806026
# val loss 4.87804861664772
# val accuracy 68.93302128232759
# val auroc 0.5797937667808344
# 
# ===== EPOCH 50/1000 =====
# 
# TRAINING...
# average train loss 3.151287997700274
# average train accuracy 89.85326839398735
# train auroc 0.9092004152584844
# val loss 4.926478970050812
# val accuracy 74.54842403017241
# val auroc 0.5345391403840749
# 
# ===== EPOCH 51/1000 =====
# 
# TRAINING...
# average train loss 3.1926820315420628
# average train accuracy 89.93097619165349
# train auroc 0.9057106150972573
# val loss 4.992599755525589
# val accuracy 71.24949488146551
# val auroc 0.5537692560338949
# 
# ===== EPOCH 52/1000 =====
# 
# TRAINING...
# average train loss 3.217565974779427
# average train accuracy 89.91300558742088
# train auroc 0.9053751304452377
# val loss 5.015929561853409
# val accuracy 77.45858028017241
# val auroc 0.5241144592741191
# 
# ===== EPOCH 53/1000 =====
# 
# TRAINING...
