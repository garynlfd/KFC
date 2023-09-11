from sklearn.metrics import roc_curve,auc
from dataset import *
from models_multi_task import *
from torch.optim import SGD
from losses import *
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import warnings; warnings.filterwarnings("ignore")

race_dict={'AA':np.array([0], dtype=np.long),'A':np.array([1], dtype=np.long),'C':np.array([2], dtype=np.long),'I':np.array([3], dtype=np.long),\
        'AA&AA':np.array([4], dtype=np.long),'AA&A':np.array([5], dtype=np.long),'AA&C':np.array([6], dtype=np.long),'AA&I':np.array([7], dtype=np.long),\
        'A&AA':np.array([8], dtype=np.long),'A&A':np.array([9], dtype=np.long),'A&C':np.array([10], dtype=np.long),'A&I':np.array([11], dtype=np.long),\
        'C&AA':np.array([12], dtype=np.long),'C&A':np.array([13], dtype=np.long),'C&C':np.array([14], dtype=np.long),'C&I':np.array([15], dtype=np.long),\
        'I&AA':np.array([16], dtype=np.long),'I&A':np.array([17], dtype=np.long),'I&C':np.array([18], dtype=np.long),'I&I':np.array([19], dtype=np.long)}


def training(args):

    batch_size=args.batch_size
    val_batch_size=args.batch_size
    epochs=args.epochs
    steps_per_epoch=600
    save_path=os.path.join(args.save_path,"weights_Cornell_multi")
    beta=args.beta
    log_path=args.log_path


    train_dataset=FIW(os.path.join(args.sample,"./aligned_dataset/mixed_dataset/mixed_dataset_train.txt"))
    val_dataset=FIW(os.path.join(args.sample,"./aligned_dataset/mixed_dataset/mixed_dataset_val_choose_cross.txt"))

    train_loader=DataLoader(train_dataset,batch_size=batch_size,num_workers=1,pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=1, pin_memory=False)

    model=MultiTask().cuda()

    optimizer_model = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    max_auc=0.0
    min_std=100

    criterion_CE=torch.nn.CrossEntropyLoss()

    for epoch_i in range(epochs):
        mylog("\n*************",path=log_path)
        mylog('epoch ' + str(epoch_i + 1),path=log_path)
        total_loss_epoch = 0
        kinship_loss_epoch = 0
        race_loss_epoch = 0
        face_recognition_loss_epoch = 0
        margin_list_sum = [0, 0, 0, 0]
        margin_list_appearance= [0, 0, 0, 0]
        model.train()
        for index_i, data in enumerate(tqdm(train_loader)): # this loop loops for 1 batch
            image1,image2,kinship,labels,races=data # one batch includes 25 image pairs
            r1,r2,e1,e2,x1,x2,bias_map,bias_pair= model([image1,image2])
            kinship_loss, margin_list = contrastive_loss(x1,x2,kinship,races,bias_map,bias_pair,beta=beta)
            race_loss = criterion_CE(r1, races) + criterion_CE(r2, races)

            for i in range(4):
                if margin_list[i] != 0: 
                    margin_list_appearance[i] += 1
            for i in range(4):
                margin_list_sum[i] += margin_list[i]

            loss = kinship_loss + race_loss


            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            kinship_loss_epoch += kinship_loss.item()
            race_loss_epoch += race_loss.item()
            total_loss_epoch += loss.item()
            
            if (index_i+1)==steps_per_epoch:
                break

        use_sample=(epoch_i+1)*batch_size*steps_per_epoch
        print(use_sample)
        train_dataset.set_bias(use_sample)

        for i in range(4):
            if margin_list_appearance[i] != 0:
                margin_list_sum[i] /= margin_list_appearance[i]

        mylog("total_loss:" + "%.6f" % (total_loss_epoch / steps_per_epoch),path=log_path)
        mylog("kinship_loss:" + "%.6f" % (kinship_loss_epoch / steps_per_epoch),path=log_path)
        mylog("race_loss:" + "%.6f" % (race_loss_epoch / steps_per_epoch),path=log_path)
        mylog("race margin: ", margin_list_sum, path=log_path)
        model.eval()
        with torch.no_grad():
            auc,std = val_model(model, val_loader)
        mylog("auc is %.6f "% auc,path=log_path)
        mylog("std is %.6f "% std,path=log_path)
        if max_auc < auc:
            mylog("auc improve from :" + "%.6f" % max_auc + " to %.6f" % auc,path=log_path)
            max_auc=auc
            mylog("save model " + save_path,path=log_path)
            model.save(save_path,"_auc_best")
        else:
            mylog("auc did not improve from %.6f" % float(max_auc),path=log_path)
            if((epoch_i+1)%10==0):
                model.save(save_path,epoch_i+1)
        if min_std > std:
            mylog("std improve from :" + "%.6f" % min_std + " to %.6f" % std,path=log_path)
            min_std=std
            mylog("save model " + save_path,path=log_path)
            model.save(save_path,"_std_best")

def save_model(model,path):
    torch.save(model.state_dict(),path)

def val_model(model, val_loader):
    y_true = []
    y_pred = []
    race_label=[]
    race_table={'A':0,'AA':0,'C':0,"I":0}
    race_total={'A':0,'AA':0,'C':0,"I":0}
    for img1, img2, kinship, labels, races in val_loader:
        r1,r2,e1,e2,x1,x2,bias_map,bias_pair=model([img1.cuda(),img2.cuda()])
        y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        y_true.extend(labels.cpu().detach().numpy().tolist())
        race_label.extend(races.cpu().detach().numpy().tolist())
    fpr, tpr, threshold_ = roc_curve(y_true, y_pred)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = threshold_[maxindex]

    pred_cos= torch.tensor(y_pred)>threshold
    for i,pred in enumerate(pred_cos):
        if(pred==y_true[i]):
            race_table[list(race_dict.keys())[list(race_dict.values()).index(int(race_label[i]))].split('&')[0]]+=1
        race_total[list(race_dict.keys())[list(race_dict.values()).index(int(race_label[i]))].split('&')[0]]+=1
        
    acc=np.array([race_table[i]/race_total[i] for i in ['A','AA','C','I']])
    std=np.std(acc)

    return auc(fpr,tpr),std


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=25,help="batch size default 25")
    parser.add_argument( "--sample", type=str, help="sample root")
    parser.add_argument( "--save_path",  type=str, help="model save path")
    parser.add_argument( "--epochs", type=int,default=100, help="epochs number default 100")
    parser.add_argument( "--beta", default=0.08, type=float, help="beta default 0.08")
    parser.add_argument( "--log_path", default="./log.txt", type=str, help="log path default log.txt")
    parser.add_argument( "--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    training(args)
