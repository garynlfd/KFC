import gc
from sklearn.metrics import roc_curve,auc
from dataset import *
from models_multi_task import *
from utils import *
import torch_resnet101
import torch
from torch.optim import SGD
from losses import *
import sys
import argparse

def find(args):
    path=os.path.join(args.save_path,"weights_Cornell_multi")
    batch_size=args.batch_size
    log_path=args.log_path
    val_dataset = FIW(os.path.join(args.sample,"./aligned_dataset/mixed_dataset/mixed_dataset_val_cross.txt"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, pin_memory=False)
    model = MultiTask().cuda()
    model.load(path, "_std_best")
    model.eval()
    with torch.no_grad():
        auc ,threshold = val_model(model, val_loader)
    mylog("auc : ",auc,path=log_path)
    mylog("threshold :" ,threshold,path=log_path)


def val_model(model, val_loader):
    y_true = []
    y_pred = []
    heartbeat = 0
    for img1, img2, kinship, labels, races in val_loader:
        r1,r2,e1,e2,x1,x2,bias_map,bias_pair=model([img1.cuda(),img2.cuda()])
        y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        y_true.extend(labels.cpu().detach().numpy().tolist())
        heartbeat += 1
        if(heartbeat % 100 == 0): print(heartbeat)
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_pred)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds_keras[maxindex]
    return auc(fpr,tpr),threshold


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="find threshold")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--log_path", type=str, default="./log.txt",help="log path default log.txt ")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(100)
    find(args)
