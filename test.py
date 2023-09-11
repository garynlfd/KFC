import argparse
import numpy as np
import gc
from models_multi_task import *
import sys
import torch
from keras.preprocessing import image
import os
from utils import *
import heapq

def baseline_model(model_path):
    model = MultiTask().cuda()
    model.load(model_path, "_std_best")
    model.eval()
    return model

def get_test(sample_path,res):
    test_file_path = os.path.join(sample_path,"./aligned_dataset/mixed_dataset/mixed_dataset_test_cross.txt")
    test=[]
    f = open(test_file_path, "r+", encoding='utf-8')
    file_leng = 0
    while True:
        line=f.readline().replace('\n','')
        if not line:
            break
        else:
            test.append(line.split(' '))
            file_leng += 1
    f.close()
    res['avg'][0]=len(test)
    for now in test:
        res[now[2]][0]+=1
    return test, file_leng


def gen(list_tuples, batch_size):
    total=len(list_tuples)
    start=0
    while True:
        if start+batch_size<total:
            end=start+batch_size
        else:
            end=total
        batch_list=list_tuples[start:end]
        datas=[]
        labels=[]
        classes=[]
        races=[]
        id1=[]
        id2=[]

        for now in batch_list:
            datas.append([now[0],now[1]])
            labels.append(int(now[3]))
            classes.append(now[2])
            races.append(now[4])
            id1.append(0)

        X1 = np.array([read_image(x[0]) for x in datas])
        X2 = np.array([read_image(x[1]) for x in datas])
        yield datas, X1, X2, labels,classes,races,id1,id2,batch_list
        start=end
        if start == total:
            yield None,None,None,None,None,None,None,None,None
        gc.collect()


def read_image(path):
    img = image.load_img(path, target_size=(112, 112))
    img = np.array(img).astype(np.float)
    return np.transpose(img, (2, 0, 1))


def test(args):
    model_path = os.path.join(args.save_path,"weights_Cornell_multi")
    sample_path = args.sample
    batch_size = args.batch_size
    log_path = args.log_path
    threshold = args.threshold
    model = baseline_model(model_path)
    classes = [
        'fd', 'md', 'fs', 'ms', 'avg'
    ]
    res={}
    for n in classes:
        res[n]=[0,0]
    test_samples, file_leng = get_test(sample_path, res)
    easy_kin_similarities = [] # p=labels=1
    hard_kin_similarities = [] # p=0, labels=1
    easy_non_kin_similarities = [] # p=labels=0
    hard_non_kin_similarities = [] # p=1, labels=0
    with torch.no_grad():
        fnp = 0
        fpp = 0
        for datas, img1, img2, labels, classes, races, id1, id2, batch_list in gen(test_samples, batch_size):
            if img1 is not None:
                img1 = torch.from_numpy(img1).type(torch.float).cuda()
                img2 = torch.from_numpy(img2).type(torch.float).cuda()
                id1 = torch.Tensor(id1).type(torch.int).cuda()
                id2 = torch.Tensor(id2).type(torch.int).cuda()
                r1,r2,e1,e2,x1,x2,bias_map,bias_pair = model([img1, img2])
                pred = torch.cosine_similarity(e1, e2, dim=1).cpu().detach().numpy().tolist()
                for i in range(len(pred)):
                    if pred[i] >= threshold:
                        p = 1
                    else:
                        p = 0
                    if p == labels[i]:
                        res['avg'][1] += 1
                        res[classes[i]][1] += 1
                        if p == 1:
                            if len(easy_kin_similarities) < 60:
                                heapq.heappush(easy_kin_similarities, (pred[i], datas[i]))
                            else:
                                heapq.heappushpop(easy_kin_similarities, (pred[i], datas[i]))
                        else: 
                            if len(easy_non_kin_similarities) < 30:
                                heapq.heappush(easy_non_kin_similarities, (pred[i], datas[i]))
                            else:
                                heapq.heappushpop(easy_non_kin_similarities, (pred[i], datas[i]))
                    else:
                        neg_similarity = -pred[i]
                        if labels[i] == 1: # p = 0; label = 1 -> false negative pair
                            fnp = fnp + 1
                            if len(hard_kin_similarities) < 30:
                                heapq.heappush(hard_kin_similarities, (neg_similarity, datas[i]))
                            else:
                                heapq.heappushpop(hard_kin_similarities, (neg_similarity, datas[i]))
                        else: # p = 1; label = 0 -> false positive pair
                            fpp = fpp + 1
                            if len(hard_non_kin_similarities) < 30:
                                heapq.heappush(hard_non_kin_similarities, (neg_similarity, datas[i]))
                            else:
                                heapq.heappushpop(hard_non_kin_similarities, (neg_similarity, datas[i]))
            else:
                break
    mylog("number of false negative pairs: ", fnp, path=log_path)
    mylog("false negative rate: ", fnp / file_leng, path=log_path)
    mylog("number of false positive pairs: ", fpp, path=log_path)
    mylog("false positive rate: ", fpp / file_leng, path=log_path)
    for key in res:
        if res[key][0] == 0: continue
        mylog(key, ':', res[key][1] / res[key][0], path=log_path)


    print("easy KIN cosine similarities:")
    for similarity, datas in heapq.nlargest(60, easy_kin_similarities):
        print(f"Image Pair: {datas}, Similarity: {similarity}")
    
    print("-"*20)

    print("easy non-KIN cosine similarities:")
    for similarity, datas in heapq.nlargest(30, easy_non_kin_similarities):
        print(f"Image Pair: {datas}, Similarity: {similarity}")
    
    print("-"*20)

    print("hard KIN cosine similarities:")
    for similarity, datas in heapq.nlargest(30, hard_kin_similarities):
        print(f"Image Pair: {datas}, Similarity: {similarity}")

    print("-"*20)

    print("hard non-KIN cosine similarities:")
    for similarity, datas in heapq.nlargest(30, hard_non_kin_similarities):
        print(f"Image Pair: {datas}, Similarity: {similarity}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="test  accuracy")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--threshold", type=float, help=" threshold ")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--log_path", type=str, default="./log.txt", help="log path default log.txt ")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(100)
    test(args)
