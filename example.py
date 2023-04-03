import numpy as np
import sys

sys.path.append('../AutoFolio')

from autofolio.facade.af_csv_facade import AFCsvFacade

def train_af_model(perf_fn,feat_fn,model_fn):

    # maximize=True,
    af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn,objective='solution_quality',\
        maximize=False,runtime_cutoff=7200)

    # fit AutoFolio; will use default hyperparameters of AutoFolio
    af.fit()

    # tune AutoFolio's hyperparameter configuration for 4 seconds
    config = af.tune(wallclock_limit=3600)

    # evaluate configuration using a 10-fold cross validation
    score = af.cross_validation(config=config)

    # re-fit AutoFolio using the (hopefully) better configuration
    # and save model to disk
    af.fit(config=config, save_fn=model_fn)

# param: feature_vec, type: np.array
def predict(feature_vec,model_fn):
    # load AutoFolio model and
    # get predictions for new meta-feature vector
    pred = AFCsvFacade.load_and_predict(vec=feature_vec, load_fn=model_fn)
    return pred



def read_perf_file(perf_fn):
    stream =open(perf_fn,'r')
    header=stream.readline()
    header=header.split(',')
    header=header[1:]
    algname2idx={}
    for idx in range(0,len(header)):
        algname2idx[header[idx]]=idx
    
    file2perf={}

    # no headers
    lines=stream.readlines()
    for line in lines:
        line=line.split(',')
        ins=str(line[0])
        line=line[1:]
        perf=[]
        for ele in line:
            perf.append(float(ele))
        file2perf[ins]=perf
    return algname2idx,file2perf

def read_feat_file(feat_fn):
    stream =open(feat_fn,'r')
    header=stream.readline()
    feature_names=header.split(',')

    file2feat={}
    # no headers
    lines=stream.readlines()
    for line in lines:
        item=line.split(',')
        name=str(item[0])
        item=item[1:]
        feat=[]
        for ele in item:
            feat.append(float(ele))
        file2feat[name]=feat
        
    return feature_names,file2feat


def validation(perf_fn,feat_fn,model_fn):
    feature_name,file2feat=read_feat_file(feat_fn)
    alname2idx,file2perf=read_perf_file(perf_fn)
    total_improvement=0
    improved_number=0
    for file,perf in file2perf.items():
        print(file)
        feat=file2feat[file]
        print(feat)
        algname=predict(feat,model_fn)
        algidx=alname2idx[algname]
        cur_pref=perf[algidx]
        cur_pref=(1.0/cur_pref)-1.0
        total_improvement+=cur_pref
        if cur_pref>0:
            improved_number+=1
    print('performance of the Autofolio on training set: total improvements',\
        total_improvement,' , improved number of instaces: ',improved_number)
    # perf_table=input_csv(perf_fn)
    # feat_table=input_csv(feat_fn)


if __name__ == "__main__":

    perf_fn = "performance.csv"
    feat_fn = "feature.csv"
    model_fn = "af_model.pkl"
    train_af_model(perf_fn,feat_fn,model_fn)
    validation(perf_fn,feat_fn,model_fn)








