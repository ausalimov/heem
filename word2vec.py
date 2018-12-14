import numpy as np
import argparse
import json 
import sys
import pandas
import nltk 
import collections
import scipy
import operator
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile

######### 
'''
Code for HEEM (Heuristic Entity Evaluation Metric 

This contains all of the models for training and evalauting HEEM. 

Included is:
    A) doc2vec (sentence level)
    B) lda2vec
    : VS-1 (Vector Similarity) HEEM Score 
    TODO: XO (Cross Oriented) HEEM Score
    

doc2vec will be used for individual comparison as well. 

We train seperate doc2vec per model, with an additional one for human responses.
We initially do training with a train/test set, then coalesce for the actual HEEM inputs.

Using JGibbsLDA we then find a per-topic representation of each sentence as well. 
We then train it with doc2vec as well to get a vector representation.


To get the VS-1 HEEM Score we simply find the cosine distance between the mean vectors. 
To get the VS-2 HEEM Score we simply find the average cosine distance between each vector. 
This is our first score. 

Finally, for our XO-N HEEM Scores: 

For each model, we combine create a dataset that has that model and the human vectors. 
We then run Soft-Margin SVM on it and find the degree of seperability.
A higher degree indicates a worse score as the vectors are easier to seperate and thus, 
in some hyper plane, can easily be seperated between human and machine responses.
XO-3 corresponds to the HEEM Vector. 

For the HEEM Vector, we concatenate the doc and lda vecs. 

For baseline models, we also compare the mean cosine distance per sentence for doc2vec (sentence level). 

Future Directions:
    End-to-end training of HEEM. 
    Adversarial metric as per HRED_GAN paper. 


'''

### FILE PATHS ###
milestone_samples = '/Users/mqa994/repos/heem/data/ubuntu_milestone_samples.json'
# saved_model_dir = "saved_models/"
saved_model_dir = ""
jgibb_lda_base = "/Users/mqa994/repos/heem/JGibbLDA_models/"
jgibb_assign = "/model-final.tassign"

model_keys = []
gt_key = None

### DATA PROCESSING ###
# Returns a dictionary of structure: 
# key : value 
# model : list of sentences
def load_data(file_name, corpus):
    global model_keys 
    global gt_key 
    with open(file_name) as f:
        data = [item for sublist in json.load(f) for item in sublist]
    data_dict = {}
    model_keys = [k for k in list(data[0].keys()) if "context" not in k]
    model_keys = [k for k in model_keys if "adv_cnoise_" + corpus + "_old" not in k]
    model_keys = [k for k in model_keys if "adv_noise_" + corpus not in k]
    model_keys.append("adv_noise_"+ corpus + "_old")
    gt_key = [k for k in model_keys if "ground_truth" in k][0]
    print(model_keys)
    for key in model_keys: 
        data_dict[key] = []
        for row in data:
            data_dict[key].append(row[key])
    return data_dict

def save_sentences(model_name, sentences_list):
    with open("raw_sentences/" + model_name, 'w') as f:
        for item in sentences_list:
            f.write("%s\n" % item)

def save_sentences_for_lda(model_name, sentences_list):
    with open("lda_inputs/" + model_name, 'w') as f:
        f.write("%s\n" % str(len(sentences_list)))
        for item in sentences_list:
            f.write("%s\n" % item)

def load_lda_results():
    lda_data = {}
    for key in model_keys:
        with open(jgibb_lda_base + key + jgibb_assign, 'r') as f:
            lda_data[key] = f.readlines()
        lda_data[key] = list(" ".join(list(str(w.split(":")[-1]) for w in s.split())) for s in lda_data[key])
    return lda_data
    

def process_data_for_doc2vec(raw_data, split=True, preprocess=True):
    data = {}
    if split: 
        for key in model_keys:
            data[key] = {}
            split_index = int(len(raw_data[key]))
            train, test = raw_data[key][:split_index], raw_data[key][split_index:]
            data[key]["train"] = train
            data[key]["test"] = test
    else:
        for key in model_keys:
            data[key] = {}
            data[key]["train"] = raw_data[key]
            data[key]["test"] = []
    tagged_sentences = {}
    test_sentences = {}
    for key in model_keys:
        tagged_sentences[key] = []
        for i, sentence in enumerate(data[key]["train"]):
            if preprocess:
                tagged_sentences[key].append(TaggedDocument(simple_preprocess(sentence), [i]))
            else:
                tagged_sentences[key].append(TaggedDocument(sentence, [i]))
        test_sentences[key] = []
        for i, sentence in enumerate(data[key]["test"]):
            if preprocess:
                tagged_sentences[key].append(simple_preprocess(sentence))
            else:
                tagged_sentences[key].append(sentence)
    return tagged_sentences, test_sentences

### COMPARISON ###

def compare_vectors(v1, v2):
    return scipy.spatial.distance.cosine(v1, v2)

def compare_model_vectors(m1_vecs, m2_vecs):
    all_cosine_sims = []
    for i in range(len(m1_vecs) - 1):
        for j in range(len(m2_vecs) - 1):
            all_cosine_sims = compare_vectors(m1_vecs[i], m2_vecs[j])
    return np.mean(all_cosine_sims)

def process_data_for_svm(data):
    svm_data = {}
    for key in model_keys: 
        svm_data[key] = {}
        svm_data[key]["train"] = {"X":[], "y":[]}
        svm_data[key]["test"] = {"X":[], "y":[]} 
        i = 0 
        cur = "train"
        train_test_switch = int(len(data[gt_key]) * 0.7)
        for s1, s2 in zip(data[gt_key], data[key]):
            if i == train_test_switch:
                cur = "test"
            i += 1
            svm_data[key][cur]["X"].append(s1)
            svm_data[key][cur]["X"].append(s2)
            svm_data[key][cur]["y"].append(0)
            svm_data[key][cur]["y"].append(1)
    return svm_data 
            
            
        
        
        

### TRAINING ### 
def train_doc2vec_model(train_data, vector_size=50):
    print("Training doc2vec model")
    model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)
    model.build_vocab(train_data)
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def train_doc2vec(data, preprocess=True, vector_size=25):
    doc2vec_train, doc2vec_test = process_data_for_doc2vec(data, split=False, preprocess=preprocess)
    doc2vec_models = {model_key:train_doc2vec_model(doc2vec_train[model_key], vector_size=vector_size) for model_key in model_keys}
    return doc2vec_models

def train_svm_model(X, y, X_test, y_test):
    results = []
    for c in [0.01]:
        for g in [0.1]: 
            for d in [1]:
                clf = SVC(C=c, gamma=g, degree=d, kernel='linear')
                clf.fit(X, y) 
                results.append(tuple((c, g, d, accuracy_score(y_test, clf.predict(X_test)))))
    # print(min(results, key=operator.itemgetter(3)))
    return clf

def train_svm(raw_data):
    svm_data = process_data_for_svm(heem_data)
    svm_models = {}
    for model in model_keys:
        if model == gt_key: continue
        print("Training SVM for", model)
        X_train = svm_data[model]["train"]["X"] 
        y_train = svm_data[model]["train"]["y"] 
        X_test = svm_data[model]["test"]["X"] 
        y_test = svm_data[model]["test"]["y"] 
        svm_models[model] = train_svm_model(X_train, y_train, X_test, y_test)
    return svm_models 

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for running the HEEM system.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--save_raw', action='store_true', help="save data input as raw file of sentences") 
    parser.add_argument('--save_lda_inputs', action='store_true', help="save input data for jgibblda") 
    parser.add_argument('--run_doc2vec', action='store_true', help="run doc2vec and get cosine sims") 
    parser.add_argument('--skip_cosine_eval', action='store_true', help="run doc2vec and get cosine sims") 
    parser.add_argument('--input_data', type=str, help="run doc2vec and get cosine sims") 
    parser.add_argument('--corpus', type=str, help="run doc2vec and get cosine sims") 
    args = parser.parse_args()

    data = load_data(args.input_data, args.corpus)
    print("Running!")

    if args.save_raw:
        print("Saving raw sentences!")
        for m1 in model_keys: 
            save_sentences(m1, data[m1]) 
        sys.exit()

    if args.save_lda_inputs:
        print("Saving LDA sentences!")
        for m1 in model_keys: 
            print(m1)
            save_sentences_for_lda(m1, data[m1]) 
        sys.exit()

    # A) Print cosine sim scores between models.
    doc2vec_models = train_doc2vec(data)
    if not args.skip_cosine_eval:
        print("Get cosine scores for doc2vec")
        for m1 in model_keys:
            vecs_1 = doc2vec_models[m1].docvecs
            vecs_2 = doc2vec_models[gt_key].docvecs
            print(m1, gt_key, compare_model_vectors(vecs_1, vecs_2)) 
             

    # B) Get and learn topic vectors. 
    lda_data = load_lda_results()
    lda2vec_models = train_doc2vec(lda_data, preprocess=False, vector_size=25)
    if not args.skip_cosine_eval:
        print("Get cosine scores for lda2vec")
        for m1 in model_keys:
            vecs_1 = lda2vec_models[m1].docvecs
            vecs_2 = lda2vec_models[gt_key].docvecs
            print(m1, gt_key, compare_model_vectors(vecs_1, vecs_2)) 
    

    # C) Assemble HEEM vector, get HEEM Cosine Scores
    heem_data = {}
    for m in model_keys:
        heem_data[m] = []
        doc2vecs = doc2vec_models[m].docvecs 
        lda2vecs = lda2vec_models[m].docvecs 
        for i in range(len(doc2vecs)):
            dvec = doc2vecs[i]
            lvec = lda2vecs[i]
            heem_data[m].append(np.concatenate([dvec, lvec]))
    if not args.skip_cosine_eval:
        print("Get cosine scores for heem vector")
        for m1 in model_keys:
            vecs_1 = heem_data[m1]
            vecs_2 = heem_data[gt_key]
            print(m1, gt_key, compare_model_vectors(vecs_1, vecs_2)) 

    svm_models = train_svm(heem_data) 
    for model in model_keys:
        if model == gt_key: continue
        svm_model = svm_models[model]
        print(model, np.mean([np.linalg.norm(v) for v in svm_model.support_vectors_]))
            

    

