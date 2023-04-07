# -*- coding: utf-8 -*-
import os
import pdb
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import argparse
import re

# import preprocessor as pre
import pandas as pd
from tokenization import BertTokenizer
from copy import deepcopy 

cwd=os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description='build data graph to npz file')
    parser.add_argument('-format', type=str, default='txt_emb')
    parser.add_argument('-obj', type=str, default='Pheme')
    parser.add_argument('-early', type=str, default='')
    parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument('-class_num', type=int, default=2)
    args = parser.parse_args()
    return args

# def clean_data(line):
#     ## Remove @, reduce length, handle strip
#     tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
#     line = ' '.join(tokenizer.tokenize(line))

#     ## Remove url, emoji, mention, prserved words, only preserve smiley
#     #pre.set_options(pre.OPT.URL, pre.OPT.EMOJI, pre.OPT.MENTION, pre.OPT.RESERVED)
#     #pre.set_options(pre.OPT.URL, pre.OPT.RESERVED, pre.OPT.MENTION)
#     pre.set_options(pre.OPT.URL, pre.OPT.RESERVED)
#     line = pre.tokenize(line)

#     ## Remove non-sacii 
#     line = ''.join([i if ord(i) else '' for i in line]) # remove non-sacii
#     return line



def clean_text(text):
    """
    This function cleans the text in the following ways
    1. Replace websites with URL
    2. Replace 's with <space>'s (e.g., her's --> her 's)
    """
    text = text.lower()
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
    #text = re.sub('@\w+','',text)
    #text = text.replace("\'s", "")
    #text = text.replace("\'", "")
    #text = text.replace("n\'t", " n\'t")
    #text = text.replace("@", "")
    #text = text.replace("#", "")
    #text = text.replace("_", " ")
    #text = text.replace("-", " ")
    text = text.replace("\"", "")
    #text = text.replace("@\w+", "")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace("\"", "")
    text = text.replace("$MENTION$", '')
    text = text.replace("$ URL $", '')
    text = text.replace("$URL$", '')
    text = text.replace("URL", '')
    #text = text.replace(".", "")
    #text = text.replace(",", "")
    #text = text.replace("(", "")
    #text = text.replace(")", "")
    text = text.replace("<end>", "")
    text = ' '.join(text.split())
    return text.strip()

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    return x_word, x_index, edgematrix,rootfeat,rootindex

def constructMat_txt(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        text = tree[j]['vec'] # raw text
        nodeC.text = text
        ## not root node ##
        if not indexP == 'None':
            #nodeP = index2node[indexP]
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_text = text
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_text=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_text.append(clean_text(index2node[index_i+1].text))
    # if row == [] and col == []:
    #     matrix[0][0] = 1
    #     row.append(0)
    #     col.append(0)
    edgematrix=[row,col]
    return x_text, edgematrix, root_text, rootindex, matrix


def hyperEdge(branch,node_count):
    count_edge1 = 0
    count_map = {}
    if branch != []:
        
        edge0 = []
        edge1 = []
        for i,val in enumerate(branch):
            edge0_tem = []
            edge1_tem = []
            
            for v in val:
                if v<node_count:
                    edge0_tem.append(v)
                    edge1_tem.append(count_edge1)
            if len(edge0_tem)>=2:
                edge0+=edge0_tem
                edge1+=edge1_tem
                count_edge1 += 1
                for va in edge0_tem:
                    count_map[va] = 1
        for v in range(node_count):
            try:
                count_map[v]
            except KeyError:
                edge0+=[0,v]
                edge1+=[count_edge1,count_edge1]
                count_edge1+=1
    else:
        edge0 = []
        edge1 = []
        for v in range(node_count-1):
            edge0+=[0,v+1]
            edge1+=[count_edge1,count_edge1]
            count_edge1+=1
    return [edge0,edge1]

def mat2branches(matric,rootindex):
    lenth = len(matric)
    branch = []
    stackss = [rootindex]
    branches = []
    while stackss!=[]:
        node = stackss.pop()
        branch.append(node)
        child = [j for j in range(lenth) if matric[node][j]==1]
        if child == []:
            if len(branches)>1 and len(branches[-1]) >=2 and branch[1]==branches[-1][1]:
                branches[-1]=list(set(branches[-1]+branch))
            else:
                branches.append(deepcopy(branch))
            branch.pop()
        else:
            stackss += child
            #branch.append(stackss[-1])
    return branches


def merge_all_text(x_text):
    res = ''
    for index,text in enumerate(x_text):
        if index != len(x_text)-1:
            res += text + ' sep '
        else:
            res += text
    return res

def get_gap_index(input_ids):
    gap_index = []
    for index in range(len(input_ids)):
        if input_ids[index] == 101 or input_ids[index] == 102:
            gap_index.append(index)
    return gap_index
       
def get_interval_index(gap_index, index):
    for th in range(len(gap_index)):
        if th == len(gap_index) - 1:
            return None
        elif gap_index[th] < index <= gap_index[th + 1]:
            return [gap_index[th], gap_index[th + 1]]
        else:
           pass


def convert_example_to_features(example,max_seq_length,tokenizer):
    tokens_a = tokenizer.tokenize(example)
    if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
    tokens = []
    segment_ids = []
    batch_ids = []
    segment_id = 0
    batch_id = 0 
    tokens.append("[CLS]")
    segment_ids.append(segment_id)
    batch_ids.append(batch_id) 
    for token in tokens_a:
        if token == "sep":
            tokens.append("[CLS]")
            segment_ids.append(segment_id)
            segment_id = (segment_id + 1) % 2
            batch_id += 1
            batch_ids.append(batch_id)
        else:
            tokens.append(token)
            segment_ids.append(segment_id)
            batch_ids.append(batch_id)
    tokens.append("[SEP]")
    segment_ids.append(segment_id)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    cls_mask = [1] * len(input_ids)
    batch_id += 1
    batch_ids.append(batch_id)
    # Zero-pad up to the sequence length.
    # while len(input_ids) < max_seq_length:
    #     input_ids.append(0)
    #     cls_mask.append(0)
    #     segment_ids.append(0)
    padding = [0] * (max_seq_length - len(input_ids))
    padding_batch = [batch_id]*(max_seq_length - len(input_ids))

    input_ids += padding
    cls_mask += padding
    segment_ids += padding    
    
    batch_ids += padding_batch
    # input_mask_pruned = []
    # input_mask_pruned.extend(cls_mask)

    # gap_index = get_gap_index(input_ids)
    # for index in range(1, len(input_ids)):
    #     interval_index = get_interval_index(gap_index, index)
    #     if interval_index is None:
    #         input_mask_pruned.extend([0] * len(input_ids))
    #     else:
    #         temp = []
    #         for th in range(len(input_ids)):
    #             if interval_index[0] <= th < interval_index[1]:
    #                 temp.append(1)
    #             else:
    #                 temp.append(0)
    #         input_mask_pruned.extend(temp)
    

    assert len(input_ids) == max_seq_length
    # assert len(input_mask_pruned) == max_seq_length * max_seq_length
    assert len(cls_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(batch_ids) == max_seq_length
    # return [input_ids,input_mask_pruned,segment_ids]
    return [input_ids,cls_mask,segment_ids,batch_ids]

def filter_edge(input_ids,edge):
    count = 0
    for val in input_ids:
        if val==101:
            count += 1 
    new_row = []
    new_col = []
    [row,col] = edge
    for index in range(len(row)):
        if row[index]<count and col[index]<count:
            new_row.append(row[index])
            new_col.append(col[index])
    return [new_row,new_col],count

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def buildgraph(obj, format_, class_num, portion='', texts=None, labels=None):
    """
    labels: lines with label and event id
    """
    if texts is None:
        if format_ == 'idx_cnt':
            treePath = os.path.join(cwd, './dataset/' + obj + '/data.TD_RvNN.vol_5000.{}txt'.format(portion))
            print("loading text from ", treePath)
        elif format_ == 'txt_emb':
            treePath = os.path.join(cwd, './dataset/' + obj + '/data.text.{}txt'.format(portion))
            print("loading text from", treePath)
        texts = open(treePath)
        
        treeDic = {}
        f_tree = open(treePath, 'r',encoding='utf-8')
        for line in f_tree:
            # maxL: max # of clildren nodes for a node; max_degree: max # of the tree depth
            line = line.rstrip()

            eid, indexP, indexC, max_degree, maxL = line.split("\t")[:5]
            Vec = line.split("\t")[-1]
            indexC = int(indexC)
            max_degree = int(max_degree)
            maxL = int(maxL)

            if not treeDic.__contains__(eid):
                # If the event id hasn't been contained
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        f_tree.close()
    else:
        treeDic = {}
        for _, line in texts.iterrows():

            eid, indexP, indexC, max_degree, maxL = line[:5]
            Vec = line[len(line)-1]
            eid = str(eid)
            indexC = int(indexC)
            max_degree = int(max_degree)
            maxL = int(maxL)

            if not treeDic.__contains__(eid):
                # If the event id hasn't been contained
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}

    print('tree number:', len(treeDic))

    # Prepare class name by class number
    if class_num == 2:
        labelset_0, labelset_1 = ['non-rumours', 'non-rumor', 'true'], ['rumours', 'rumor', 'false']
        labelset_2, labelset_3 = [], []
    elif class_num == 4:
        labelset_0, labelset_1, labelset_2, labelset_3 = ['true'], ['false'], ['unverified'], ['non-rumor']
    elif class_num == 3:
        labelset_0, labelset_1, labelset_2, labelset_3 = ['true'], ['false'], ['unverified'], []

    # Load the label file
    if labels is None:
        labels = pd.read_csv(os.path.join(cwd, "./dataset/" + obj + "/data.label.txt"), delimiter="\t", header=None)

    print("loading tree label")
    event, y = [], []
    l0 = l1 = l2 = l3 = 0
    labelDic = {}

    for _, line in labels.iterrows():
        label, eid = line[0], str(line[2])
        label=label.lower()
        event.append(eid)
        if label in labelset_0:
            labelDic[eid]=0
            l0 += 1
        if label in labelset_1:
            labelDic[eid]=1
            l1 += 1
        if label in labelset_2:
            labelDic[eid]=2
            l2 += 1
        if label in labelset_3:
            labelDic[eid]=3
            l3 += 1
    print(len(labelDic))
    print(l1, l2)

    if format_ =='idx_cnt':
        os.makedirs(os.path.join(cwd, './dataset/'+obj+'graph'), exist_ok=True)
    elif format_ =='txt_emb':
        print(os.path.join(cwd, './dataset/'+obj+ 'textgraph'))
        os.makedirs(os.path.join(cwd, './dataset/'+obj+ 'textgraph'), exist_ok=True)

    def loadEid(event, id, y, format_):
        if event is None:
            return None
        if len(event) < 1:
            return None
        if len(event)>= 1:
            if format_ == 'idx_cnt':
                x_word, x_index, tree, rootfeat, rootindex = constructMat(event) 
                x_x = getfeature(x_word, x_index) # x_word: the occur times of words, x_index: the index of words
                rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
                np.savez( os.path.join(cwd, './dataset/'+obj+'graph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            elif format_ == 'txt_emb':
                x_text, tree, root_text, rootindex, matrix = constructMat_txt(event) 
                x_all_text = merge_all_text(x_text)
                [input_ids,input_mask,segment_ids,batch_ids] = convert_example_to_features(x_all_text,300,tokenizer)
                tree,count = filter_edge(input_ids,tree)
                branches = mat2branches(matrix,rootindex)
                hyper_tree = hyperEdge(branches,count)
                tree, rootindex, y = np.array(hyper_tree), np.array(rootindex), np.array(y)
                input_ids,input_mask,segment_ids,batch_ids = np.array(input_ids),np.array(input_mask),np.array(segment_ids),np.array(batch_ids)
                np.savez( os.path.join(cwd, './dataset/'+obj+'textgraph/'+id+'.npz'), input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids,batch_ids=batch_ids,root=root_text,edgeindex=tree,rootindex=rootindex,y=y)
            return None

    print("loading dataset", )
    Parallel(n_jobs=1, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid], format_) for eid in tqdm(event))
    return treeDic

if __name__ == '__main__':
    args = parse_args()
    obj = args.obj
    if args.format=='idx_cnt':
        path = os.path.join(cwd, './dataset/'+obj+'graph/')
        print('Building the graph data by index:cnt at ', path)
    elif args.format=='txt_emb':
        path = os.path.join(cwd, './dataset/'+obj+'text'+'graph/')
        print('Building the graph data by raw text at ', path)
    bert_path = os.path.join(cwd, args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=args.do_lower_case)
    #portion = '{}.'.format(args.early)
    portion = ''
    os.makedirs(path, exist_ok=True)
    buildgraph(args.obj, args.format, args.class_num, portion)

