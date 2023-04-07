import sys,os
import logging
sys.path.append(os.getcwd())
# from Process.process import *
import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from Process.optimization import BertAdam, warmup_linear
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *

from HyperGCN import BertForHgcn
from torch.utils.data import Dataset
from torch_geometric.data import Data
import copy

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def load_fold_data(str):
    def read_label_txt(path):
        data = []
        f = open(path, 'r',encoding='utf-8')
        for line in f:
            line = line.strip()
            eid, label = line.split('\t')
            data.append(eid)
        f.close()
        return data

    train_path = os.path.join(cwd, './dataset/'+dataset_name+'/'+str+'/train.label.txt')
    test_path = os.path.join(cwd, './dataset/'+dataset_name+'/'+str+'/test.label.txt')
    return [read_label_txt(train_path),read_label_txt(test_path)]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels),outputs

class HyperEdgeData(Data):
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.node_num], [self.edge_num]])
        else:
            return super(HyperEdgeData, self).__inc__(key, value, *args, **kwargs)

class GraphDataset(Dataset):
    def __init__(self, fold_x, dataset_name, data_path=os.path.join(cwd,'dataset')):
        self.fold_x = fold_x
        self.data_path = os.path.join(data_path,dataset_name+'textgraph')


    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        edge_num = max(data['edgeindex'][1])+1
        node_num = max(data['edgeindex'][0])+1
        # input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids,root=root_text,edgeindex=tree,rootindex=rootindex,y=y
        return HyperEdgeData(x_input_ids=torch.LongTensor([data['input_ids']]),
                    x_input_mask = torch.LongTensor([data['input_mask']]),
                    x_segment_ids = torch.LongTensor([data['segment_ids']]),
                    edge_index=torch.LongTensor(edgeindex),
                    edge_num = torch.LongTensor([edge_num]),
                    node_num = torch.LongTensor([node_num]),
                    y=torch.LongTensor([int(data['y'])]))


def train_fold(train_fold,test_fold):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare model
    model = BertForHgcn.from_pretrained(bert_path,num_labels = num_labels)
    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = int(
            len(train_fold) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    # train_losses,val_losses,train_accs,val_accs = [],[],[],[]
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list = GraphDataset(train_fold,dataset_name)
    testdata_list = GraphDataset(test_fold,dataset_name)
    train_loader = DataLoader(traindata_list, batch_size=train_batch_size,
                                shuffle=True, num_workers=0)
    test_loader = DataLoader(testdata_list, batch_size=eval_batch_size,
                                shuffle=True, num_workers=0)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    model.train()
    for epoch_ids in range(num_train_epochs):
            tr_loss = 0
            result_loss = 100
            acc_,pre_,rec_,f1_ = 0,0,0,0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                
                input_ids, input_mask, segment_ids,edge,label_ids = batch.x_input_ids,batch.x_input_mask,batch.x_segment_ids,batch.edge_index,batch.y
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                edge = edge.to(device)
                # batch = tuple(t.to(device) for t in batch)
                # input_ids, input_mask_tweet, segment_ids, label_ids = batch
                
                loss = model(input_ids, segment_ids, input_mask,edge, label_ids)
                # if n_gpu > 1:
                #     loss = loss.mean() # mean() to average on multi-gpu.
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps

                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y,prediction = [],[]
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, input_mask, segment_ids,edge,label_ids = batch.x_input_ids,batch.x_input_mask,batch.x_segment_ids,batch.edge_index,batch.y
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                edge = edge.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask,edge, label_ids)
                    logits = model(input_ids, segment_ids, input_mask,edge)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy,pre_ids = accuracy(logits, label_ids)
                y+=label_ids.tolist()
                prediction+=pre_ids.tolist()
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
            Acc,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2=evaluationclass(prediction,y)
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss/nb_tr_steps
            if (eval_loss<result_loss):
                acc_2,pre_2,rec_2,f1_2 = Acc2,Prec2,Recll2,F2
                acc_1,pre_1,rec_1,f1_1 = Acc1,Prec1,Recll1,F1
            result = {'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'global_step': global_step,
                    'loss': loss,
                    'acc':Acc,
                    'acc1':Acc1,
                    'pre1':Prec1,
                    'recall1':Recll1,
                    'f1':F1,
                    'acc2':Acc2,
                    'pre2':Prec2,
                    'recall2':Recll2,
                    'f2':F2}

            output_eval_file = os.path.join(output_dir, f"{fold5data_val}_eval_results{epoch_ids}.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** %s Eval results *****",str(epoch_ids))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            # Save a trained model
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # output_model_file = os.path.join(output_dir, f"pytorch_model{epoch_ids}.bin")
    return acc_1,pre_1,rec_1,f1_1,acc_2,pre_2,rec_2,f1_2
            #torch.save(model_to_save.state_dict(), output_model_file)
            

dataset_name = "Weibo"

max_seq_length = 300
train_batch_size = 8
eval_batch_size = 8
learning_rate = 5e-5
num_train_epochs = 15
gradient_accumulation_steps = 1
output_dir = "results/hypergcn/Weibo"
bert_model = "bert-base-chinese"
num_labels = 2
bert_path = os.path.join(cwd, bert_model)
warmup_proportion = 0.1
patience = 7
seed = 42

fold5data = ["split_0","split_1","split_2","split_3","split_4"]

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
acc1,pre1,rec1,F1_1 = 0,0,0,0
acc2,pre2,rec2,F1_2 = 0,0,0,0
for fold5data_val in fold5data:
    fold0_train,fold0_test = load_fold_data(fold5data_val)
    acc_1,pre_1,rec_1,f1_1,acc_2,pre_2,rec_2,f1_2 = train_fold(fold0_train,fold0_test)
    acc1+=acc_1
    pre1+=pre_1
    rec1+=rec_1
    F1_1+=f1_1
    acc2+=acc_2
    pre2+=pre_2
    rec2+=rec_2
    F1_2+=f1_2
result = {"acc1:":acc1/5,"pre1:":pre1/5,"rec1:":rec1/5,"F1_1:":F1_1/5,"acc2:":acc2/5,"pre2:":pre2/5,"rec2:":rec2/5,"F1_2:":F1_2/5}
output_eval_file = os.path.join(output_dir, f"results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** %s Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
