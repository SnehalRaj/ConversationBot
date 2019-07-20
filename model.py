from __future__ import print_function
import numpy as np
import sys
import os
import cntk as C
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
from cntk.device import try_set_default_device, gpu
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

try_set_default_device(gpu(0))

#Initialize Global variables
validation_query_vectors = []
validation_passage_vectors = []
validation_labels = []
r=1
tf=1
l= np.zeros(10)   
q_max_words=15
p_max_words=120
emb_dim=50
'''
## The following LoadValidationSet method reads ctf format validation file and creates query, passage feature vectors and also copies labels for each pair.
## the created vectors will be useful to find metrics on validation set after training each epoch which will be useful to decide the best model 
def LoadValidationSet(validationfile):
    f = open(validationfile,'r',encoding="utf-8")
    for line in f:
        tokens = line.strip().split("|")  
        #tokens[0] will be empty token since the line is starting with |
        x1 = tokens[1].replace("qfeatures","").strip() #Query Features
        x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
        y = tokens[3].replace("labels","").strip() # labels
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]
        y = [int(w) for w in y.split()]        
        y = y[1] # label will be at index 1, i.e. if y = "1 0" then label=0 else if y="0 1" then label=1

        validation_query_vectors.append(x1)
        validation_passage_vectors.append(x2)
        validation_labels.append(y)

        #print("1")
    
    print("Validation Vectors are created")'''
def cosine(vector_a, vector_b):
    return C.cosine_distance(vector_a, vector_b)


def create_loss(x,a):
    if (a==1):
        return 1-x
    else:
        return x


EMB_DIM   = 50 # Embedding dimension
HIDDEN_DIM = 50 # LSTM dimension
DSSM_DIM = 50 # Dense layer dimension
DROPOUT_RATIO = 0.2



#The following method defines a RNN network which runs a unindirectional GRU on query features and passage features to generate a enconding of question and answer respectively.
def rnn_network(queryfeatures, passagefeatures, num_classes):
    with C.layers.default_options(initial_state=0.1):
        q_gru = C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM), go_backwards=True, name = 'q_gru')(queryfeatures)
        last1 = C.sequence.last(q_gru)
        q_proj = C.layers.Dense(DSSM_DIM, activation=C.tanh, name='q_proj')(last1)
        dropout_qdo1 = C.layers.Dropout(DROPOUT_RATIO, name='dropout_qdo1')(q_proj)
        q_enc = C.layers.Dense(DSSM_DIM, activation=C.tanh, name='q_enc')(dropout_qdo1)

        a_gru = C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM), go_backwards=True, name = 'a_gru')(passagefeatures)
        last2 = C.sequence.last(a_gru)
        a_proj = C.layers.Dense(DSSM_DIM, activation=C.tanh, name='a_proj')(last2)
        dropout_ado1 = C.layers.Dropout(DROPOUT_RATIO, name='dropout_ado1')(a_proj)
        a_enc = C.layers.Dense(DSSM_DIM, activation=C.tanh, name='a_enc')(dropout_ado1)

        model = C.cosine_distance(q_enc, a_enc)

    return model


def create_reader(path, is_training, query_total_dim, passage_total_dim, label_total_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs( queryfeatures = StreamDef(field='qfeatures', shape=query_total_dim,is_sparse=False), 
                                                            passagefeatures = StreamDef(field='pfeatures', shape=passage_total_dim,is_sparse=False), 
                                                            labels   = StreamDef(field='labels', shape=label_total_dim,is_sparse=False)
                                                            )), 
                           randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)



def TrainAndValidate(trainfile):

    #*****Hyper-Parameters******
    global tf, l, a, r
    q_max_words= 15
    p_max_words = 120
    emb_dim = 50
    num_classes = 2
    minibatch_size = 4000
    epoch_size = 5241880 #No.of samples in training set
    total_epochs = 19 #Total number of epochs to run
    query_total_dim = q_max_words*emb_dim
    label_total_dim = num_classes
    passage_total_dim = p_max_words*emb_dim


    #****** Create placeholders for reading Training Data  ***********
    query_input_var =  C.sequence.input_variable((1,q_max_words,emb_dim),np.float32,is_sparse=False)
    passage_input_var =  C.sequence.input_variable((1,p_max_words,emb_dim),np.float32,is_sparse=False)
    output_var = C.input_variable(num_classes,np.float32,is_sparse = False)
    train_reader = create_reader(trainfile, True, query_total_dim, passage_total_dim, label_total_dim)
    input_map = { query_input_var : train_reader.streams.queryfeatures, passage_input_var:train_reader.streams.passagefeatures, output_var : train_reader.streams.labels}

    # ********* Model configuration *******
    model_output = rnn_network(query_input_var, passage_input_var, num_classes)
    # model_output.restore('RNN_{}.dnn')  // This line should be uncommented to restore training from a particular model
    
    if(output_var[1]=='1'):
        a=1
    else:
        a=0
    loss = C.sigmoid(create_loss(model_output, a))
    pe = None
    
    lr_per_sample = [0.0015625]*20 + [0.00046875]*20 + [0.00015625]*20 + [0.000046785]*10 + [0.000015625]
    lr_schedule = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size = epoch_size)
    mms = [0]*20 + [0.9200444146293233]*20 + [0.9591894571091382]
    mm_schedule = C.learners.momentum_schedule(mms, epoch_size=epoch_size, minibatch_size = minibatch_size)
    l2_reg_weight = 0.0002

    dssm_learner = C.learners.momentum_sgd(model_output.parameters, lr_schedule , mm_schedule)   
    learner = dssm_learner
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=total_epochs)

    #************Create Trainer with model_output object, learner and loss parameters*************  
    trainer = C.Trainer(model_output, (loss, pe), learner, progress_printer)
    C.logging.log_number_of_parameters(model_output) 

    # **** Train the model in batchwise mode *****
    for epoch in range(total_epochs):       # loop over epochs
        print("Epoch : ",epoch)
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = train_reader.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)        # training step
            sample_count += data[output_var].num_samples   # count samples processed so far

        trainer.summarize_training_progress()
        model_output.save("RNN_{}.dnn".format(epoch+1))
        '''
        #*** Find metrics on validation set after every epoch ******# 
        predicted_labels=[]
        for i in range(len(validation_query_vectors)):
            queryVec   = np.array(validation_query_vectors[i],dtype="float32").reshape(1,q_max_words,emb_dim)
            passageVec = np.array(validation_passage_vectors[i],dtype="float32").reshape(1,p_max_words,emb_dim)
            scores = model_output(queryVec,passageVec)[0]   # do forward-prop on model to get score  
            predictLabel = 1 if scores[1]>=scores[0] else 0
            predicted_labels.append(predictLabel) 
        metrics = precision_recall_fscore_support(np.array(validation_labels), np.array(predicted_labels), average='binary')'''
        #print("precision : "+str(metrics[0])+" recall : "+str(metrics[1])+" f1 : "+str(metrics[2])+"\n")



    return model_output 

## The following GetPredictionOnEvalSet method reads all query passage pair vectors from CTF file and does forward prop with trained model to get similarity score
## after getting scores for all the pairs, the output will be written into submission file. 

def GetPredictionOnEvalSet(model,testfile,submissionfile):
    global q_max_words,p_max_words,emb_dim

    f = open(testfile,'r',encoding="utf-8")
    all_scores={} # Dictionary with key = query_id and value = array of scores for respective passages
    for line in f:
        tokens = line.strip().split("|")  
        #tokens[0] will be empty token since the line is starting with |
        x1 = tokens[1].replace("qfeatures","").strip() #Query Features
        x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
        query_id = tokens[3].replace("qid","").strip() # Query_id
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]    
        queryVec   = np.array(x1,dtype="float32").reshape(1,q_max_words,emb_dim)
        passageVec = np.array(x2,dtype="float32").reshape(1,p_max_words,emb_dim)
        score = (C.sigmoid(model(queryVec, passageVec)).eval())[0] # do forward-prop on model to get score
        if(query_id in all_scores):
            all_scores[query_id].append(score)
        else:
            all_scores[query_id] = [score]
    fw = open(submissionfile,"w",encoding="utf-8")
    for query_id in all_scores:
        scores = all_scores[query_id]
        scores_str = [str(sc) for sc in scores] # convert all scores to string values
        scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.  
        fw.write(query_id+"\t"+scores_str+"\n")
    fw.close()

    
if __name__ == "__main__":

    trainSetFileName = "TrainData_120.ctf"
    validationSetFileName = "ValidationData.ctf"
    testSetFileName = "EvaluationData_120.ctf"
    submissionFileName = "answer.tsv"
   
   # LoadValidationSet(validationSetFileName)    #Load Validation Query, Passage Vectors from Validation CTF File
    model = TrainAndValidate(trainSetFileName) # Training and validation methods    
    GetPredictionOnEvalSet(model,testSetFileName,submissionFileName) # Get Predictions on Evaluation Set



   

