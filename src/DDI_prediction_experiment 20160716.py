# in this program,we want to write object-oriented codes,and enhance the extension.
__author__ = 'zhang'
import argparse
import os
from pathlib import Path
# code cũ: from pylab import *
import networkx as nx
import math
from numpy import linalg
from numpy.linalg import inv, pinv
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import copy
from numpy import linalg as LA
import csv
import array
import random
import numpy
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import linear_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_project_path(filename):
    path = Path(filename)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def to_1d_float_array(values):
    return np.real(np.asarray(values)).astype(float).ravel()

def normalize_score_vector(values):
    values = to_1d_float_array(values).reshape(-1, 1)
    return MinMaxScaler().fit_transform(values).ravel()

def cross_validation(drug_drug_matrix, CV_num, seed):
    link_number = 0
    link_position = []
    nonLinksPosition = []  # all non-link position
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            else:
                nonLinksPosition.append([i, j])

    link_position = np.array(link_position)
    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)

    fold_num = link_number // CV_num
    print(fold_num)

    for CV in range(0, CV_num):
        print('*********round:' + str(CV) + "**********\n")
        test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
        test_index.sort()
        testLinkPosition = link_position[test_index]
        train_drug_drug_matrix = copy.deepcopy(drug_drug_matrix)
        for i in range(0, len(testLinkPosition)):
            train_drug_drug_matrix[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0
            train_drug_drug_matrix[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0
            testPosition = list(testLinkPosition) + list(nonLinksPosition)
        #  GA
        weights,cf1,cf2 = internal_determine_parameter(copy.deepcopy(train_drug_drug_matrix))
        # cf1,cf2=internal_determine_parameter(copy.deepcopy(train_drug_drug_matrix))
        [multiple_predict_matrix,multiple_predict_results] = ensemble_method(copy.deepcopy(drug_drug_matrix), train_drug_drug_matrix, testPosition)
        # logstic weight

        ensemble_results, ensemble_results_cf1,ensemble_results_cf2= ensemble_scoring(copy.deepcopy(drug_drug_matrix), multiple_predict_matrix,testPosition, weights,cf1,cf2)
        for i in range(0,len(multiple_predict_results)):
            [auc_score, aupr_score, precision, recall, accuracy, f]=multiple_predict_results[i]
            file_results.write(auc_score+' '+aupr_score+' '+precision+' '+recall+' '+accuracy+' '+f+"\n")
            file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results
        file_results.write(auc_score + ' ' + aupr_score + ' ' + precision+ ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results_cf1
        file_results.write(auc_score + ' ' + aupr_score + ' ' + precision + ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results_cf2
        file_results.write(auc_score + ' ' + aupr_score + ' ' + precision+ ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        file_results.flush()

        weights_str=''
        for i in range(0,len(weights)):
            weights_str=weights_str+' '+str(weights[i])
        file_weights.write(weights_str + "\n")
        file_results.flush()
        file_weights.flush()


class Topology:
    def topology_similarity_matrix(drug_drug_matrix):
       drug_drug_matrix=np.matrix(drug_drug_matrix)
       # code cũ: G = nx.from_numpy_matrix(drug_drug_matrix)
       G = nx.from_numpy_array(np.asarray(drug_drug_matrix, dtype=float))
       drug_num=len(drug_drug_matrix)
       common_similarity_matrix=np.zeros(shape=(drug_num,drug_num))
       AA_similarity_matrix=np.zeros(shape=(drug_num,drug_num))
       RA_similarity_matrix=np.zeros(shape=(drug_num,drug_num))

       eigenValues,eigenVectors = linalg.eig(drug_drug_matrix)
       idx = eigenValues.argsort()[::-1]
       eigenValues = eigenValues[idx[0]]

       beta=0.5*(1/eigenValues)
       Katz_similarity_matrix=inv(np.identity(drug_num)-beta*drug_drug_matrix)-np.identity(drug_num)
       for i in range(0,drug_num):
         for j in range(i+1,drug_num):
             commonn_neighbor=list(nx.common_neighbors(G, i, j))
             common_similarity_matrix[i][j]=len(commonn_neighbor)
             AA_score=0
             RA_score=0
             for k in range(0,len(commonn_neighbor)):
                  neighbor_degree = G.degree(commonn_neighbor[k])
                  # code cũ: AA_score=AA_score+1/math.log(len(G.neighbors(commonn_neighbor[k])))
                  if neighbor_degree > 1:
                     AA_score=AA_score+1/math.log(neighbor_degree)
                  # code cũ: RA_score=RA_score+1/len(G.neighbors(commonn_neighbor[k]))
                  if neighbor_degree > 0:
                     RA_score=RA_score+1/neighbor_degree
             AA_similarity_matrix[i][j]=AA_score
             RA_similarity_matrix[i][j]=RA_score

             common_similarity_matrix[j][i]=common_similarity_matrix[i][j]
             AA_similarity_matrix[j][i]=AA_similarity_matrix[i][j]
             RA_similarity_matrix[j][i]=RA_similarity_matrix[i][j]

       D=np.diag(((drug_drug_matrix.sum(axis=1)).getA1()))
       L=D-drug_drug_matrix
       LL=pinv(L)
       LL=np.matrix(LL)
       ACT_similarity_matrix=np.zeros(shape=(drug_num,drug_num))
       for i in range(0,drug_num):
          for j in range(i+1,drug_num):
              ACT_similarity_matrix[i][j]=1/(LL[i,i]+LL[j,j]-2*LL[i,j])
              ACT_similarity_matrix[j][i]=ACT_similarity_matrix[i][j]

       D=np.diag(((drug_drug_matrix.sum(axis=1)).getA1()))
       N=pinv(D)*drug_drug_matrix
       alpha=0.9
       RWR_similarity_matrix=(1-alpha)*pinv(np.identity(drug_num)-alpha*N)
       RWR_similarity_matrix= RWR_similarity_matrix+ np.transpose(RWR_similarity_matrix)

       return np.matrix(common_similarity_matrix),np.matrix(AA_similarity_matrix),np.matrix(RA_similarity_matrix),np.matrix(Katz_similarity_matrix),np.matrix(ACT_similarity_matrix),np.matrix(RWR_similarity_matrix)

def load_csv(filename,type): #  load csv, ignore the first row,type=int, data read as int， else float
        matrix_data=[]
        # code cũ: with open(filename, 'r') as csvfile:
        with open(resolve_project_path(filename), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row_vector in csvreader:
              if type=='int':
                 matrix_data.append(list(map(int,row_vector[1:])))
              else:
                 matrix_data.append(list(map(float,row_vector[1:])))
        return np.matrix(matrix_data)

def modelEvaluation(real_matrix,predict_matrix,testPosition,featurename): #  compute cross validation results
       real_labels=[]
       predicted_probability=[]

       for i in range(0,len(testPosition)):
           real_labels.append(real_matrix[testPosition[i][0],testPosition[i][1]])
           predicted_probability.append(predict_matrix[testPosition[i][0],testPosition[i][1]])

       # code cũ: normalize=MinMaxScaler()
       # code cũ: predicted_probability= normalize.fit_transform(predicted_probability)
       predicted_probability=normalize_score_vector(predicted_probability)
       real_labels=np.array(real_labels).ravel()
       # code cũ: predicted_probability=np.array(predicted_probability)

       precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
       aupr_score = auc(recall, precision)

       all_F_measure=np.zeros(len(pr_thresholds))
       for k in range(0,len(pr_thresholds)):
           if (precision[k]+precision[k])>0:
              all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
           else:
              all_F_measure[k]=0
       max_index=all_F_measure.argmax()
       threshold=pr_thresholds[max_index]

       fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
       auc_score = auc(fpr, tpr)
       predicted_score=np.zeros(len(real_labels))
       predicted_score[predicted_probability>threshold]=1

       f=f1_score(real_labels,predicted_score)
       accuracy=accuracy_score(real_labels,predicted_score)
       precision=precision_score(real_labels,predicted_score)
       recall=recall_score(real_labels,predicted_score)
       print('results for feature:'+featurename)
       print('************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f, f-measure:%.3f************************' %(auc_score,aupr_score,recall,precision,accuracy,f))
       auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
       results=[auc_score,aupr_score,precision, recall,accuracy,f]
       return results

def fitFunction(individual,parameter1,parameter2):
       real_labels=parameter1
       multiple_prediction=parameter2
       ensemble_prediction=np.zeros(len(real_labels))
       for i in range(0,len(multiple_prediction)):
            ensemble_prediction=ensemble_prediction+individual[i]*multiple_prediction[i]
       precision, recall, pr_thresholds = precision_recall_curve(real_labels, ensemble_prediction)
       aupr_score = auc(recall, precision)
       return (aupr_score),


def getParamter(real_matrix, multiple_matrix, testPosition):
    # code cũ: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # code cũ: creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_float", random.uniform, 0, 1)
    # Structure initializers
    variable_num = len(multiple_matrix)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, variable_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #################################################################################################
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(predict_matrix[testPosition[j][0], testPosition[j][1]])
        # code cũ: normalize = MinMaxScaler()
        # code cũ: predicted_probability = normalize.fit_transform(predicted_probability)
        predicted_probability = normalize_score_vector(predicted_probability)
        multiple_prediction.append(predicted_probability)

    #################################################################################################
    toolbox.register("evaluate", fitFunction, parameter1=real_labels, parameter2=multiple_prediction)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(0)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    pop.sort(key=lambda ind: ind.fitness, reverse=True)
    print(pop[0])
    return pop[0]

class MethodHub():
    def neighbor_method(similarity_matrix,train_drug_drug_matrix):
        return_matrix=np.matrix(train_drug_drug_matrix)*np.matrix(similarity_matrix)
        D=np.diag(((similarity_matrix.sum(axis=1)).getA1()))
        return_matrix=return_matrix*pinv(D)
        return_matrix=return_matrix+np.transpose(return_matrix)
        return return_matrix

    def Label_Propagation(similarity_matrix,train_drug_drug_matrix):
       alpha=0.9
       similarity_matrix=np.matrix(similarity_matrix)
       train_drug_drug_matrix=np.matrix(train_drug_drug_matrix)
       D=np.diag(((similarity_matrix.sum(axis=1)).getA1()))
       N=pinv(D)*similarity_matrix

       transform_matrix=(1-alpha)*pinv(np.identity(len(similarity_matrix))-alpha*N)
       return_matrix=transform_matrix*train_drug_drug_matrix
       return_matrix=return_matrix+np.transpose(return_matrix)
       return return_matrix

    def generate_distrub_matrix(drug_drug_matrix):
       A=np.matrix(drug_drug_matrix)
       [num,num]=A.shape
       upper_A= np.triu(A, k=0)
       [row_index,col_index]=np.where(upper_A==1)

       ratio=0.1   # disturb how many links are removed
       select_num=int(len(row_index)*ratio)
       # code cũ: index=arange(0, (upper_A.sum()).sum())
       index=np.arange(0, (upper_A.sum()).sum())
       # print(index.shape)

       random.seed(0)
       random.shuffle(index)
       # np.random.shuffle(index)
       select_index=index[0:select_num]
       delta_A=np.zeros(shape=(num,num))
       for i in range(0,select_num):
          delta_A[row_index[select_index[i]]][col_index[select_index[i]]]=1
          delta_A[col_index[select_index[i]]][row_index[select_index[i]]]=1

       return delta_A,row_index,col_index,select_num

    def disturb_matrix_method(train_drug_drug_matrix):
       input_A=np.matrix(train_drug_drug_matrix)
       [num,num]=input_A.shape
       delta_A,row_index,col_index,select_num=MethodHub.generate_distrub_matrix(input_A)
       A=input_A-delta_A
       eigenvalues, eigenvectors = LA.eig(A)
       num_eigenvalues=len(eigenvalues)

       delta_eigenvalues=np.zeros(num_eigenvalues)
       for i in range(0,num_eigenvalues):
             delta_eigenvalues[i]=(np.transpose(eigenvectors[:,i])*delta_A*eigenvectors[:,i])/(np.transpose(eigenvectors[:,i])*eigenvectors[:,i])

       reconstructed_A=np.zeros(shape=(num,num))
       for i in range(0,num_eigenvalues):
           reconstructed_A=reconstructed_A+(eigenvalues[i]+delta_eigenvalues[i])*eigenvectors[:,i]*np.transpose(eigenvectors[:,i])

       reconstructed_A[np.where(input_A==1)]=1

       return_matrix=reconstructed_A+np.transpose(reconstructed_A)
       return return_matrix


def ensemble_method(drug_drug_matrix,train_drug_drug_matrix,testPosition):

    chem_sim_similarity_matrix=load_csv('dataset/chem_Jacarrd_sim.csv','float')
    target_similarity_matrix=load_csv('dataset/target_Jacarrd_sim.csv','float')

    transporter_similarity_matrix=load_csv('dataset/transporter_Jacarrd_sim.csv','float')
    enzyme_similarity_matrix=load_csv('dataset/enzyme_Jacarrd_sim.csv','float')

    pathway_similarity_matrix=load_csv('dataset/pathway_Jacarrd_sim.csv','float')
    indication_similarity_matrix=load_csv('dataset/indication_Jacarrd_sim.csv','float')

    label_similarity_matrix=load_csv('dataset/sideeffect_Jacarrd_sim.csv','float')
    offlabel_similarity_matrix=load_csv('dataset/offsideeffect_Jacarrd_sim.csv','float')


    multiple_matrix=[]
    multiple_result = []
    print('********************************************************')
    predict_matrix=MethodHub.neighbor_method(chem_sim_similarity_matrix,train_drug_drug_matrix)
    results=modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'chem_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(target_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'target_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(transporter_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'transporter_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(enzyme_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'enzyme_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(pathway_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'pathway_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(indication_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'indication_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)


    predict_matrix=MethodHub.neighbor_method(label_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'label_neighbor')
    multiple_result.append(results)
    results =multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(offlabel_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'offlabel_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # print('*************************************************************************************************************************************')
    common_similarity_matrix,AA_similarity_matrix,RA_similarity_matrix,Katz_similarity_matrix,ACT_similarity_matrix,RWR_similarity_matrix=Topology.topology_similarity_matrix(train_drug_drug_matrix)
    predict_matrix=MethodHub.neighbor_method(common_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=common_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'common_similarity_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(AA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=AA_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'AA_similarity_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(RA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RA_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'RA_similarity_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(Katz_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=Katz_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'Katz_similarity_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(ACT_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=ACT_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'ACT_similarity_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(RWR_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RWR_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'RWR_similarity_neighbor')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # print('*************************************************************************************************************************************')
    predict_matrix=MethodHub.Label_Propagation(chem_sim_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'chem_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)  #12

    predict_matrix=MethodHub.Label_Propagation(target_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'target_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(transporter_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'transporter_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(enzyme_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'enzyme_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(pathway_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'pathway_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(indication_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'indication_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(label_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'label_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(offlabel_similarity_matrix,train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'offlabel_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)   #14

    # print('*************************************************************************************************************************************')
    predict_matrix=MethodHub.Label_Propagation(common_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=common_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'common_similarity_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(AA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=AA_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'AA_similarity_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(RA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RA_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'RA_similarity_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(Katz_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=Katz_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'Katz_similarity_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(ACT_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=ACT_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'ACT_similarity_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(RWR_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RWR_similarity_matrix
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'RWR_similarity_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

   # print('*************************************************************************************************************************************')
    predict_matrix=MethodHub.disturb_matrix_method(train_drug_drug_matrix)
    results =modelEvaluation(drug_drug_matrix,predict_matrix,testPosition,'disturb_matrix_label')
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    return multiple_matrix,multiple_result

def internal_determine_parameter(drug_drug_matrix):
    train_drug_drug_matrix,testPosition=holdout_by_link(copy.deepcopy(drug_drug_matrix),0.2,1)
    [multiple_matrix,multiple_result]=ensemble_method(copy.deepcopy(drug_drug_matrix),train_drug_drug_matrix,testPosition)
    weights=getParamter(copy.deepcopy(drug_drug_matrix),multiple_matrix,testPosition)
    # weights=[]

    input_matrix=[]
    output_matrix = []
    for i in range(0, len(testPosition)):
        vector=[]
        for j in range(0, len(multiple_matrix)):
           # code cũ: vector.append(multiple_matrix[j][testPosition[i][0], testPosition[i][1]])
           vector.append(float(np.real(multiple_matrix[j][testPosition[i][0], testPosition[i][1]])))
        input_matrix.append(vector)
        # code cũ: output_matrix.append(drug_drug_matrix[testPosition[i][0], testPosition[i][1]])
        output_matrix.append(int(drug_drug_matrix[testPosition[i][0], testPosition[i][1]]))


    input_matrix=np.array(input_matrix)
    output_matrix= np.array(output_matrix)
    # code cũ: clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', solver='liblinear', tol=1e-6, max_iter=1000)
    clf1.fit(input_matrix, output_matrix)

    # code cũ: clf2 = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    clf2 = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6, max_iter=1000)
    clf2.fit(input_matrix, output_matrix)
    print('*************************parameter determined*************************')
    # return weights
    return weights,clf1, clf2

##########################################################################################################################
def holdout_by_link(drug_drug_matrix, ratio, seed):
    link_number = 0
    link_position = []
    nonLinksPosition = []  # all non-link position
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            else:
                nonLinksPosition.append([i, j])

    link_position = np.array(link_position)
    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)
    train_index = index[(int(link_number * ratio) + 1):]
    test_index = index[0:int(link_number * ratio)]
    train_index.sort()
    test_index.sort()
    testLinkPosition = link_position[test_index]
    train_drug_drug_matrix = copy.deepcopy(drug_drug_matrix)

    for i in range(0, len(testLinkPosition)):
        train_drug_drug_matrix[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0
        train_drug_drug_matrix[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0
    testPosition = list(testLinkPosition) + list(nonLinksPosition)

    return train_drug_drug_matrix, testPosition


def ensemble_scoring(real_matrix, multiple_matrix, testPosition, weights,cf1,cf2):
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(predict_matrix[testPosition[j][0], testPosition[j][1]])
        # code cũ: normalize = MinMaxScaler()
        # code cũ: predicted_probability = normalize.fit_transform(predicted_probability)
        predicted_probability=normalize_score_vector(predicted_probability)
        # code cũ: predicted_probability=np.array(predicted_probability)
        multiple_prediction.append(predicted_probability)
    ensemble_prediction = np.zeros(len(real_labels))
    for i in range(0, len(multiple_matrix)):
        ensemble_prediction = ensemble_prediction + weights[i] * multiple_prediction[i]

    ensemble_prediction_cf1 = np.zeros(len(real_labels))
    ensemble_prediction_cf2= np.zeros(len(real_labels))
    for i in range(0, len(testPosition)):
        vector=[]
        for j in range(0, len(multiple_matrix)):
           # code cũ: vector.append(multiple_matrix[j][testPosition[i][0], testPosition[i][1]])
           vector.append(float(np.real(multiple_matrix[j][testPosition[i][0], testPosition[i][1]])))
        # code cũ: vector=np.array(vector)
        vector=np.array(vector).reshape(1, -1)

        aa=cf1.predict_proba(vector)
        # code cũ: print(aa)
        ensemble_prediction_cf1[i]=(cf1.predict_proba(vector))[0][1]
        ensemble_prediction_cf2[i]=(cf2.predict_proba(vector))[0][1]


    # code cũ: normalize = MinMaxScaler()
    # code cũ: ensemble_prediction = normalize.fit_transform(ensemble_prediction)
    ensemble_prediction = normalize_score_vector(ensemble_prediction)

    result = calculate_metric_score(real_labels, ensemble_prediction)
    result_cf1=calculate_metric_score(real_labels, ensemble_prediction_cf1)
    result_cf2=calculate_metric_score(real_labels, ensemble_prediction_cf2)

    return result,result_cf1,result_cf2

def calculate_metric_score(real_labels,predict_score):
   real_labels=np.array(real_labels).ravel()
   predict_score=to_1d_float_array(predict_score)
   precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
   aupr_score = auc(recall, precision)

   all_F_measure = np.zeros(len(pr_thresholds))
   for k in range(0, len(pr_thresholds)):
      if (precision[k] + precision[k]) > 0:
          all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
      else:
          all_F_measure[k] = 0
   max_index = all_F_measure.argmax()
   threshold = pr_thresholds[max_index]
   fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
   auc_score = auc(fpr, tpr)

   predicted_score = np.zeros(len(real_labels))
   predicted_score[predict_score > threshold] = 1

   f = f1_score(real_labels, predicted_score)
   accuracy = accuracy_score(real_labels, predicted_score)
   precision = precision_score(real_labels, predicted_score)
   recall = recall_score(real_labels, predicted_score)
   print('results for feature:' + 'weighted_scoring')
   print(    '************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f************************' % (
        auc_score, aupr_score, recall, precision, accuracy))
   auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
   results = [auc_score, aupr_score, precision, recall, accuracy, f]
   return results

def run_smoke_test(sample_size, seed):
    print('DAY 1 smoke test: load data -> holdout split -> chem_neighbor -> metrics')
    drug_drug_matrix = load_csv('dataset/drug_drug_matrix.csv', 'int')
    chem_similarity_matrix = load_csv('dataset/chem_Jacarrd_sim.csv', 'float')

    drug_drug_matrix = drug_drug_matrix[:sample_size, :sample_size]
    chem_similarity_matrix = chem_similarity_matrix[:sample_size, :sample_size]

    train_drug_drug_matrix, testPosition = holdout_by_link(copy.deepcopy(drug_drug_matrix), 0.2, seed)
    predict_matrix = MethodHub.neighbor_method(chem_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, 'day1_chem_neighbor_smoke')

    print('DAY 1 smoke test finished')
    print('sample_size:', sample_size)
    print('test_pairs:', len(testPosition))
    print('metrics:', results)

def run_legacy_full_experiment(runtimes, cv_num):
    global file_results
    global file_weights

    drug_drug_matrix = load_csv('dataset/drug_drug_matrix.csv', 'int')
    file_results_str="result/result_on_our_dataset_5CV"
    weights_results_str="result/weights_on_our_dataset_5CV"
    os.makedirs(resolve_project_path('result'), exist_ok=True)
    for seed in range(0, runtimes):
        file_results_path=resolve_project_path(file_results_str+"_"+str(seed)+".txt")
        weights_results_path=resolve_project_path(weights_results_str+"_"+str(seed)+".txt")
        file_results = open(file_results_path, "w")
        file_weights = open(weights_results_path, "w")
        try:
            cross_validation(drug_drug_matrix, cv_num, seed)
        finally:
            file_results.close()
            file_weights.close()

def parse_args():
    parser = argparse.ArgumentParser(description='DDI prediction legacy experiment runner')
    parser.add_argument('--smoke-test', action='store_true', help='Run a small Day 1 check instead of the full legacy experiment')
    parser.add_argument('--sample-size', type=int, default=50, help='Number of drugs used by --smoke-test')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used by --smoke-test')
    parser.add_argument('--legacy-full-run', action='store_true', help='Run the original full cross-validation experiment')
    parser.add_argument('--runtimes', type=int, default=20, help='Number of seeds for --legacy-full-run')
    parser.add_argument('--cv-num', type=int, default=5, help='Number of folds for --legacy-full-run')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.legacy_full_run:
        run_legacy_full_experiment(args.runtimes, args.cv_num)
    else:
        run_smoke_test(args.sample_size, args.seed)

# code cũ:
# runtimes=20        # implement 20 runs of 5-fold cross validation for base predictors and the ensmeble model
#
# drug_drug_matrix = load_csv('dataset/drug_drug_matrix.csv', 'int')
# file_results_str="result/result_on_our_dataset_5CV"
# weights_results_str="result/weights_on_our_dataset_5CV"
# for seed in range(0, runtimes):
#     file_results_path=file_results_str+"_"+str(seed)+".txt"
#     weights_results_path=weights_results_str+"_"+str(seed)+".txt"
#     file_results = open(file_results_path, "w")
#     file_weights = open(weights_results_path, "w")
#     cross_validation(drug_drug_matrix, 5, seed)

if __name__ == '__main__':
    main()
