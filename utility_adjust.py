import sklearn.preprocessing as sp
import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
import collections
from sklearn.feature_extraction.text import CountVectorizer
import random
import sys

def encode_X(X):
  categories_indexes = list(np.where(X.dtypes == "object")[0])
  numerics_indexes = list(np.where(X.dtypes != "object")[0])
  categories = []
  numerics = []
  for i in categories_indexes:
    categories.append(X.columns[i])
  for i in numerics_indexes:
    numerics.append(X.columns[i])
  encX = sp.OneHotEncoder(drop="first", sparse=False)
  encX.fit(X[categories])

  features = numerics.copy() 
  features.extend(encX.get_feature_names())
      
  encoded_X = np.hstack([X[numerics].values, encX.transform(X[categories])])
  return encoded_X, features

def replace_value_using_decision_tree(learning_df, synthetic_df, X_index, Y_index):
  # 決定木を作る
  X = learning_df[X_index]
  encoded_X, features_X = encode_X(X)
  y = learning_df[Y_index]
  y = pd.DataFrame(np.where(y == "<=50K", "Y","N"))
  encY = sp.OrdinalEncoder()
  encY.fit(y)
  encoded_y = encY.transform(y)

  dtree = DecisionTreeClassifier()
  param = list(ParameterGrid({"max_depth": [10], "random_state": [0]}))[0]
  dtree.set_params(**param)
  dtree.fit(encoded_X, y)

  pred_X = synthetic_df[X_index]
  encoded_X_pred, features_X_pred = encode_X(pred_X)
  featuresList = list(features_X)
  col_diff = list(set(features_X) - set(features_X_pred))
  col_diff_idx = []
  for col in set(features_X) - set(features_X_pred):
    col_diff_idx.append(featuresList.index(col))

  for col_idx in np.sort(col_diff_idx):
    encoded_X_pred = np.insert(encoded_X_pred,col_idx,0,axis=1)
    features_X_pred = np.insert(features_X_pred,col_idx,features_X[col_idx])

  pred_y = dtree.predict(encoded_X_pred)
  return np.where(pred_y == "Y", "<=50K",">50K")

# 比較記号（＞、≦）を文字に変更する。CountVectorizerで分割されないため。
def replace_comp_symbol(str):
    str = str.replace('<=','LE')
    str = str.replace('>','GT')
    return str

# 出力時に元に戻す
def replace_back_comp_symbol(str):
    str = str.replace('LE','<=')
    str = str.replace('GT','>')
    return str

# １タプルの値を一つのセンテンスとみなし、文書ベクトルで表す
def generate_tuple_vectors(df):
    #値の前にカラム名をつける（カラム名_値）
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(col)+"_"+str(x))
    #タプルをセンテンスにする
    tuple_sentence_list = []
    df_T=df.T
    for data in df_T.columns:
        tuple_sentence = " ".join(list(df_T[data]))
        tuple_sentence = replace_comp_symbol(tuple_sentence)
        tuple_sentence_list.append(tuple_sentence)  
    #文書ベクトル（単語出現頻度ベクトル）を作る
    cv = CountVectorizer(lowercase=False, analyzer="word", token_pattern="(?u)\\b[\\w(-<=)]+\\b")
    X = cv.fit_transform(tuple_sentence_list)
    return pd.DataFrame(X.toarray(), columns=cv.get_feature_names())

# 二つの文書ベクトルの次元を揃える
def match_vector_dimensions(vectorA, vectorB):
    for col in set(vectorA.columns) - set(vectorB.columns):
        vectorB[col]=0
    for col in set(vectorB.columns) - set(vectorA.columns):
        vectorA[col]=0
    vectorA = vectorA.sort_index(axis=1, ascending=True)
    vectorB = vectorB.sort_index(axis=1, ascending=True)
    return vectorA, vectorB

# 共起行列を求める
def cooccurrence_matrix(vector):
    return vector.T.dot(vector)

# カラムのヒストグラムを求める
def column_histo(col_num,coocc):
    matrix = coocc.filter(regex="^"+str(col_num)+"_*",axis=0)
    histo = np.diag(matrix[matrix.index])
    return histo

#変更する共起ペアを決定する
def select_replace_cooccurence_pair(_coocc_diff):
  ##共起行列のdiffで絶対値の一番大きなところを一つピックアップ
  ## (ただしヒストグラム部分は取らないように、対角成分を０にしておく)
  _coocc_diff_copy = _coocc_diff.copy()
  for i in _coocc_diff_copy.index:
    _coocc_diff_copy.loc[i, i] = 0
  replace_pair_idx = np.unravel_index(np.argmax(np.abs(_coocc_diff_copy)), _coocc_diff.shape)
  replace_pair_values = [_coocc_diff.columns[replace_pair_idx[0]],_coocc_diff.columns[replace_pair_idx[1]]]
  return replace_pair_values

def replace_pair_both_columns(_src_pair, _coocc_diff):
  attr_A_idx = _src_pair[0].split('_')[0]
  attr_B_idx = _src_pair[1].split('_')[0]

  candidates = _coocc_diff.filter(regex="^"+attr_A_idx+"_*", axis=0).filter(regex="^"+attr_B_idx+"_*", axis=1)
  if (candidates.at[_src_pair[0], _src_pair[1]]>0):
    dst = np.unravel_index(np.argmin(candidates), candidates.shape)
  else:
    dst = np.unravel_index(np.argmax(candidates), candidates.shape)
  dst_A = candidates.index[dst[0]]
  dst_B = candidates.columns[dst[1]]

  return [dst_A, dst_B]

#入れ替えるレコードを決定する
def select_replace_records_both_columns(_src_pair, _dst_pair, _coocc_diff, _vector):
    num_replace = min([abs(_coocc_diff.at[_src_pair[0],_src_pair[1]]), abs(_coocc_diff.at[_dst_pair[0],_dst_pair[1]]), random.randint(5,10)])
    src_diff_vector = _coocc_diff[_src_pair[0]]
    dst_diff_vector = _coocc_diff[_dst_pair[0]] 
    src_diff_sum = _vector[(_vector[_src_pair[0]]==1) & (_vector[_src_pair[1]]==1) ].dot(src_diff_vector.T)
    dst_diff_sum = _vector[(_vector[_src_pair[0]]==1) & (_vector[_src_pair[1]]==1) ].dot(dst_diff_vector.T)
    replace_diff_sum = src_diff_sum - dst_diff_sum
    return replace_diff_sum.sort_values(ascending=False)[:num_replace].index

# 値を入れ替える（値（dfB）, 文書ベクトル（vectorB）, 共起行列の差分（coocc_diff））
def replace_value_pair(records, src_pair, dst_pair, dfB, vectorB, coocc_diff):

    #print(coocc_diff.at[src_pair[0],src_pair[1]])

    col_nums = [src_pair[0].split('_')[0], src_pair[1].split('_')[0]]
    for r in records: 
        
        #文書ベクトルを変える
        vectorB.at[r, src_pair[0]] = 0
        vectorB.at[r, src_pair[1]] = 0
        vectorB.at[r, dst_pair[0]] = 1
        vectorB.at[r, dst_pair[1]] = 1

        #共起行列の差分を変える
        coocc_diff.at[src_pair[0],src_pair[1]]-=1
        coocc_diff.at[src_pair[1],src_pair[0]]-=1
        coocc_diff.at[src_pair[0],src_pair[0]]-=1
        coocc_diff.at[src_pair[1],src_pair[1]]-=1

        coocc_diff.at[dst_pair[0],dst_pair[1]]+=1
        coocc_diff.at[dst_pair[1],dst_pair[0]]+=1
        coocc_diff.at[dst_pair[0],dst_pair[0]]+=1
        coocc_diff.at[dst_pair[1],dst_pair[1]]+=1

        for val in dfB.loc[r,:]:  
            val = replace_comp_symbol(val)

            if(val!=src_pair[0] and val!=src_pair[1]):
              coocc_diff.at[src_pair[0],val]-=1
              coocc_diff.at[src_pair[1],val]-=1
              coocc_diff.at[val,src_pair[0]]-=1
              coocc_diff.at[val,src_pair[1]]-=1

              coocc_diff.at[dst_pair[0],val]+=1
              coocc_diff.at[dst_pair[1],val]+=1
              coocc_diff.at[val,dst_pair[0]]+=1
              coocc_diff.at[val,dst_pair[1]]+=1            
              
        #値を変える
        dfB.at[r, int(col_nums[0])] = dst_pair[0]
        dfB.at[r, int(col_nums[1])] = dst_pair[1]

    #print(coocc_diff.at[src_pair[0],src_pair[1]])

    return dfB, vectorB, coocc_diff


def swap_src_dst(src_pair, dst_pair):
  return dst_pair, src_pair

# dfB を元に戻す
def format_anonymized_data(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: replace_back_comp_symbol(x.split("_")[1]))
    return df


## main
if __name__ == "__main__":
    args = sys.argv

    if len(args) != 5:
        print("Usage : python [{}] [samplingfilename] [syntheticfilename] [anonymizefilename] [number of repetition]".format(
            args[0]))
        exit(1)

    samplingfilename = args[1]
    syntheticfilename = args[2]
    anonymizefilename = args[3]
    repetition = int(args[4])

    dfA = pd.read_csv(samplingfilename, header=None, skipinitialspace=True)
    dfB = pd.read_csv(syntheticfilename, header=None, skipinitialspace=True)

    #決定木を適用する
    X_index = [0,1,2,3,4,5,6,7]
    Y_index = [8]
    dfB[8] = replace_value_using_decision_tree(dfA, dfB, X_index, Y_index)

    #共起行列を作る
    dfA_vector = generate_tuple_vectors(dfA)
    dfB_vector = generate_tuple_vectors(dfB)
    dfA_vector, dfB_vector = match_vector_dimensions(dfA_vector, dfB_vector)
        
    coocc_dfA = cooccurrence_matrix(dfA_vector)
    coocc_dfB = cooccurrence_matrix(dfB_vector)
    coocc_diff = coocc_dfB - coocc_dfA

    for i in range(repetition):
        src_pair = select_replace_cooccurence_pair(coocc_diff)
        dst_pair = replace_pair_both_columns(src_pair, coocc_diff)
        print(src_pair)
        print(coocc_diff.at[src_pair[0],src_pair[1]])
        print(dst_pair)
        print(coocc_diff.at[dst_pair[0],dst_pair[1]])
        if(coocc_diff.at[src_pair[0], src_pair[1]]<0):
            src_pair, dst_pair = swap_src_dst(src_pair, dst_pair)
        records = select_replace_records_both_columns(src_pair, dst_pair, coocc_diff, dfB_vector)
        dfB, dfB_vector, coocc_diff = replace_value_pair(records, src_pair, dst_pair, dfB, dfB_vector, coocc_diff)
        if(np.max(np.ravel(coocc_diff))<3):
            break
    print(i)

    dfB_out = format_anonymized_data(dfB)
    dfB_out.to_csv(anonymizefilename,header=False, index=False)

