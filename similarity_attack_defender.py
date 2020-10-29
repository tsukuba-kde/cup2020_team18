import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import sys
import random

# レコードリンケージによるメンバーシップ攻撃
class RecordLinkageAttack():

    # dfAは擬似データ, dfBはサンプリングされたデータ(もしくはそれから加工されたデータ)
    def __init__(self, dfA: pd.DataFrame, dfB: pd.DataFrame):
        self.dfA = dfA
        self.dfB = dfB
        categories = list(np.where(self.dfA.dtypes == "object")[0])
        self.enc = ce.OneHotEncoder(cols=categories, drop_invariant=False)
        self.enc.fit(pd.concat([self.dfA, self.dfB], ignore_index=True))

    def execute(self, k=1):
        encDfA = self.enc.transform(self.dfA).astype("float64").values
        encDfB = self.enc.transform(self.dfB).astype("float64").values

        # KDTreeを構築. 後のステップでクエリ探索をおこなう.
        tree = KDTree(encDfA)
        # 指定したk近傍値を取得する. 近傍の行番号がk個分付与される
        dist, index = tree.query(encDfB, k=k, return_distance=True)
        return dist, index

    def execute_(self, last=100):
        dist, index = self.execute()
        atk = np.hstack([dist, index,list(map(lambda x: [x], list(range(10000))))])
        atk_sort = atk[np.argsort(atk[:, 0])]
        result = atk_sort[atk_sort[:,0]==0]

        return result


## main
if __name__ == "__main__":
    args = sys.argv

    if len(args) != 4:
        print("Usage : python [{}] [samplingfilename] [anonymfilename] [outputfilename]".format(
            args[0]))
        exit(1)

    samplingfilename = args[1]
    anonymfilename = args[2]
    outputfilename = args[3]

    header =  None
    skipinitialspace = True

    df = pd.read_csv(samplingfilename, header=header,
                    skipinitialspace=skipinitialspace)
    anondf = pd.read_csv(anonymfilename, header=header,
                        skipinitialspace=skipinitialspace)

    ra = RecordLinkageAttack(df, anondf)
    attackable_records = ra.execute_()[:,2]
    
    #print(attackable_records)
    # 距離が0になってるレコードの数値属性（age）をちょっと動かしただけ
    for r in attackable_records:
        anondf.iloc[int(r),0]=abs(anondf.iloc[int(r),0]-random.randint(3,5))

    anondf.to_csv(outputfilename,header=False, index=False)
