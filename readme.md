# PWSCup 2020 Team18 Source code

このプログラムは匿名加工コンテスト PWSCup2020 に参加した時に
作成したプログラムです。

コンテストの詳細は
https://www.iwsec.org/pws/2020/cup20.html
をご覧ください。

## Cup2020のお題
* 匿名メンバーシップ推定（通称AMIC）
* 攻撃者が知っているレコードが、匿名化されたデータセットの中に含まれているかどうかを推定する

## Team18のとったアプローチ
* サンプルで提供された擬似データ生成コード(https://www.iwsec.org/pws/2020/Images/PubInfo_20200826.zip ) を使い、擬似データを作る
* 擬似データは有用性をクリアしていないので、有用性をクリアするように調整する(utility_adjuster.py)
* 結果的に元データのレコードと一致しているレコードが数個作られるので、その値を少し摂動する（similarity_attack_defender.py）

## 使い方

(事前にcategory_encoderをインストールしておく必要があります)

1. 手元に匿名化前データ（originaldata.csv）があるとします。
1. サンプルに含まれている擬似データ生成コードを実行し、擬似データファイルを作る。ここでは仮に出力されたデータファイルを pseudodata.csv とします。
1. pseudodata.csvをutlity_adjuster.pyに適用します。ここでは結果をadjusted_pseudodata.csvに出力しています。
```
% python utility_adjuster.py originaldata.csv pseudodata.csv adjusted_pseudodata.csv 6000
```
1. adjusted_pseudodata.csvをsimilarity_attack_defender.pyに適用します。ここでは結果をanonymizeddata.csvに出力しています。
```
% python similarity_attack_defender.py adjusted_pseudodata.csv anonymizeddata.csv
```

