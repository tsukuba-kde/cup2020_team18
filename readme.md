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
* サンプルで提供された擬似データ生成コード(https://www.iwsec.org/pws/2020/Images/PubInfo_20200826.zip)を使い、擬似データを作る
* 擬似データは有用性をクリアしていないので、有用性をクリアするように調整する(utility_adjuster.py)
* 結果的に元データのレコードと一致しているレコードが数個作られるので、その値を少し摂動する（similarity_attack_defender.py）




