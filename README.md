## Preparation

### Requirements

* TensorFlow
* Pandas
* Numpy
* ipdb

### mount '/share/dataset' on Leto -> local '/share'

E.g.;

	$ sshfs leto:/share/dataset /share

## How to use

	python train4.py

## Memos

### ファイルの説明

* util.py...画像処理のためのファイル
* model4.py..trianのためのファイル
* train4.py...DCGAN本体のファイル

データ保存用にvis10_4というフォルダがある。
保存されるデータは以下の通り

* vis104/sample*.py ... 生成されてきた画像が500epochごとに保存
* vis10_4/array1*.py....経験分布に対するDiscriminatorに対する評価値(密度比)が500epochごとに保存
* vis10_4/array2*.py...生成されたきた分布に対するDiscriminatorに対する評価値(密度比)が5000epochごとに保存
* vis10_4/array3*.py.... Generatorに対するlossが500epochごとに保存
* see_breg.py ... 可視化のためのipynb

### 普通のDCGANとの違い
* cost関数の違い(今回はピアソンdivergence version)
* Discriminatorの出力が2*sigmoid
* learning rateは低め
* 畳み込みが 4*4
