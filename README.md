## Preparation

### Requirements

* TensorFlow
* Others: Pandas, Numpy, ipdb, invoke, etc.

	$ pip install -r requirements.txt

### mount '/share/dataset' on Leto -> local '/share'

E.g.;

	$ sshfs leto:/share/dataset /share

## How to use

	python train4.py

## How to use (invoke ver.)
	
	$ inv prepare
	$ inv train

for debugging;

	$ ipython tasks.py

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

## Coding feedbacks

* use autopep8 
* Variable names:
 - MSCOCO_TRAIN_PATH -> for const

* use consts
 - 500, 128 -> magic numbers

* WTF!?

```
for start, end in zip(
        range(0, len(lsun_images), batch_size + 15),
        range(batch_size + 15, len(lsun_images), batch_size + 15)
):
``` 

Probably TensorFlow / Scipy have convenient functions for batch split

* separate the model part and training part more explicitly. Following lines can be moved DCBAN class?
 
```
Z_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen, h_real, h_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())

train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

Z_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.initialize_all_variables().run()
```

* Data cropping and reshape should be done before the main task.

```
	batch_images = []
	for i in batch_image_files:
        if len(batch_images) == 128:
            break
        ddd = crop_resize(os.path.join(MSCOCO_TRAIN_PATH, i))
        if ddd.shape == (32, 32, 3):
            batch_images.append(ddd)
```
