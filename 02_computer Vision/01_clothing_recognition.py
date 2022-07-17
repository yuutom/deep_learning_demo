import tensorflow as tf

# 目標：28x28ピクセルの配列を入力し、中間層のニューロンが重みとバイアス（m値とc値）を持ち、
# それらを組み合わせることで、入力したピクセルを10個の出力値の内の1つに一致させること

data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# /255で配列全体に演算を適用
# 画像内の全てのピクセルが0~255の値のグレースケール→0~1で表現（正規化）
# 正規化されたデータの方がニューラルネットワークの学習に適している
training_images = training_images/255
test_images = test_images/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',  # オプティマイザの選択（'sgd'(確率的勾配降下法)の進化系）
              loss='sparse_categorical_crossentropy',  # 損失関数の選択
              metrics=['accuracy'])  # メトリクスの選択（学習中のネットワークの正解率を確認）
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)


classifications = model.predict(test_images)
# 10個の出力ニューロンの値（画像がそのインデックスのラベルと一致する確率）
print(classifications[0])
print(test_labels[0])
