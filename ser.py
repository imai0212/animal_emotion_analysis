# モデルの構築・評価

from sklearn.neural_network import MLPClassifier # 多層パーセプトロンモデル
from sklearn.metrics import accuracy_score # 精度計測

from utils import load_data
import os
import pickle # 学習済みモデルの保存


# データ(75%：訓練, 25%テスト)n
X_train, X_test, y_train, y_test = load_data(test_size=0.15)
# 訓練データのサンプル数
print("[+] 訓練用サンプル数:", X_train.shape[0])
# テストデータのサンプル数
print("[+] テスト用サンプル数:", X_test.shape[0])
# 使用した特徴量の数（抽出された特徴量のベクトル）
print("[+] 特徴量数:", X_train.shape[1])
# グリッドサーチで決定された最適なモデル
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}
# 多層パーセプトロン分類器の初期化(現時点での最適なパラメータ)
model = MLPClassifier(**model_params)

# モデルの訓練
print("[*] モデルの訓練中...")
model.fit(X_train, y_train)

# 25%のデータを予測
y_pred = model.predict(X_test)

# 精度の計算
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("モデルの精度: {:.2f}%".format(accuracy*100))

# モデルの保存
# resultディレクトリが存在しない場合は作成
if not os.path.isdir("result"):
    os.mkdir("result")
pickle.dump(model, open("result/mlp_classifier.model", "wb"))
