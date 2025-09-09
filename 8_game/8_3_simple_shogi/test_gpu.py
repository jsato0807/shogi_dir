import tensorflow as tf

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# GPU に配置されているか確認
tf.debugging.set_log_device_placement(True)

# ダミーデータを作成
a = tf.constant([[1.0, 2.0, 3.0]])
b = tf.constant([[4.0], [5.0], [6.0]])

# 行列積を計算（GPU で実行されるか確認）
c = tf.matmul(a, b)
print(c)
