import tensorflow as tf
import time

# 大きな行列サイズ
size = 3000

# CPU での処理時間計測
with tf.device('/CPU:0'):
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    start_time = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f} sec")

# GPU での処理時間計測
with tf.device('/GPU:0'):
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    start_time = time.time()
    c = tf.matmul(a, b)
    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time:.4f} sec")

# 比較結果
print(f"GPU Speedup: {cpu_time / gpu_time:.2f}x faster")
