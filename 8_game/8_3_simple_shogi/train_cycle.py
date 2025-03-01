# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
import tensorflow as tf
from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network

# GPU メモリの動的割り当て（オプション）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # GPU メモリを必要に応じて使用
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # 1つ目のGPUを使用
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)

# デバイス配置のログを出力
tf.debugging.set_log_device_placement(True)


# デバッグ用（不要ならコメントアウト）
# tf.debugging.set_log_device_placement(True)

# デュアルネットワークの作成
print("Creating Dual Network...")
dual_network()
print("Dual Network created successfully.")

# 学習サイクル
num_iterations = 10  # 変更可能
for i in range(num_iterations):
    print(f"========== Training Cycle {i+1}/{num_iterations} ==========")
    
    try:
        print("[Step 1] Self-Play...")
        self_play()
        print("[Step 1] Self-Play Completed.")

        print("[Step 2] Training Network...")
        train_network()
        print("[Step 2] Training Network Completed.")

        print("[Step 3] Evaluating Network...")
        evaluate_network()
        print("[Step 3] Evaluation Completed.")
        
    except Exception as e:
        print(f"Error occurred during training cycle {i+1}: {e}")
        break  # エラー時にループを停止
