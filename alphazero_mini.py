import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DobutsuShogiEnv:
    def __init__(self):
        self.board = self.reset()
        self.hand = {1: [0, 0], -1: [0, 0]}  # 手持ちの駒を格納する
        print("初期ボード配置:\n", self.board)

    def reset(self):
        board = np.zeros((4, 3), dtype=int)
        board[0, 0] = 1  # 左にぞう (elephant)
        board[0, 2] = 2  # 右にきりん (giraffe)
        board[0, 1] = 3  # 真ん中にらいおん (lion)
        board[1, 1] = 4  # らいおんの前にひよこ (chick)
        board[3, 0] = -1  # 左にぞう (elephant)
        board[3, 1] = -2  # 右にきりん (giraffe)
        board[3, 2] = -3  # 真ん中にらいおん (lion)
        board[2, 1] = -4  # らいおんの前にひよこ (chick)
        print("リセット後のボード配置:\n", board)
        return board

    def valid_moves(self, player):
        print(f"プレイヤー {player} の有効な手を計算中...")
        valid_actions = []
        # 駒の移動
        for i in range(4):
            for j in range(3):
                if self.board[i, j] * player > 0:
                    moves = self.get_piece_moves(i, j, player)
                    valid_actions.extend(moves)
        # 手持ちの駒を打つ
        if self.hand[player]:
            for i in range(4):
                for j in range(3):
                    if self.board[i, j] == 0:
                        for piece in self.hand[player]:
                            valid_actions.append(('place', piece, i, j))
        print(f"有効な手: {valid_actions}")
        return valid_actions

    def get_piece_moves(self, x, y, player):
        moves = []
        piece = abs(self.board[x, y])
        directions = []
        if piece == 4:  # ひよこ
            directions = [(1, 0)]
        elif piece == 1:  # ぞう
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif piece == 2:  # きりん
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif piece == 3:  # らいおん
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx * player, y + dy * player
            if 0 <= nx < 4 and 0 <= ny < 3:
                if self.board[nx, ny] * player <= 0:
                    moves.append((x, y, nx, ny))
        print(f"ピース ({x}, {y}) の移動先: {moves}")
        return moves

    def step(self, action, player):
        print(f"アクション: {action} を実行中...")
        if action[0] == 'move':
            x, y, nx, ny = action[1:]
            if self.board[nx, ny] != 0:
                captured_piece = self.board[nx, ny]
                self.hand[player].append(captured_piece)
            self.board[nx, ny] = self.board[x, y]
            self.board[x, y] = 0
        elif action[0] == 'place':
            piece, x, y = action[1:]
            self.board[x, y] = piece
            self.hand[player].remove(piece)
        
        print("移動後のボード配置:\n", self.board)

        if self.is_game_over():
            winner = self.game_result()
            print(f"ゲーム終了! 勝者: {winner}")
        else:
            winner = 0
        return self.board, winner

    def is_game_over(self):
        game_over = self.board[0, :].any() == 3 or self.board[3, :].any() == -3
        print(f"ゲーム終了判定: {game_over}")
        return game_over

    def game_result(self):
        if self.board[0, :].any() == 3:
            print("プレイヤー1の勝ち")
            return 1
        elif self.board[3, :].any() == -3:
            print("プレイヤー2の勝ち")
            return -1
        print("引き分け")
        return 0

def build_model():
    print("モデルの構築...")
    inputs = layers.Input(shape=(4, 3, 1))

    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)

    policy = layers.Dense(204, activation='softmax')(x)  # 204アクションの確率分布
    value = layers.Dense(1, activation='tanh')(x)  # 状態の勝率予測

    model = models.Model(inputs=inputs, outputs=[policy, value])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=['categorical_crossentropy', 'mean_squared_error'])
    print("モデル構築完了")
    return model

class MCTS:
    def __init__(self, model, game):
        self.model = model
        self.game = game
        self.tree = {}
        print("MCTS初期化完了")

    def selection(self, state, player):
        print("ノード選択中...")
        if state.tostring() in self.tree:
            return self.tree[state.tostring()]
        return None

    def expansion(self, state, player):
        print("ノード拡張中...")
        policy, value = self.model.predict(state.reshape(1, 4, 3, 1))
        self.tree[state.tostring()] = (policy, value)
        print(f"拡張結果 - ポリシー: {policy}, バリュー: {value}")
        return policy, value

    def simulation(self, state, player):
        print("シミュレーション中...")
        policy, value = self.expansion(state, player)
        valid_actions = self.game.valid_moves(player)
        action_probs = policy[0]
        best_action = valid_actions[np.argmax(action_probs)]
        print(f"シミュレーション結果 - 最良アクション: {best_action}, バリュー: {value}")
        return best_action, value

    def backpropagation(self, path, value):
        print("バックプロパゲーション中...")
        for state in reversed(path):
            node = self.tree[state]
            node[1] += value
            node[2] += 1
        print("バックプロパゲーション完了")

    def search(self, state, player):
        print("探索中...")
        path = []
        node = self.selection(state, player)
        if node is None:
            policy, value = self.expansion(state, player)
        else:
            policy, value = node
        action, value = self.simulation(state, player)
        path.append(state.tostring())
        self.backpropagation(path, value)
        print(f"探索結果 - アクション: {action}")
        return action

    def select_action(self, state, player):
        print("アクション選択中...")
        action = self.search(state, player)
        print(f"選択アクション: {action}")
        return action

def collect_random_play_data(env, num_games=100):
    print(f"ランダムプレイデータ収集中 ({num_games} ゲーム)...")
    states, actions, results = [], [], []

    for _ in range(num_games):
        state = env.reset()
        player = 1
        game_data = []

        while not env.is_game_over():
            valid_actions = env.valid_moves(player)
            action = valid_actions[np.random.choice(len(valid_actions))]
            next_state, result = env.step(action, player)
            game_data.append((state, action, result))
            state = next_state
            player *= -1

        for state, action, result in game_data:
            states.append(state)
            actions.append(action)
            results.append(result)

    print("ランダムプレイデータ収集完了")
    return np.array(states), np.array(actions), np.array(results)

def collect_mcts_play_data(env, model, mcts, num_games=100):
    print(f"MCTSプレイデータ収集中 ({num_games} ゲーム)...")
    states, actions, results = [], [], []

    for _ in range(num_games):
        state = env.reset()
        player = 1
        game_data = []

        while not env.is_game_over():
            action = mcts.select_action(state, player)
            next_state, result = env.step(action, player)
            game_data.append((state, action, result))
            state = next_state
            player *= -1

        for state, action, result in game_data:
            states.append(state)
            actions.append(action)
            results.append(result)

    print("MCTSプレイデータ収集完了")
    return np.array(states), np.array(actions), np.array(results)

def train(model, env, mcts, episodes, new_inputs, new_policies, new_values):
    print(f"モデルのトレーニングを開始 ({episodes} エピソード)...")
    for episode in range(episodes):
        state = env.reset()
        player = 1
        while not env.is_game_over():
            action = mcts.select_action(state, player)
            next_state, result = env.step(action, player)
            state = next_state
            player *= -1

    model.fit(new_inputs, [new_policies, new_values], epochs=10, batch_size=32)
    print("モデルのトレーニング完了")

# 環境とモデルの初期化
env = DobutsuShogiEnv()
model = build_model()
mcts = MCTS(model, env)

# ランダムプレイデータの収集
random_states, random_actions, random_outcomes = collect_random_play_data(env, num_games=100)
print("ランダムプレイデータのサイズ:", random_states.shape, random_actions.shape, random_outcomes.shape)

# モデルのトレーニング
train(model, env, mcts, episodes=10, new_inputs=random_states, new_policies=random_actions, new_values=random_outcomes)

# MCTSプレイデータの収集
mcts_states, mcts_actions, mcts_results = collect_mcts_play_data(env, model, mcts, num_games=100)
print("MCTSプレイデータのサイズ:", mcts_states.shape, mcts_actions.shape, mcts_results.shape)
