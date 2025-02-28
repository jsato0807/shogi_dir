import numpy as np
from tensorflow.keras import layers, models, optimizers

class DobutsuShogiEnv:
    def __init__(self):
        self.board = self.reset()
        self.winner = 0
        print("初期ボード配置:\n", self.board)

    def reset(self):
        board = np.zeros((2, 7, 4, 3), dtype=int)
        
        # 自分の駒を配置
        board[0, 0, 1, 1] = 1  # 自分のヒヨコ (chick)
        board[0, 1, 0, 0] = 1  # 自分のゾウ (elephant)
        board[0, 2, 0, 2] = 1  # 自分のキリン (giraffe)
        board[0, 3, 0, 1] = 1  # 自分のライオン (lion)

        # 相手の駒を配置
        board[1, 0, 2, 1] = -1  # 相手のヒヨコ (chick)
        board[1, 1, 3, 0] = -1  # 相手のゾウ (elephant)
        board[1, 2, 3, 2] = -1  # 相手のキリン (giraffe)
        board[1, 3, 3, 1] = -1  # 相手のライオン (lion)

        print("リセット後のボード配置:\n", board)
        return board

    def valid_moves(self, player):
        print(f"プレイヤー {player} の有効な手を計算中...")
        valid_actions = []
        player_index = 0 if player == 1 else 1
        
        # 駒の移動
        for i in range(4):
            for j in range(3):
                piece_positions = np.nonzero(self.board[player_index, :, i, j])[0]
                for piece in piece_positions:
                    moves = self.get_piece_moves(i, j, piece, player)
                    for move in moves:
                        valid_actions.append((move))  
        
        # 手持ちの駒を打つ
        empty_positions = np.argwhere(self.board[:, :, :3, player_index] == 0)
        for piece in [0, 1, 2]:
            if np.any(self.board[player_index, piece, :, :]) == 0:  # 持ち駒がある
                for (i, j) in empty_positions:
                    valid_actions.append((piece + 5, i * 3 + j))  # (手持ち駒の種類、移動先)
        
        print(f"有効な手: {valid_actions}")
        return valid_actions

    def get_piece_moves(self, x, y, piece, player):
        moves = []
        piece_type = piece % 4
        directions = []
        if piece_type == 0:  # ひよこ
            directions = [(1, 0)] if player == 1 else [(-1, 0)]
        elif piece_type == 1:  # ぞう
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif piece_type == 2:  # きりん
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif piece_type == 3:  # らいおん
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        
        player_index = 0 if player == 1 else 1
        for direction in directions:
            dx, dy = direction
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 3:
                if not np.any(self.board[player_index, :, nx, ny]):  # 自分の駒がない場合
                    moves.append((dx * 3 + dy, nx * 3 + ny))  # 移動方向と移動先
        
        print(f"ピース ({x}, {y}) の移動先: {moves}")
        return moves

    def step(self, action, player):
        print(f"アクション: {action} を実行中...")
        player_index = 0 if player == 1 else 1
        
        if action[0] < 5:  # 駒の移動
            direction, move_to = action
            dx, dy = divmod(direction, 3)
            to_x, to_y = divmod(move_to, 3)
            from_x, from_y = to_x - dx, to_y - dy
            piece_type = np.nonzero(self.board[player_index, :, from_x, from_y])[0][0]
            
            opponent_index = 1 - player_index

            if self.board[opponent_index, 3, to_x, to_y] == -1:  # ライオンを取った場合
                self.board[player_index, piece_type, to_x, to_y] = 1
                self.board[player_index, piece_type, from_x, from_y] = 0
                print(f"プレイヤー {player} が相手のライオンを取りました!")
                #return self.board, player  # 勝者は現在のプレイヤー
            elif np.any(self.board[opponent_index, :3, to_x, to_y]):
                captured_piece = np.nonzero(self.board[opponent_index, :3, to_x, to_y])[0][0]
                self.board[player_index, piece_type, to_x, to_y] = 1
                self.board[opponent_index, captured_piece, to_x, to_y] = 0
            self.board[player_index, piece_type, to_x, to_y] = 1
            self.board[player_index, piece_type, from_x, from_y] = 0
        else:  # 持ち駒の配置
            piece_type = action[0] - 5
            x, y = divmod(action[1], 3)
            self.board[player_index, piece_type, x, y] = 1
        
        print(f"プレイヤー {player} のアクション: {action}")
        print("移動後のボード配置:\n", self.board)

        # ゲーム終了の確認
        if self.is_game_over(player):
            winner = self.game_result(player)
            self.reset()  # ゲーム終了時にリセット
            return self.board, winner
        return self.board, 0

    def is_game_over(self, player):
        if np.any(self.board[0, 3, :, :]) or np.any(self.board[1, 3, :, :]):
            print(f"ゲーム終了判定:トライ成功")
            self.winner = player
            return True

        # ライオンが取られたかどうか
        if np.all(self.board[0, 3, :, :] == 0) or np.all(self.board[1, 3, :, :] == 0):
            print(f"ゲーム終了判定:ライオンが取られた")
            self.winner = player
            return True

        # 詰みのチェック
        if self.is_checkmated(1) or self.is_checkmated(-1):
            print(f"ゲーム終了判定:詰み")
            return True
        
        return False

    def is_checkmated(self, player):
        lion_value = 3 if player == 1 else -3
        lion_position = np.nonzero(self.board[0 if player == 1 else 1, 3, :, :])

        x, y = lion_position[0][0], lion_position[1][0]
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 3:
                if not self.is_under_attack(nx, ny, player):
                    return False

        print(f"ライオンは詰みです。")
        opponent = -player
        self.winner = opponent
        return True

    def is_under_attack(self, x, y, player):
        opponent_index = 1 - (0 if player == 1 else 1)
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for piece in range(3):
            for direction in directions:
                from_x, from_y = np.array((x, y)) - np.array(direction)
                piece_moves = self.get_piece_moves(from_x, from_y, piece, -player)
                for move in piece_moves:
                    move_dir, move_pos = move
                    if move_pos == (x * 3 + y):
                        return True
        return False

    def game_result(self, player):
        if np.any(self.board[player_index, 3, 0, :]) and player == self.winner:
            print(f"プレイヤー {player} が勝ちました!")
            return player  # 勝者
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

        while not env.is_game_over(player):
            valid_actions = env.valid_moves(player)
            if not valid_actions:
                break
            action = valid_actions[np.random.choice(len(valid_actions))]
            next_state, result = env.step(action, player)
            game_data.append((state, action, result))
            state = next_state
            player *= -1

        for state, action, result in game_data:
            states.append(state)
            actions.append(action)
            results.append(result)

    # デバッグ: 各リストのサイズを確認
    print(f"states: {np.array(states).shape}")
    print(f"actions: {actions}")
    print(f"results: {np.array(results).shape}")

    print("ランダムプレイデータ収集完了")
    return states, actions, results

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
        while not env.is_game_over(player):
            action = mcts.select_action(state, player)
            next_state, result = env.step(action, player)
            state = next_state
            player *= -1
    print(f'new_inputs:{new_inputs}')
    print(f'new_polices:{new_policies}')
    print(f'new_values:{new_values}')
    
    model.fit(new_inputs, [new_policies, new_values], epochs=10, batch_size=32)
    print("モデルのトレーニング完了")


if __name__ == "__main__":
    # 環境とモデルの初期化
    env = DobutsuShogiEnv()
    model = build_model()
    mcts = MCTS(model, env)

    # ランダムプレイデータの収集
    random_states, random_actions, random_outcomes = collect_random_play_data(env, num_games=100)
    #print("ランダムプレイデータのサイズ:", random_states.shape, random_actions.shape, random_outcomes.shape)

    # モデルのトレーニング
    train(model, env, mcts, episodes=10, new_inputs=random_states, new_policies=random_actions, new_values=random_outcomes)

    # MCTSプレイデータの収集
    mcts_states, mcts_actions, mcts_results = collect_mcts_play_data(env, model, mcts, num_games=100)
    print("MCTSプレイデータのサイズ:", mcts_states.shape, mcts_actions.shape, mcts_results.shape)
