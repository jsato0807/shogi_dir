import random
import numpy as np
from tensorflow.keras import layers, models, optimizers
from collections import defaultdict

class State:
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):
        # 方向定数
        self.dxy = ((0, 1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))

        #コマの配置
        self.pieces = pieces if pieces != None else [0] * (12 + 3)
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * (12 + 3)
        self.depth = depth

        #コマの初期配置
        if pieces == None or enemy_pieces == None:
            self.pieces = [0,0,0,0,0,0,0,1,0,2,4,3,0,0,0]
            self.enemy_pieces = [0,0,0,0,0,0,0,1,0,2,4,3,0,0,0]
        


    def is_lose(self):
        for i in range(12):
            if self.pieces[i] == 4: #ライオンが盤上にいるかどうか
                return False
        return True
    
    def is_draw(self):
        return self.depth >= 300
    
    def is_done(self):
        return self.is_lose() or self.is_draw()
    
    def pieces_array(self):
        #プレイヤーごとのディアルネットワークの入力の２次元配列を取得
        def pieces_array_of(pieces):
            table_list = []

            #ひよこ０、ぞう１、キリン２、ライオン３
            for j in range(1,5):
                table =  [0] * 12
                table_list.append(table)
                for i in range(12):
                    if self.pieces[i] == j:
                        table[i] = 1

            #ひよこの持ち駒４、像の持ち駒５、キリンの持ち駒６
            for j in range(1,4):
                flag = 1 if pieces[11+j] > 0 else 0
                table = [flag] * 12
                table_list.append(table)
            return table_list
        return [pieces_array_of(self.pieces), pieces_array_of(self.enemy_pieces)]
    
    #コマの移動先と移動元を行動に変換
    def position_to_action(self, position, direction):
        return position * 11 + direction
    
    #行動をコマの移動先と移動元に変換
    def action_to_position(self, action):
        return (action//11, action%11)
    
    #合法手のリストの取得
    def legal_actions(self):
        actions = []
        for p in range(12):
            #コマの移動時
            if self.pieces[p] != 0:
                actions.extend(self.legal_actions_pos(p))

            #持ち駒の配置時
            if self.pieces[p] == 0 and self.enemy_pieces[11-p] == 0:
                for capture in range(1, 4):
                    if self.pieces[11+capture] != 0:
                        actions.append(self.position_to_action(p, 8-1+capture))
        return actions
    
    #コマの移動時の合法手のリストの取得
    def legal_actions_pos(self, position_src):
        actions = []

        #コマの移動可能な方向
        piece_type = self.pieces[position_src]
        if piece_type > 4: piece_type - 4
        directions = []
        if piece_type == 1: #ひよこ
            directions = [0]
        elif piece_type == 2: #ぞう
            directions = [1,3,5,7]
        elif piece_type == 3: #キリン
            directions = [0,2,4,6]
        elif piece_type == 4:
            directions = [0,1,2,3,4,5,6,7]

        # 合法手の取得
        for direction in directions:

            #コマの移動先
            x = position_src%3 + self.dxy[direction][0]             #in reference this point is written as addition
            y = position_src//3 + self.dxy[direction][1]
            p = x + y * 3

            #移動可能時は合法手として追加
            if 0 <= x and x <= 2 and 0 <= y and y <= 3 and self.pieces[p] == 0:
                actions.append(self.position_to_action(p, direction))

        return actions
    
    def next(self, action):
        #次の状態の作成
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth+1)

        #行動を(移動先、移動元)に変換
        position_dst, position_src = self.action_to_position(action)

        #コマの移動
        if position_src < 8:
            #コマの移動元
            x = position_dst%3 - self.dxy[position_src][0]
            y = position_dst//3 - self.dxy[position_src][1]
            position_src = x + y * 3

            #コマの移動
            state.pieces[position_dst] = state.pieces[position_src]
            state.pieces[position_src] = 0

            #相手のコマが存在する時はとる
            piece_type = state.enemy_pieces[11-position_dst]

            if piece_type != 0:
                if piece_type != 4:
                    state.pieces[11+piece_type] += 1  #持ち駒+1
                state.enemy_pieces[11-position_dst] = 0

        #持ち駒の配置
        else:
            capture = position_src-7
            state.pieces[position_dst] = capture
            state.pieces[11+capture] -= 1 #持ち駒-1

        #コマの交代
        w = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = w
        return state
    
    def is_first_player(self):
        return self.depth%2 == 0
    
    def __str__(self):
        pieces0 = self.pieces if self.is_first_player() else self.enemy_pieces
        pieces1 = self.enemy_pieces if self.is_first_player() else self.pieces
        hzkr0 = ('', 'H', 'Z', 'K', 'R')
        hzkr1 = ('', 'h', 'z', 'k', 'r')


            # デバッグ用プリント
        print("pieces0:", pieces0)
        print("pieces1:", pieces1)

        #後手の持ち駒
        result = '['
        for i in range(12,15):
            if pieces1[i] >= 2: result += hzkr1[i-11]
            if pieces1[i] >= 1: result += hzkr1[i-11]
        result += ']\n'

        #print("After enemy pieces:", result)

        #ボード
        for i in range(12):
            if pieces0[i] != 0:
                result += hzkr0[pieces0[i]]
            elif pieces1[11-i] != 0:
                result += hzkr1[pieces1[11-i]]
            else:
                result += '-'
            if i % 3 == 2:
                result += '\n'

        #print("After board construction:", result)

        #先手の持ち駒
        result += '['
        for i in range(12,15):
            if pieces0[i] >= 2: result += hzkr0[i-11]
            if pieces0[i] >= 1: result += hzkr0[i-11]
        result += ']\n'

        #print("Final result:", result)

        return result
    
    

def build_model(input_shape, action_size):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (2, 2), padding="same", activation='relu')(inputs)
    x = layers.Conv2D(128, (2, 2), padding="same" ,activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    
    policy_head = layers.Dense(action_size, activation='softmax', name='policy_head')(x)
    value_head = layers.Dense(1, activation='tanh', name='value_head')(x)
    
    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss={'policy_head': 'categorical_crossentropy', 'value_head': 'mean_squared_error'})
    return model


class MCTS:
    def __init__(self, model, num_simulations):
        self.model = model
        self.num_simulations = num_simulations
        self.Q = defaultdict(lambda: defaultdict(float))  # 行動の価値
        self.N = {}  # 行動の訪問回数
        self.P = {}  # 行動の確率

    def run(self, state):
        for _ in range(self.num_simulations):
            self._simulate(state)

        actions = state.legal_actions()
        counts = [self.N.get((state, action), 0) for action in actions]
        best_action = actions[np.argmax(counts)]
        return best_action

    def _simulate(self, state):
        if state.is_done():
            return -1 if state.is_lose() else 0

        # 選択フェーズ
        next_state, action = self._select(state)
        
        # 評価フェーズ
        value = self._evaluate(next_state)
        
        # 展開フェーズ
        if (next_state, action) not in self.P:
            self._expand(next_state)
        
        # 更新フェーズ
        value = -self._simulate(next_state)
        self._backpropagate(state, action, value)

        return value

    def _select(self, state):
        """UCT (Upper Confidence Bound for Trees) を用いて次の行動を選択"""
        actions = state.legal_actions()
            # Ensure P[state] is a dict
        if state not in self.P:
            self.P[state] = {a: 0 for a in actions}

        total_visits = sum(self.N.get((state, a), 0) for a in actions)
        ucb_values = {
            a: self.Q.get((state, a), 0) + self.P[state][a] * np.sqrt(total_visits) / (1 + self.N.get((state, a), 0))
            for a in actions
        }
        best_action = max(ucb_values, key=ucb_values.get)
        next_state = state.next(best_action)
        return next_state, best_action

    def _evaluate(self, state):
        """ゲームの終局状況を評価"""
        if state.is_done():
            return -1 if state.is_lose() else 0
        return None

    def _expand(self, state):
        """ニューラルネットワークを用いてノードを展開"""
        policy, value = self.model.predict(np.array([state.pieces_array()]))
        policy = policy[0]
        self.P[state] = {a: policy[a] for a in state.legal_actions()}
        return value[0]

    def _backpropagate(self, state, action, value):
        """バックプロパゲーションによって値と訪問回数を更新"""
        self.Q[(state, action)] = self.Q.get((state, action), 0) + (value - self.Q.get((state, action), 0)) / (1 + self.N.get((state, action), 0))
        self.N[(state, action)] = self.N.get((state, action), 0) + 1

def self_play(model, num_games, mcts_simulations):
    memory = []
    for _ in range(num_games):
        state = State()
        mcts = MCTS(model, mcts_simulations)
        game_memory = []

        while not state.is_done():
            action = mcts.run(state)
            game_memory.append((state.pieces_array(), action))
            state = state.next(action)

        reward = -1 if state.is_lose() else 0
        for state_data, action in game_memory:
            action_prob = np.zeros(action_size)
            action_prob[action] = 1
            memory.append((state_data, action_prob, reward))
            reward = -reward

    return memory

def train_model(model, memory, epochs=10, batch_size=64):
    states, policies, rewards = zip(*memory)
    states = np.array(states)
    policies = np.array(policies)
    rewards = np.array(rewards)
    model.fit(states, [policies, rewards], epochs=epochs, batch_size=batch_size)

def alpha_zero_training(model, num_iterations, num_games_per_iteration, mcts_simulations):
    for i in range(num_iterations):
        memory = self_play(model, num_games_per_iteration, mcts_simulations)
        train_model(model, memory)
        print(f"Iteration {i+1}/{num_iterations} completed")




if __name__ == "__main__":
    state = State()
    
    input_shape = (2, 7, 12)  # (channels, height, width)
    action_size = 12 * (8 + 3)  # (positions * directions + drop actions)
    model = build_model(input_shape, action_size)
    # トレーニングの実行
    alpha_zero_training(model, num_iterations=10, num_games_per_iteration=100, mcts_simulations=50)

            


