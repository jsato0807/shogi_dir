import random
from tensorflow.keras import layers, models, optimizers

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
        def pieces_array_of(self):
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
                flag = 1 if self.pieces[11+j] > 0 else 0
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
    
    

def random_action(state):
    legal_actions = state.legal_actions()
    print(f"legal_actions:", legal_actions)
    return legal_actions[random.randint(0, len(legal_actions)-1)]


def collect_random_play_data(state,num_games=10):
    pieces_arrays, enemy_pieces_arrays = [], []
    for _ in range(num_games):
        while True:
            if state.is_done():
                break
    
            pieces_array, enemy_pieces_array = state.pieces_array()
            pieces_arrays.append(pieces_array)
            enemy_pieces_arrays.append(enemy_pieces_array)

            #次の状態の取得
            state = state.next(random_action(state))

            #文字列表示
            print(state)
            print()
    return pieces_arrays, enemy_pieces_arrays


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

def build_model():
    print("モデルの構築...")
    inputs = layers.Input(shape=(3,4,14))

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

def train(model, state, mcts, episodes):
    print(f"モデルのトレーニングを開始 ({episodes} エピソード)...")
    for episode in range(episodes):
        state = state.reset()
        player = 1
        while not state.is_done():
            action = mcts.select_action(state, player)
            pieces_arrays, enemy_pieces_arrays = state.next(action)
            player *= -1
    
    model.fit(new_inputs, [new_policies, new_values], epochs=10, batch_size=32)
    print("モデルのトレーニング完了")


if __name__ == "__main__":
    state = State()

            


