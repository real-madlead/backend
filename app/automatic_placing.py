


class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.fc(x)

def get_high_score_indices(model_path, test_df):
    # データフレームをテストデータに変換
    X_test = torch.tensor(test_df.values, dtype=torch.float32)

    # 保存したモデルを読み込む
    model = Net(X_test.shape[1])  # モデルのインスタンスを作成
    model.load_state_dict(torch.load(model_path))  # 保存したモデルのパラメータを読み込む
    model.eval()  # モデルを評価モードに設定

    # X_testデータを使って予測を行う
    with torch.no_grad():
        predictions = model(X_test)


    # 予測結果をPyTorchのテンソルからnumpy配列に変換
    predictions_list = predictions.numpy().flatten().tolist()
    print(f'prediction_list: {predictions_list}')

    # リスト内の要素が閾値を超える場合、そのインデックスを取得
    max_index = predictions_list.index(max(predictions_list))

    max_score = predictions_list[max_index]

    print(f'max_score: {max_score}')
    print(f'max_index: {max_index}')
    return max_index, max_score

def squeeze_room(df):
    """AIによる絞り込み(未実装)
    Parameters
    ---------
    df : pd.DataFrame
        複数の家具配置がはいったdataframe
    
    Returns
    ------
    index : int
        ベストな家具の配置パターンのindex値
    """
    #for i in df.columns:
    #    print(i)
    print(f'''----------------->{df.shape[1]}''')
    model_path = './AI_model/torch_model.pth'
    df_test = df.drop(['room_num', 'target'], axis=1)
    index, score = get_high_score_indices(model_path, df_test)
    
    return index, score

def predict(model_path, df)


def get_position(name:str, name_counter:dict, series):
    """家具の置き場所を探す関数
    Parameters
    ---------
    name : str
        家具の名前
    name_counter : dict
        各家具の出現数を数えるための辞書
    series : pd.Series
        ベストな家具配置を表すdataframeの一行
    
    Returns
    ------
    x, y : float
        家具の配置場所
    """
    cur_name = f'''{name}_{name_counter[name]}'''
    x, y, rotation = series[f'''{cur_name}_x'''], series[f'''{cur_name}_y'''], series[f'''{cur_name}_rotation''']
    return x, y, rotation

def recommend_furniture_using_AI(
        candidate_furnitures_for_additional_placement,
        current_furniture_layout,
        current_best_score
                                 ):
    """AIにより渡された部屋に新たな家具を配置するようにおすすめする
    Parameters
    ---------
    candidate_furnitures_for_additional_placement : list(FURNITURE)
        追加で配置する家具の候補
    current_furniture_layout : FloorPlanOutputSchema
        現在の家具配置
    current_score : flaot
        現在の最高スコア
    
    Returns
    ------
    recommend_furnitureplace : FurniturePlace
        AIが最適だと考えた家具の配置
    score : float
        おすすめの家具を配置した場合の部屋のスコア 
    """
    edges = [
        [0, 0],
        [0, current_furniture_layout.floor.length],
        [current_furniture_layout.floor.width, current_furniture_layout.floor.length],
        [current_furniture_layout.floor.width, 0]
    ]
    room = Room(edges=edges)
    room.plot_room()
    furnitures_list = [Furniture(v_width=furnitureplace.length, h_width=furnitureplace.width, rotation=furnitureplace.rotation, name=furnitureplace.name) for furnitureplace in current_furniture_layout.furnitures]
    furnitures_coord_list = [[furniture_coord.x, furniture_coord.y] for furniture_coord in current_furniture_layout.furnitures]
    _ = room.plot_furniture(furnitrues=furnitures_list, furnitures_coord=furnitures_coord_list)
    while True:
        for candidate_furniture in candidate_furnitures_for_additional_placement:
            new_placement_furniture = [Furniture(v_width=candidate_furniture.length, h_width=candidate_furniture.width, rotation)]
            e_flag =  room.plot_furniture()
            if e_flag[0]== :
                continue
            #AIによる採点
            if current_best_score <= score:
                return recommend_furnitureplace, score

