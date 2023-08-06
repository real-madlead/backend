from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate
import math
import random
import pandas as pd
import os
import copy
import re
import torch
import torch
from torch import nn
from torch.autograd import Variable
from app.schemas import Furniture as FURNITURE
import copy

class Furniture():
    """家具クラス
    """
    def __init__(self, v_width:float, h_width:float, rotation:int=0, name:str=None, color:str=None):
        self.v_width = v_width
        self.h_width = h_width
        self.rotation = rotation
        self.name = name
        self.color = color

class Room():
    """部屋クラス
    """
    def __init__(self, edges:list, windows:list=None, doors:list=None):
        """
        Parameters
        ----------
        edges : list
            部屋の角の座標(左下を始点に時計周りで記述)[[x座標, y座標], [float, float], ]
        windows : list
            窓の情報[
            {"start":[x座標, y座標], "end":[x座標, y座標]},
            {"start":[x座標, y座標], "end":[x座標, y座標]},
            ]
        doors : list
            ドアの情報[
            {"start":[x座標, y座標], "end":[x座標, y座標]},
            {"start":[x座標, y座標], "end":[x座標, y座標]},,
            ]
        """
        self.edges = edges
        self.windows = windows
        self.doors = doors
        self.line_objects = []
        self.furniture_objects = []
        
    def plot_room(self):
        """家具を抜きにした部屋と窓、ドアを描画するメソッド
        """
        x_coords = [edge[0] for edge in self.edges]
        y_coords = [edge[1] for edge in self.edges]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        points = [lst for lst in self.edges]
        if self.windows!=None:
            wind_starts = [dic["start"] for dic in self.windows]
            wind_ends = [dic["end"] for dic in self.windows]
            points += wind_starts + wind_ends
            winds = wind_starts + wind_ends 
        
        if self.doors!=None:
            door_starts = [dic["start"] for dic in self.doors]
            door_ends = [dic["end"] for dic in self.doors]
            points += door_starts + door_ends
            doors = door_starts + door_ends
            
        
        points = sort_points(points)
        last_ps = [points[-1], points[0]]
        last_line = {"x":[last_ps[0][0], last_ps[1][0]], "y":[last_ps[0][1], last_ps[1][1]]}
        if (self.windows!=None) and (last_ps[0] in winds) and (last_ps[1] in winds):
            last_line["color"] = "blue"
        elif (self.doors!=None) and (last_ps[0] in doors) and (last_ps[1] in doors):
            last_line["color"] = "red"
        else:
            last_line["color"] = "k"
        lines = [
            last_line
        ]
        for i in range(len(points)-1):
            ps = points[i:i+2]
            line = {"x":[ps[0][0], ps[1][0]], "y":[ps[0][1], ps[1][1]]}
            if (self.windows!=None) and (ps[0] in winds) and (ps[1] in winds):
                line["color"] = "blue"
            elif (self.doors!=None) and (ps[0] in doors) and (ps[1] in doors):
                line["color"] = "red"
            else:
                line["color"] = "k"
            lines.append(line)
        for line in lines:
            calculate_line = create_line(start=[line["x"][0], line["y"][0]], end=[line["x"][1], line["y"][1]], color=line["color"])
            self.line_objects.append(calculate_line)
        
    def plot_furniture(self, furnitures:list, furnitures_coord:list):
        """家具を配置するメソッド

        Parameters
        ----------
        furnitures : list
            家具オブジェクトが入ったリスト
        furniture_coord : list
            家具オブジェクトの位置が入ったリスト

        Returns
        ------
        error_flag : list
            配置した家具の状態を表した数字が格納されたリスト(1:壁と重ねっている、2:他の家具に重なっている、0:正常に配置されている)
        """
        error_flag = list()
        x_coords = [edge[0] for edge in self.edges]
        y_coords = [edge[1] for edge in self.edges]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        for furniture, coord in zip(furnitures, furnitures_coord): 
            calculate_furniture = create_rectangle(coord, furniture.h_width, furniture.v_width, furniture.rotation, furniture.color)
        
            if (multi_check_overlap(calculate_furniture, self.line_objects)) or (coord[0]<=min_x) or (coord[0]>=max_x) or (coord[1]<=min_y) or (coord[1]>=max_y):
                error_flag.append(1)
            elif multi_check_overlap(calculate_furniture, self.furniture_objects):
                error_flag.append(2)
            else:
                error_flag.append(0)
            self.furniture_objects.append(calculate_furniture)
        return error_flag

    def clear_furniture(self, furniture_index:int=None, all_clear:bool=False):
        """配置した家具を削除するメソッド

        Parameters
        ---------
        furniture_index : int
            削除したい家具のfurnitureにおけるインデックス値
        all_clear : bool
            描画した家具全てを削除するかどうか
        """
        if furniture_index!=None:
            del self.furniture_objects[furniture_index]
        if all_clear:
            self.furniture_objects = list()
    
    def random_plot_furniture(self, random_furniture:list):
        """家具を部屋、他の家具とかさならないように配置するメソッド

        Parameters
        ---------
        random_furniture : list
            ランダムに配置する家具の情報が辞書オブジェクトで入ったリスト
        
        Returns
        -------
        furniture_info : list
            各家具の情報が記録してある辞書オブジェクトが入ってるリスト
        """
        x_coords = [edge[0] for edge in self.edges]
        y_coords = [edge[1] for edge in self.edges]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        furniture_info = list()
        max_attempts = 10000
        for _ in range(max_attempts):
            restart = False  # ループを再開するかどうかをチェックするフラグ
            for f_dic in random_furniture:
                dic = dict()
                name = f_dic['name']
                if f_dic["exist"]==1:
                    counter = 0
                    while True:
                        dic["name"] = name
                        dic['exist'] = 1
                        dic["v_width"] = f_dic["length"]#家具の長さ追加した
                        dic["h_width"] = f_dic["width"]
                        delta = 0.01
                        if ("restriction" in f_dic) and ("alongwall" in f_dic["restriction"]):
                            rand = random.choice([0, 1])
                            if rand == 0:
                                dic["x"], dic["y"] = random.choice([min_x + delta, max_x - delta]), random.uniform(min_y, max_y)
                            elif rand == 1:
                                dic["x"], dic["y"] = random.uniform(min_x, max_x), random.choice([min_y + delta, max_y - delta])
                            dic["rotation"] = random.choice(f_dic["rotation_range"])
                        elif ("restriction" in f_dic) and ("set" in f_dic["restriction"]):
                            f_dic["set_furniture"] = "desk"
                            set_furnitures = filtered_furniture = [item for item in furniture_info if re.match(f_dic["set_furniture"] + "_" + r'\d+', item['name'])]
                            if len(set_furnitures) == 0:#setする家具が配置されていない場合その家具も配置されない
                                rand = 4       
                            elif len(set_furnitures) != 0:
                                set_furniture = random.choice(set_furnitures)
                                set_f_x, set_f_y, set_f_rotation = set_furniture["x"] + delta*math.sin(math.radians(set_furniture["rotation"])), set_furniture["y"] - delta*math.cos(math.radians(set_furniture["rotation"])), set_furniture["rotation"]
                                rand = random.choice([0, 1, 2, 3])

                            if rand == 0:
                                set_f_rand_len = random.uniform(0, set_furniture["h_width"] - dic["v_width"])
                                dic["x"] = set_f_x + set_f_rand_len*math.cos(math.radians(set_f_rotation)) + dic["v_width"]*math.cos(math.radians(set_f_rotation)) + dic["h_width"]*math.sin(math.radians(set_f_rotation))
                                dic["y"] = set_f_y + set_f_rand_len*math.sin(math.radians(set_f_rotation)) + dic["v_width"]*math.sin(math.radians(set_f_rotation)) - dic["h_width"]*math.cos(math.radians(set_f_rotation))
                                dic["rotation"] = set_f_rotation + 90
                            elif rand == 1:
                                set_f_rand_len = random.uniform(0, set_furniture["v_width"] - dic["v_width"])
                                dic["x"] = set_f_x + set_furniture["h_width"]*math.cos(math.radians(set_f_rotation)) - set_f_rand_len*math.sin(math.radians(set_f_rotation)) + dic["h_width"]*math.cos(math.radians(set_f_rotation)) - dic["v_width"]*math.sin(math.radians(set_f_rotation))
                                dic["y"] = set_f_y + set_furniture["h_width"]*math.sin(math.radians(set_f_rotation)) + set_f_rand_len*math.cos(math.radians(set_f_rotation)) + dic["h_width"]*math.sin(math.radians(set_f_rotation)) + dic["v_width"]*math.cos(math.radians(set_f_rotation))
                                dic["rotation"] = set_f_rotation + 180
                            elif rand == 2:
                                set_f_rand_len = random.uniform(0, set_furniture["h_width"] - dic["v_width"])
                                dic["x"] = set_f_x + set_f_rand_len*math.cos(math.radians(set_f_rotation)) - dic["h_width"]*math.sin(math.radians(set_f_rotation)) - set_furniture["v_width"]*math.sin(math.radians(set_f_rotation))
                                dic["y"] = set_f_y + set_f_rand_len*math.sin(math.radians(set_f_rotation)) + dic["h_width"]*math.cos(math.radians(set_f_rotation)) + set_furniture["v_width"]*math.cos(math.radians(set_f_rotation))
                                dic["rotation"] = set_f_rotation - 90
                            elif rand == 3:
                                set_f_rand_len = random.uniform(0, set_furniture["v_width"] - dic["v_width"])
                                dic["x"] = -1*set_f_rand_len*math.sin(math.radians(set_f_rotation)) - dic["h_width"]*math.cos(math.radians(set_f_rotation))
                                dic["y"] = set_f_rand_len*math.cos(math.radians(set_f_rotation)) - dic["h_width"]*math.sin(math.radians(set_f_rotation))
                                dic["rotation"] = set_f_rotation
                            elif rand == 4:#setする家具がない場合
                                dic["x"], dic["y"] = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
                                dic["rotation"] = dic["rotation"] = random.choice(f_dic["rotation_range"])
                        elif (f_dic["restriction"]==""):
                            dic["x"], dic["y"] = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
                            dic["rotation"] = dic["rotation"] = random.choice(f_dic["rotation_range"])
                        fur = Furniture(dic["v_width"], dic["h_width"], dic["rotation"], f_dic["name"], None)
                        error_flag = self.plot_furniture([fur], [[dic["x"], dic["y"]]])
                        if error_flag[0]!=0:
                            self.clear_furniture(furniture_index=-1)
                            counter += 1
                        elif error_flag[0]==0:
                            furniture_info.append(dic)
                            break
                        if counter>50:#50回以上エラーが出たら論理的におけないと判断し、もう一度全ての家具を置きなおす
                            self.clear_furniture(all_clear=True)
                            restart = True
                            break
                    if restart:
                        break
                    
                else:
                    dic["name"] = name
                    dic['exist'] = 0
                    dic["x"], dic["y"] = 0, 0
                    dic["rotation"] = 0
                    dic["v_width"] = 0
                    dic["h_width"] = 0
                    furniture_info.append(dic)
            if not restart:  # もし再開フラグがFalseの場合、外部ループを終了
                break
        return furniture_info

def create_rectangle(center, width, height, angle, color):
    """四角形を作成する関数
    Returns
    -------
    rectangle : matplotlib.patches.Rectangle
        描画するための四角形のオブジェクト
    rectangle_polygon : shapely.geometry.polygon.Polygon
        あたり判定を計算するために仮想的に作成された四角形
    """
    rectangle_coordinates = [(center[0], center[1]), 
                             (center[0], center[1] + height), 
                             (center[0] + width, center[1] + height), 
                             (center[0] + width, center[1])]
    rectangle_polygon = Polygon(rectangle_coordinates)
    rectangle_polygon = rotate(rectangle_polygon, angle, origin=(center[0], center[1]))
    return rectangle_polygon


def create_line(start, end, color):
    """部屋の枠を表す線を作成する関数
    Returns
    -------
    line_polygon : shapely.geometry.LineString
        あたり判定を計算するために仮想的に作成された四角形
    """
    line_polygon = LineString([(start[0], start[1]), (end[0], end[1])])
    return line_polygon

def trigonometric_addition_sin(sin_a:float, cos_a:float, b:int):
    """sin(a + b) = sin(a)cos(b) + cos(a)sin(b)を計算します
    """
    return sin_a * math.cos(math.radians(b)) + cos_a * math.sin(math.radians(b))

def trigonometric_addition_sin_minus(sin_a:float, cos_a:float, b:int):
    """sin(a - b) = sin(a)cos(b) - cos(a)sin(b)を計算します
    """
    return sin_a * math.cos(math.radians(b)) - cos_a * math.sin(math.radians(b))

def trigonometric_addition_cos(sin_a:float, cos_a:float, b:int):
    """cos(a + b) = cos(a)cos(b) - sin(a)sin(b)を計算します"""
    return cos_a * math.cos(math.radians(b)) - sin_a * math.sin(math.radians(b))

def trigonometric_addition_cos_minus(sin_a:float, cos_a:float, b:int):
    """cos(a - b) = cos(a)cos(b) + sin(a)sin(b)を計算します"""
    return cos_a * math.cos(math.radians(b)) + sin_a * math.sin(math.radians(b))
    
def find_center(points):
    """中心を見つけるための関数
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    return (center_x, center_y)


def calculate_angle(point, center):
    """角度を計算する関数
    """
    return math.atan2(point[1] - center[1], point[0] - center[0])

def sort_points(points):
    """各点を角度に基づいてソートする関数
    """
    center = find_center(points)
    return sorted(points, key=lambda point: calculate_angle(point, center))

def multi_check_overlap(obj1, objs2:list):
    """あたり判定を計算する関数
    
    Parameters
    ---------
    obj1 : shapely.geometry.polygon.Polygon
        あたり判定を計算するために作成された四角形のオブジェクト
    obj2 : list
        あたり判定を計算するために作成された四角形のオブジェクトが複数格納されたリスト
    """
    for obj in objs2:
        if obj.intersects(obj1):
            return True
        else:
            continue
    return False
   
def find_max_values(arr):
    max_val_1 = max(arr, key=lambda x: x[0])[0]
    max_val_2 = max(arr, key=lambda x: x[1])[1]
    return max_val_1, max_val_2
   
def calculate_distance(p1, p2):
    """2点の座標の距離を計算する関数
    Parameters
    ---------
    p1, p2 : dict
        x,yのキーが入っている辞書
    """
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

def find_dict_by_name(dict_list, name, selfdict):
    """家具同士の距離を算出したカラムを作成する際に使用した関数

    Parameters
    ---------
    dict_list : list
        他の家具の情報が辞書形式で入ったリスト
        ex) [{"name":"bed_1", "x":4, "y":3}, {"name":"sofa_1", "x":8, "y":2}]
    name : str
        selfdictの家具との距離を測りたい家具の名前
        ex) bed_1
    selfdict : dict
        主観的な家具
        ex) {"name":"sofa_1", "x":8, "y":2}
    
    Returns
    ------
    distance : float
        二つの家具の距離
        ex) sofa_1とbed_1の距離
    """
    selfname = selfdict["name"]
    for dictionary in dict_list:
        if (dictionary.get("name") == name) and (name != selfname):
            distance = calculate_distance(selfdict, dictionary)
            return distance
    return 0

def get_values_from_dicts(key, list_of_dicts):
    values = []
    for dictionary in list_of_dicts:
        if key in dictionary:
            values.append(dictionary[key])
    return values

def make_random_furniture_prob_set(data_list, furniture_names):
    """家具の辞書オブジェクトを作成する関数

    Parameters
    ---------
    data_list : list
        配置する家具の情報を辞書オブジェクトでいれたリスト
    furniture_names : list
        引数で渡された家具を複製して配置する家具の最大値

    Returns
    ------
    data_list : list
        元のdata_listのnameキーに番号と存在するかどうかを記したexistキーを追加したもの
        {"name":ソファ_1, "width":1.4, "length":0.5, "rotation_range":[0, 90, 180], "restriction":["alongwall", "set"], "set_furniture":ベッド, "exist":1},
        {...}
    """
    name_count = {}  # 各名前の出現回数を数えるための辞書
    #print(f'''DATA LIST : {data_list}''')
    for data in data_list:
        name = data["name"]
        if name not in name_count:
            name_count[name] = 1
        else:
            name_count[name] += 1
        data["name"] = f'''{name}_{name_count[name]}'''
    for data in data_list:
        data["exist"] = 1
    #print(f'''DATA LIST : {data_list}''')
    data_list_names = get_values_from_dicts("name", data_list)
    for furniture_name in furniture_names:
        if furniture_name not in data_list_names:
            d = {"name":furniture_name, "exist":0}
            data_list.append(d)
    #print(f'''NAME COUNT : {name_count}''')

    return data_list

def rereformat_dataframe(df):
    """データフレームを機械学習用の構造に変換する関数
    """
    #print(f'''DATAFRAME : {df}''')
    new_df = pd.DataFrame()
    room_num_unique_list = list(df["room"].unique())
    all_column_list = df.columns.tolist()
    remove_column_list = ["room", "room_h_length", "room_v_length", "target", "name"]
    column_list = [x for x in all_column_list if x not in remove_column_list]
    for room_num in room_num_unique_list:
        df_split = df[df["room"] == room_num]
        df_split = df_split.reset_index(drop=True)
        room_h, room_v = df_split.at[0 ,"room_h_length"], df_split.at[0 ,"room_v_length"]
        dic = {"room_num":room_num, "room_v":room_v, "room_h":room_h, "target":"uninspected"}
        for index in range(len(df_split)):
            df_split_one_line = df_split.iloc[index, :]
            name = df_split_one_line["name"]
            for column in column_list:
                dic[f"""{name}_{column}"""] = df_split_one_line[column]
            new_df_split_one_line = pd.DataFrame(dic, index=[0])
        new_df = pd.concat([new_df, new_df_split_one_line], ignore_index=True)
    return new_df
        
def generate_room(room_width:int, room_length:int, furnitures:list, generate_num:int, windows:list=None, doors:list=None):
    """ランダムな家具配置の情報を生成する関数
    Parameters
    ---------
    room_width : int
        部屋の横幅
    room_length : int
        部屋の縦幅
    furnitures : list
        配置する家具の情報を辞書オブジェクトでいれたリスト
        ex) furnitures = [
            {"name":ソファ, "width":1.4, "length":0.5, "rotation_range":[0, 90, 180], "restriction":["alongwall", "set"], "set_furniture":ベッド},
            {...}
        ]
    generate_num : int
        何パターンの家具配置を出力するか
    windows : list
        部屋の窓の端を示したもの(詳しくはRoomクラスの説明で)
    doors : list
        部屋のドアの端を示したもの(詳しくはRoomクラスの説明で)

    Returns
    -------
    room_info : pd.DataFrame
        各家具配置パターンでの家具の情報が入ったdataframe
    """
    #print(f'''FURNITURES : {furnitures}''')
    furniture_list = [{"name":furniture.name, "width":furniture.width, "length":furniture.length, "rotation_range":furniture.rotation_range, "restriction":furniture.restriction} for furniture in furnitures]
    edges = [
        [0, 0],
        [0, room_length],
        [room_width, room_length],
        [room_width, 0]
    ]
    room_info = pd.DataFrame()
    for I in range(generate_num):
        room = Room(edges, windows=windows, doors=doors)
        room.plot_room()
        furniture_name_non_duplicated = ["sofa", "desk", "chair", "TV&Stand", "light", "plant", "shelf", "drawer", "bed", "TV"]
        furniture_names = [f"{item}_{i}" for item in furniture_name_non_duplicated for i in range(1, 4)]#[sofa_1, sofa_2, ..]
        #家具をランダムで複製
        dummy_furniture_list = copy.deepcopy(furniture_list)
        new_random_furniture = make_random_furniture_prob_set(dummy_furniture_list, furniture_names)#dictにexistキーを追加しなきゃいけない
        #print(f'''ALL FURNITURE : {new_random_furniture}''')
        furniture_info_list = room.random_plot_furniture(random_furniture=new_random_furniture)
        #print(f'''FURNITURE INFO list: {furniture_info_list}''')
       
        #各家具の相対的な距離を算出したカラムを追加        
        for i in furniture_info_list:
            for furniture_name in furniture_names:
                if i['exist'] == 0:
                    i[f'd_{furniture_name}'] = 0
                elif i["name"]!=furniture_name:
                    distance = find_dict_by_name(furniture_info_list, furniture_name, i)
                    i[f"""d_{furniture_name}"""] = distance
                else:
                    i[f'd_{furniture_name}'] = 0
        #print(f'''FURNITURE INFO 1: {furniture_info_list[0]}''')
        #print(f'''FURNITURE INFO 2: {furniture_info_list[1]}''')
        for furniture_info in furniture_info_list:
            df = pd.DataFrame(furniture_info, index=[0])
            df["room"] = f"""room_{str(I)}"""# dataframeに生成されたランダムな部屋配置の番号を追加
            #部屋の縦横に関してのカラムを追加
            df["room_h_length"] = room_width
            df["room_v_length"] = room_length
            room_info = pd.concat([room_info, df])
    room_info["target"] = "uninspected"
    room_info = rereformat_dataframe(room_info)
    #print(room_info.columns)
    return room_info

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

    # リスト内の要素が閾値を超える場合、そのインデックスを取得
    max_index = predictions_list.index(max(predictions_list))

    return max_index

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
    model_path = './AI_model/torch_model.pth'
    df_test = df.drop(['room_num', 'target'], axis=1)
    index = get_high_score_indices(model_path, df_test)
    
    index = 0
    return index


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

