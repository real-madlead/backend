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
from app.schemas import Furniture, FloorPlanInputSchema, FloorPlanOutputSchema, FurnitureInput, FurniturePlace, Floor
from app.furniture_data import furniture_list_all
import copy



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
            pseudogeometric_line_for_calculation = create_line(start=[line["x"][0], line["y"][0]], end=[line["x"][1], line["y"][1]], color=line["color"])
            self.line_objects.append(pseudogeometric_line_for_calculation)
        
    def plot_furniture(self, furniture_places_list:list):
        """家具を配置するメソッド

        Parameters
        ----------
        furnitures : list[FurniturePlace]
            FurniturePlaceオブジェクトが入ったリスト

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
        for furniture_place in furniture_places_list: 
            calculate_furniture = create_rectangle([furniture_place.x, furniture_place.y], furniture_place.width, furniture_place.length, furniture_place.rotation, "blue")
        
            if (multi_check_overlap(calculate_furniture, self.line_objects)) or (furniture_place.x<=min_x) or (furniture_place.x>=max_x) or (furniture_place.y<=min_y) or (furniture_place.y>=max_y):
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
    
    def place_furnitures_with_restriction(self, furniture_objects_list:list):
        """家具を部屋、他の家具とかさならないように配置するメソッド

        Parameters
        ---------
        furniture_objects_list : list[Furniture]
            Furnitureオブジェクトが入ったリスト
        
        Returns
        -------
        furnitureplace_objects_list : list[FurniturePlace]
            FurniturePlaceオブジェクトが入ったリスト
        """
        # 新しいソートされたリストを作成
        sorted_furniture_objects_list = []
        for item in furniture_list_all:
            count = furniture_objects_list.count(item)
            sorted_furniture_objects_list.extend([item] * count)

        x_coords = [edge[0] for edge in self.edges]
        y_coords = [edge[1] for edge in self.edges]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        max_attempts = 10000
        for _ in range(max_attempts):
            restart = False  # ループを再開するかどうかをチェックするフラグ
            furnitureplace_objects_list = list()
            for furniture_object in sorted_furniture_objects_list:
                counter = 0
                while True:
                    if furniture_object.restriction=="alongwall":
                        x, y, rotation = set_alongwall(min_x, max_x, min_y, max_y, furniture_object.rand_rotation, delta=0.01)
                    elif furniture_object.restriction=="alongwall direction center":
                        x, y, rotation = set_alongwall_dir_ctr(furniture_object.length, min_x, max_x, min_y, max_y, delta=0.01)
                    elif furniture_object.restriction.split("_")[0]=="set":
                        furnitureplace_objects_targeted_for_set = [i for i in furnitureplace_objects_list if i.name==furniture_object.restriction.split("_")[1]]
                        if len(furnitureplace_objects_targeted_for_set)!=0:
                            x, y, rotation = set_combo(furniture_object.length, furniture_object.width, min_x, max_x, min_y, max_y, set_furniture=random.choice(furnitureplace_objects_targeted_for_set), delta=0.01)
                        else:
                            x, y, rotation = random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.choice(furniture_object.rand_rotation)
                    elif furniture_object.restriction.split("_")[0]=="facing":
                        furnitureplace_objects_targeted_for_facing = [i for i in furnitureplace_objects_list if i.name==furniture_object.restriction.split("_")[1]]
                        if len(furnitureplace_objects_targeted_for_facing)!=0:
                            x, y, rotation = set_facing(furniture_object.length, furniture_object.width, min_x, max_x, min_y, max_y, face_furniture=random.choice(furnitureplace_objects_targeted_for_facing), delta=0.01)
                        else:
                            x, y, rotation = random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.choice(furniture_object.rand_rotation)
                    else:
                        x, y, rotation = random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.choice(furniture_object.rand_rotation)
                    furnitureplace_object = FurniturePlace(
                        id = furniture_object.id,
                        name = furniture_object.name,
                        width = furniture_object.width,
                        length = furniture_object.length,
                        restriction = furniture_object.restriction,
                        rand_rotation = furniture_object.rand_rotation,
                        x = x,
                        y = y,
                        rotation = rotation,
                        materials = []
                    )
                    error_flag = self.plot_furniture([furnitureplace_object])
                    if error_flag[0]!=0:
                        self.clear_furniture(furniture_index=-1)
                        counter += 1
                    elif error_flag[0]==0:
                        furnitureplace_objects_list.append(furnitureplace_object)
                        break
                    if counter>50:#50回以上エラーが出たら論理的におけないと判断し、もう一度全ての家具を置きなおす
                        self.clear_furniture(all_clear=True)
                        restart = True
                        break
                if restart:
                    break
            if not restart:  # もし再開フラグがFalseの場合、外部ループを終了
                break
        return furnitureplace_objects_list

def create_rectangle(center, width, height, angle, color):
    """四角形を作成する関数
    Returns
    -------
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

def set_alongwall_dir_ctr(v_width, min_x, max_x, min_y, max_y, delta=0.01):
    """家具が配置される際に壁沿い かつ 中心に向かう用に配置できる

    Parameters
    ---------
    v_width : float
        配置する家具の縦幅
    min_x, max_x, min_y, max_y : int
        部屋のx,y座標の最大値、最小値
    delta : float
        家具が壁と完全に接触してしまうとエラーが出るので、この数値だけ壁からずらす

    Returns
    ------
    x, y, rotation : float, float, int
        配置する家具のx, y座標と角度(度数法)
    """
    rand = random.choice([0, 1, 2, 3])
    if rand == 0:
        x, y, rotation = random.uniform(min_x + v_width, max_x), min_y + delta, 90
    elif rand == 1:
        x, y, rotation = max_x - delta, random.uniform(min_y+v_width, max_y), 180
    elif rand == 2:
        x, y, rotation = random.uniform(min_x, max_x-v_width), max_x - delta, 270 
    elif rand == 3:
        x, y, rotation = min_x + delta, random.uniform(min_y ,max_y-v_width), 0
    return x, y, rotation

def set_alongwall(min_x, max_x, min_y, max_y, rand_rotation, delta=0.01):
    """家具が配置される際に壁沿い配置できる
    
    Parameters
    ---------
    min_x, max_x, min_y, max_y : int
        部屋のx,y座標の最大値、最小値
    rand_rotation : list[int]
        配置される家具がランダムにとる角度（度数法）が入ったリスト
    delta : float
        家具が壁と完全に接触してしまうとエラーが出るので、この数値だけ壁からずらす

    Returns
    ------
    x, y, rotation : float, float, int
        配置する家具のx, y座標と角度(度数法)
    """
    rand = random.choice([0, 1])
    rotation = random.choice(rand_rotation)
    if rand == 0:
        x, y = random.choice([min_x + delta, max_x - delta]), random.uniform(min_y, max_y)
    elif rand == 1:
        x, y = random.uniform(min_x, max_x), random.choice([min_y + delta, max_y - delta])
    return x, y, rotation

def set_combo(v_width, h_width, min_x, max_x, min_y, max_y, set_furniture=None, delta=0.01):
    """家具の配置の際に特定の家具とセットで配置できる
    Parameters
    ---------
    v_width, h_width : float
        配置する家具の縦幅、横幅
    min_x, max_x, min_y, max_y : int
        部屋のx,y座標の最大値、最小値
    set_furniture : Furniture
        セットでおく家具のFurnitureオブジェクト
    delta : float
        家具が壁と完全に接触してしまうとエラーが出るので、この数値だけ壁からずらす

    Returns
    ------
    x, y, rotation : float, float, int
        配置する家具のx, y座標と角度(度数法)
    """
    set_f_x, set_f_y, set_f_rotation = set_furniture.x + delta*math.sin(math.radians(set_furniture.rotation)), set_furniture.y - delta*math.cos(math.radians(set_furniture.rotation)), set_furniture.rotation
    rand = random.choice([0, 1, 2, 3]) 
    if rand == 0:
        set_f_rand_len = random.uniform(0, set_furniture.width - v_width)
        x = set_f_x + set_f_rand_len*math.cos(math.radians(set_f_rotation)) + v_width*math.cos(math.radians(set_f_rotation)) + h_width*math.sin(math.radians(set_f_rotation))
        y = set_f_y + set_f_rand_len*math.sin(math.radians(set_f_rotation)) + v_width*math.sin(math.radians(set_f_rotation)) - h_width*math.cos(math.radians(set_f_rotation))
        rotation = set_f_rotation + 90
    elif rand == 1:
        set_f_rand_len = random.uniform(0, set_furniture.length - v_width)
        x = set_f_x + set_furniture.width*math.cos(math.radians(set_f_rotation)) - set_f_rand_len*math.sin(math.radians(set_f_rotation)) + h_width*math.cos(math.radians(set_f_rotation)) - v_width*math.sin(math.radians(set_f_rotation))
        y = set_f_y + set_furniture.width*math.sin(math.radians(set_f_rotation)) + set_f_rand_len*math.cos(math.radians(set_f_rotation)) + h_width*math.sin(math.radians(set_f_rotation)) + v_width*math.cos(math.radians(set_f_rotation))
        rotation = set_f_rotation + 180
    elif rand == 2:
        set_f_rand_len = random.uniform(0, set_furniture.width - v_width)
        x = set_f_x + set_f_rand_len*math.cos(math.radians(set_f_rotation)) - h_width*math.sin(math.radians(set_f_rotation)) - set_furniture.length*math.sin(math.radians(set_f_rotation))
        y = set_f_y + set_f_rand_len*math.sin(math.radians(set_f_rotation)) + h_width*math.cos(math.radians(set_f_rotation)) + set_furniture.length*math.cos(math.radians(set_f_rotation))
        rotation = set_f_rotation - 90
    elif rand == 3:
        set_f_rand_len = random.uniform(0, set_furniture.length - v_width)
        x = -1*set_f_rand_len*math.sin(math.radians(set_f_rotation)) - h_width*math.cos(math.radians(set_f_rotation))
        y = set_f_rand_len*math.cos(math.radians(set_f_rotation)) - h_width*math.sin(math.radians(set_f_rotation))
        rotation = set_f_rotation
    return x, y, rotation

def set_facing(v_width, h_width, min_x, max_x, min_y, max_y, face_furniture, delta=0.01):
    """家具配置の際に特定の家具と向かい合いになるように配置できる
    Parameters
    ---------
    v_width, h_width : float
        配置する家具の縦幅、横幅
    min_x, max_x, min_y, max_y : int
        部屋のx,y座標の最大値、最小値
    face_furniture : Furniture
        向かい合って配置する家具のFurnitureオブジェクト
    delta : float
        家具が壁と完全に接触してしまうとエラーが出るので、この数値だけ壁からずらす

    Returns
    ------
    x, y, rotation : float, float, int
        配置する家具のx, y座標と角度(度数法)
    """
    face_rotation = face_furniture.rotation
    face_x, face_y, face_h, face_v = face_furniture.x, face_furniture.y, face_furniture.width, face_furniture.length
    if face_rotation == 0:
        x, y, rotation = random.uniform(face_x+h_width+face_h, max_x), face_y + face_v/2 + v_width/2, 180
    elif face_rotation == 90:
        x, y, rotation = face_x - face_v/2 - v_width/2, random.uniform(face_y+h_width+face_h, max_y), 270
    elif face_rotation == 180:
        x, y, rotation = random.uniform(min_x, face_x-face_h-h_width), face_y - face_v/2 - v_width/2, 0
    elif face_rotation == 270:
        x, y, rotation = face_x + face_v/2 + v_width/2, random.uniform(min_y ,face_y-h_width-face_h), 90
    return x, y, rotation

        

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
   
def calculate_distance(p1, p2):
    """2点の座標の距離を計算する関数
    Parameters
    ---------
    p1, p2 : dict
        x,yのキーが入っている辞書
    """
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

def get_furniture_distance(dict_list, name, selfdict):
    """家具同士の距離を取得できる

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


def generate_room(floor_object:Floor, furniture_list:list[Furniture], generate_num:int, windows:list=None, doors:list=None):
    """ランダムな家具配置の情報を生成する関数
    Parameters
    ---------
    floor_object : Floor
        schemas.pyにあるFloorオブジェクト
    furniture_list : list[Furniture]
        schemas.pyにあるFurnitureオブジェクトが複数入ったリスト
    generate_num : int
        何パターンの家具配置を出力するか
    windows : list
        部屋の窓の端を示したもの(詳しくはRoomクラスの説明で)
    doors : list
        部屋のドアの端を示したもの(詳しくはRoomクラスの説明で)

    Returns
    -------
    rooms_furniture_placement_df : pd.DataFrame
        各家具配置パターンでの家具の情報が入ったdataframe
    """
    #print(f'''FURNITURES : {furnitures}''')
    edges = [
        [0, 0],
        [0, floor_object.length],
        [floor_object.width, floor_object.length],
        [floor_object.width, 0]
    ]
    rooms_furniture_placement_list = list()
    for I in range(generate_num):
        room = Room(edges, windows=windows, doors=doors)
        room.plot_room()
        furnitureplace_list = room.place_furnitures_with_restriction(furniture_objects_list=furniture_list)
        rooms_furniture_placement_list.append(furnitureplace_list)
    rooms_furniture_placement_df = convert_furniture_list_to_dataframe(rooms_furniture_placement_list, floor_object=floor_object)
    return rooms_furniture_placement_df        

def convert_furniture_list_to_dataframe(rooms_furniture_placement_list:list[list[FurniturePlace]], floor_object:Floor):
    """furnitureplaceオブジェクトが格納されているリストからAIの機会学習用dataframeへの変換

    Parameters
    ---------
    rooms_furniture_placement_list : list[list[FurniturePlace]]
        家具と家具の配置の情報であるFurniturePlaceオブジェクトが複数入ったリスト ->（一つの部屋の家具情報）が複数格納されているリスト -> (複数の部屋の情報が格納されているリスト)
    floor_object : Floor
        schemas.pyにあるFloorオブジェクト

    Returns
    ------
    rooms_furniture_placement_df : pd.DataFrame
        AIのデータ用にフォーマットしたDataFrame
    """
    rooms_furniture_placement_df = pd.DataFrame()
    for index, furniture_placement_list in enumerate(rooms_furniture_placement_list):
        each_furniture_count_dict = dict()
        for item in furniture_list_all:
            each_furniture_count_dict[item.name] = furniture_placement_list.count(item)
        furniture_placement_dict = {'room_num':f'room_{index}', 'room_v':floor_object.length, 'room_h':floor_object.width, 'target':'uninspected'}
        dummy_dict_list = list()
        for furniture in furniture_list_all:
            specific_furniture_placements_list = [furniture_placement for furniture_placement in furniture_placement_list if furniture_placement.name==furniture.name]
            for i in range(len(specific_furniture_placements_list)):
                dummy_dict = dict()
                dummy_dict['name']=f'{furniture.name}_{i+1}'
                dummy_dict['x']=specific_furniture_placements_list[i].x
                dummy_dict['y']=specific_furniture_placements_list[i].y
                dummy_dict['rotation']=specific_furniture_placements_list[i].rotation
                dummy_dict['width']=specific_furniture_placements_list[i].width
                dummy_dict['length']=specific_furniture_placements_list[i].length
                dummy_dict_list.append(dummy_dict)
        only_name_list = [dic['name'] for dic in dummy_dict_list]
        for furniture in furniture_list_all:
            for i in range(3):
                if f'{furniture.name}_{i+1}' in only_name_list:
                    furniture_placement_dict[f'{furniture.name}_{i+1}_exist']=1
                    furniture_placement_dict[f'{furniture.name}_{i+1}_v_width']=next((d['length'] for d in dummy_dict_list if d['name'] == f'{furniture.name}_{i+1}'), None)
                    furniture_placement_dict[f'{furniture.name}_{i+1}_h_width']=next((d['width'] for d in dummy_dict_list if d['name'] == f'{furniture.name}_{i+1}'), None)
                    furniture_placement_dict[f'{furniture.name}_{i+1}_x']=next((d['x'] for d in dummy_dict_list if d['name'] == f'{furniture.name}_{i+1}'), None)
                    furniture_placement_dict[f'{furniture.name}_{i+1}_y']=next((d['y'] for d in dummy_dict_list if d['name'] == f'{furniture.name}_{i+1}'), None)
                    furniture_placement_dict[f'{furniture.name}_{i+1}_rotation']=next((d['rotation'] for d in dummy_dict_list if d['name'] == f'{furniture.name}_{i+1}'), None)
                    for fur in furniture_list_all:
                        for i_2 in range(3):
                            furniture_placement_dict[f'{furniture.name}_{i+1}_d_{fur.name}_{i_2+1}']=get_furniture_distance(dict_list=dummy_dict_list, 
                                                                                                                       name=f'{fur.name}_{i_2+1}', 
                                                                                                                       selfdict=next((d for d in dummy_dict_list if d['name'] == f'{furniture.name}_{i+1}'), None))
                elif f'{furniture.name}_{i+1}' not in only_name_list:
                    furniture_placement_dict[f'{furniture.name}_{i+1}_exist']=0
                    furniture_placement_dict[f'{furniture.name}_{i+1}_v_width']=0
                    furniture_placement_dict[f'{furniture.name}_{i+1}_h_width']=0
                    furniture_placement_dict[f'{furniture.name}_{i+1}_x']=0
                    furniture_placement_dict[f'{furniture.name}_{i+1}_y']=0
                    furniture_placement_dict[f'{furniture.name}_{i+1}_rotation']=0
                    for fur in furniture_list_all:
                        for i_2 in range(3):
                            furniture_placement_dict[f'{furniture.name}_{i+1}_d_{fur.name}_{i_2+1}']=0   
        furniture_placement_df = pd.DataFrame(furniture_placement_dict, index=[0])
        rooms_furniture_placement_df = pd.concat([rooms_furniture_placement_df, furniture_placement_df], ignore_index=True)
                      
    return rooms_furniture_placement_df


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
        candidate_furnitures_for_additional_placement:list[Furniture],
        current_floor_plan_output_schema:FloorPlanOutputSchema,
                                 ):
    """AIにより渡された部屋に新たな家具を配置するようにおすすめする
    Parameters
    ---------
    candidate_furnitures_for_additional_placement : list[Furniture]
        追加で配置する家具の候補
    current_floor_plan_output_schena : FloorPlanOutputSchema
        現在の家具配置
    current_score : flaot
        現在の最高スコア
    
    Returns
    ------
    recommend_furnitureplace : FurniturePlace
        AIが最適だと考えた家具の配置
    best_score : float
        おすすめの家具を配置した場合の部屋のスコア 
    """
    edges = [
        [0, 0],
        [0, current_floor_plan_output_schema.floor.length],
        [current_floor_plan_output_schema.floor.width, current_floor_plan_output_schema.floor.length],
        [current_floor_plan_output_schema.floor.width, 0]
    ]
    room = Room(edges=edges)
    room.plot_room()
    furnitures_list = current_floor_plan_output_schema.furnitures
    _ = room.plot_furniture(furniture_places_list=furnitures_list)
    while True:
        copy_current_room = copy.deepcopy(room)
        candidate_furnitureplace_list = list()# この中に候補のFurniturePlaceオブジェクトが格納される
        candidate_room_furnitureplace_list = list()
        for candidate_furniture in candidate_furnitures_for_additional_placement:
            additinal_furnitre_placement_list = copy_current_room.place_furnitures_with_restriction(furniture_objects_list=[candidate_furniture])
            candidate_furnitureplace_list.append(additinal_furnitre_placement_list)

            current_floor_plan_output_schema.furnitures += additinal_furnitre_placement_list
            candidate_room_furnitureplace_list.append(current_floor_plan_output_schema.furnitures)
            
        test_df = convert_furniture_list_to_dataframe(rooms_furniture_placement_list=candidate_room_furnitureplace_list, floor_object=current_floor_plan_output_schema.floor)
        
        #　AIによる採点
        test_df = test_df.drop(['room_num', 'target'], axis=1)
        best_index, best_score = get_high_score_indices(model_path='./AI_model/torch_model.pth', test_df=test_df)
        if current_floor_plan_output_schema.scoring_of_room_layout_using_AI <= best_score:
            recommend_furnitureplace = candidate_furnitureplace_list[best_index]
            return recommend_furnitureplace[0], best_score
        