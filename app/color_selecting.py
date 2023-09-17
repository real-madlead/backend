import colorsys
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from dotenv import load_dotenv
import os
import openai
from app.schemas import Floor, FurniturePlace, FloorPlanOutputSchema
import re


# .env ファイルをロード
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])

def get_similar_color(hex_color, hue_shift=0.1):
    rgb = hex_to_rgb(hex_color)
    hsv = colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    new_hue = (hsv[0] + hue_shift) % 1
    new_rgb = colorsys.hsv_to_rgb(new_hue, hsv[1], hsv[2])
    return rgb_to_hex((int(new_rgb[0]*255), int(new_rgb[1]*255), int(new_rgb[2]*255)))

def get_complementary_color(hex_color):
    return get_similar_color(hex_color, 0.5)

def color_distance(color1, color2):
    """Calculate distance between two colors in HSV space."""
    h1, s1, v1 = colorsys.rgb_to_hsv(*color1)
    h2, s2, v2 = colorsys.rgb_to_hsv(*color2)
    dh = min(abs(h1-h2), 1-abs(h1-h2)) * 2
    ds = abs(s1-s2)
    dv = abs(v1-v2) / 255.0
    return dh*dh + ds*ds + dv*dv

def closest_color(ref_color, colors):
    """Find the closest color from a list."""
    ref_color_rgb = hex_to_rgb(ref_color)
    min_colors = None
    min_distance = float('inf')
    
    for color in colors:
        color_rgb = hex_to_rgb(color)
        distance = color_distance(ref_color_rgb, color_rgb)
        if distance < min_distance:
            min_distance = distance
            min_colors = color
            
    return min_colors

def set_optimized_color_each_furniture(furnitureplace_list: list[FurniturePlace], floor: Floor):
    """各家具に最適な色を与える
    """
    all_furniture_area = sum([furnitureplace.length * furnitureplace.width for furnitureplace in furnitureplace_list])
    room_area = floor.length * floor.width - all_furniture_area
    furniture_groups = distribute_furnitures(furnitureplace_list=furnitureplace_list, room_area=room_area)
    base_color_code, assorted_color_code, accent_color_code = get_color_pairs_from_text()
    new_furnitureplace_list = list()
    base_color_furniture_group = furniture_groups[0]
    assorted_color_furniture_group = furniture_groups[1]
    accent_color_furniture_group = furniture_groups[2]
    for base_color_furniture in base_color_furniture_group:
        base_color_furniture.color_map_path = f'/materials/texture/desk/{base_color_code[1:]}.jpg'
        new_furnitureplace_list.append(base_color_furniture)
    for assorted_color_furniture in assorted_color_furniture_group:
        assorted_color_furniture.color_map_path = f'/materials/texture/desk/{assorted_color_code[1:]}.jpg'
        new_furnitureplace_list.append(assorted_color_furniture)
    for accent_color_furniture in accent_color_furniture_group:
        accent_color_furniture.color_map_path = f'/materials/texture/desk/{accent_color_code[1:]}.jpg'
        new_furnitureplace_list.append(accent_color_furniture)    
    return new_furnitureplace_list

def set_optimized_color_each_furniture(floor_plan_output_schema: FloorPlanOutputSchema, input_text: str):
    """
    Parameters
    ---------
    floor_plan_output_schema: FloorPlanOutputSchema
        家具情報、配置情報、部屋の大きさの情報など
    input_text: str
        ユーザーの要望のテキスト

    Returns
    ------
    set_color_floor_plan_output_schema: FloorPlanOutputSchema
        色情報を追加したFloorPlanOutputSchema
    """
    set_color_floor_plan_output_schema = FloorPlanOutputSchema()
    
    return set_color_floor_plan_output_schema


def distribute_furnitures(furnitureplace_list:list[FurniturePlace], room_area:float, ratios:list[float]=[7, 2.5, 0.5]):
    """

    Returns
    ------
    groups : list[list[float], list[float], list[float]]
    """
    # 合計を計算
    total = sum([furnitureplace.length * furnitureplace.width for furnitureplace in furnitureplace_list])
    
    # グループの目標合計値を計算
    target_sum_area = [total * r / sum(ratios) for r in ratios]

    # グループの現在の合計値を保存するリスト
    group_area_sums = [0, 0, 0]
    
    # グループの内容を保存するリスト
    groups = [[], [], []]
    
    # 数字を降順にソート
    sorted_furnitureplace_list = sorted(furnitureplace_list, reverse=True, key=lambda x: x.length*x.width)
    
    for furnitureplace in sorted_furnitureplace_list:
        # 各グループの目標合計値と現在の合計値の差を計算
        diffs = [target_sum_area[i] - group_area_sums[i] for i in range(3)]
        
        # 最も合計が低いグループを選択
        group_idx = diffs.index(max(diffs))
        
        # 数字を選択したグループに追加
        groups[group_idx].append(furnitureplace)
        group_area_sums[group_idx] += furnitureplace.width * furnitureplace.length
    print(groups)
    return groups



def get_color_pairs_from_text(text='水色が中心のお部屋がいい'):
    """テキストから色の組み合わせ3色を取得
    Parameters
    ---------
    text : str
        テキスト
    
    Returns
    ------
    base_color : str
        ベースカラーのカラーコード
    assorted_color : str
        アソートカラーのカラーコード
    accent_color : str
        アクセントカラーのカラーコード
    """
    prompt = f'''
    あなたは部屋のコーディネーターです。
    ##顧客の要望に沿うように部屋のメインカラーを3色下記の##選択肢の色の中から選び##制限に従って出力しなさい

    ##出力フォーマット
    カラーコード１：〇〇
    カラーコード２：〇〇
    カラーコード３：〇〇

    ##制限
    余計な文章は出力しない

    ##顧客の要望
    {text}

    ##選択肢の色
    #CC01CC
    #6600CD
    #3401CC
    #0166FF
    #00CCCB
    #01CC34
    #97CA00 
    #FFFF01
    #FF6600
    #CC0001
    '''
    chat_model = ChatOpenAI(temperature=0 ,model_name="gpt-3.5-turbo")
    chat = chat_model([
        SystemMessage(content="日本語で回答して"),
        HumanMessage(content=prompt),
    ])
    output_text = chat.content
    print(f'''テキストアウトプット{output_text}''')

    """
    color_code_1 = output_text.split(":")[1]
    color_code_similar_color_code_1 = get_similar_color(color_code_1, hue_shift=0.1)
    color_code_complementary_color_code_1 = get_complementary_color(color_code_1)
    """
    color_codes = re.findall(r"#(?:[0-9a-fA-F]{3}){1,2}", output_text)
    print(color_codes)

    base_color = color_codes[0]
    assorted_color = color_codes[1]
    accent_color = color_codes[2]
    
    return base_color, assorted_color, accent_color

    


if __name__ == '__main__':
    # 使用例
    color = "#FF5733"
    print(f"Original Color: {color}")
    print(f"Similar Color: {get_similar_color(color)}")
    print(f"Complementary Color: {get_complementary_color(color)}")