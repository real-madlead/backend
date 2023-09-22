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
from app.furniture_color_data import furniture_color_data, furniture_materials_data


# .env ファイルをロード
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])


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
    min_color = None
    min_distance = float('inf')
    
    for color in colors:
        color_rgb = hex_to_rgb(color)
        distance = color_distance(ref_color_rgb, color_rgb)
        if distance < min_distance:
            min_distance = distance
            min_color = color
            
    return min_color

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
    recommended_colorcode = get_color_from_text_slab(text=input_text)
    print(recommended_colorcode)
    new_furnitureplace_object_list = list()
    for furnitureplace_object in floor_plan_output_schema.furnitures:
        list_of_colorcodes_furnitureplace_object_possesses = furniture_color_data[furnitureplace_object.name]
        print(list_of_colorcodes_furnitureplace_object_possesses)
        optimized_furniture_colorcode = closest_color(recommended_colorcode, list_of_colorcodes_furnitureplace_object_possesses)
        print(optimized_furniture_colorcode)
        optimized_furniture_colorcode = optimized_furniture_colorcode[1:]
        new_furnitureplace_object = FurniturePlace(
            id=furnitureplace_object.id,
            name=furnitureplace_object.name,
            width=furnitureplace_object.width,
            length=furnitureplace_object.length,
            restriction=furnitureplace_object.restriction,
            rand_rotation=furnitureplace_object.rand_rotation,
            x=furnitureplace_object.x,
            y=furnitureplace_object.y,
            rotation=furnitureplace_object.rotation,
            color_code = optimized_furniture_colorcode
        )
        new_furnitureplace_object_list.append(new_furnitureplace_object)
    print(new_furnitureplace_object_list)

    set_color_floor_plan_output_schema = FloorPlanOutputSchema(
        floor=floor_plan_output_schema.floor,
        furnitures=new_furnitureplace_object_list,
        score_of_room_layout_using_AI=floor_plan_output_schema.score_of_room_layout_using_AI
    )
    return set_color_floor_plan_output_schema

    
def get_color_from_text(text):
    """テキストから色の組み合わせ3色を取得
    Parameters
    ---------
    text : str
        テキスト
    
    Returns
    ------
    recommneded_colorcode : str
        AIがおすすめしてくれた色
    """
    prompt = f'''
    あなたは部屋のコーディネーターです。
    ##顧客の要望に沿うように部屋のメインカラーを3色下記の##選択肢の色の中から選び##制限に従って出力しなさい

    ##出力フォーマット
    カラーコード１：〇〇

    ##制限
    余計な文章は出力しない

    ##顧客の要望
    {text}

    ##選択肢の色
    #ae5cae
    #865cae
    #705daf
    #5d85c3
    #5dadae
    #5dad71
    #9aad5c
    #c1c25c
    #c1845b
    #ad5c5b

    '''
    chat_model = ChatOpenAI(temperature=0 ,model_name="gpt-3.5-turbo")
    chat = chat_model([
        SystemMessage(content="日本語で回答して"),
        HumanMessage(content=prompt),
    ])

    output_text = chat.content
    print(output_text)
    colorcodes = re.findall(r"#(?:[0-9a-fA-F]{3}){1,2}", output_text)

    recommended_colorcode = colorcodes[0]

    print(recommended_colorcode)
    
    return recommended_colorcode

def get_color_from_text_slab(text):
    """ダミーの関数
    """
    recommended_colorcode = '#FF6600'
    
    return recommended_colorcode

if __name__ == '__main__':
    # 使用例
    color = "#FF5733"
    print(f"Original Color: {color}")
 