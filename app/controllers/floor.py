from fastapi import APIRouter
from app.schemas import FloorPlanInputSchema, FloorPlanOutputSchema, FurniturePosition, Furniture
from app.furniture_data import furniture_list_all
from app.AI import generate_room, squeeze_room, get_position
router = APIRouter()

# 家具のリストを受け取り、床の上に配置した家具のリストを返す
@router.post("/floor/generate")
def generate_floor_plan(
    floor_info: FloorPlanInputSchema,
) -> FloorPlanOutputSchema:
    # 配置する家具のリスト{name, width, length}
    furniture_list = floor_info.furnitures
    # 部屋の縦横の長さ
    floor_width = floor_info.floor.width
    floor_length = floor_info.floor.length
    #ランダムに家具の配置を作成
    generated_room = generate_room(room_width=floor_width, room_length=floor_length, furnitures=furniture_list, generate_num=100)
    #AIによりベストな家具配置を見つける
    squeezed_room = generated_room.iloc[squeeze_room(generated_room)]
    furniture_position_list = []
    #各家具の出現数を数えるための辞書
    name_counter = {}
    for furniture in furniture_list:
        if furniture.name not in name_counter:
            name_counter[furniture.name] = 1
        else:
            name_counter[furniture.name] += 1
        #ベストな家具配置パターンの家具の位置を取得
        x, y = get_position(furniture.name, name_counter, squeezed_room)
        furniture_postion = FurniturePosition(
            name=furniture.name,
            width=furniture.width,
            length=furniture.length,
            x=x,
            y=y
        )
        furniture_position_list.append(furniture_postion)
    
    return FloorPlanOutputSchema(floor=floor_info.floor, furniture=furniture_position_list)

# 家具のリストを取得
@router.get("/floor/furnitures")
def get_furnitures() -> list[Furniture]:
    return furniture_list_all

"""
・furniture_listにrotation_range,restrictionというキーを作成して欲しい
・座標の話(原点が家の中心だとか、家具の座標は端ではなく家具の中心だとか)
"""