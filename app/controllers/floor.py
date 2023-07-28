from fastapi import APIRouter
from app.schemas import FloorPlanInputSchema, FloorPlanOutputSchema, FurniturePosition, Furniture
from app.furniture_data import furniture_list_all
from app.AI import create_room, squeeze_room, get_position
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
    created_room = create_room(room_width=floor_width, room_length=floor_length, furnitures=furniture_list, create_num=100)#ランダムに家具の配置を作成
    squeezed_room = created_room.iloc[squeeze_room(created_room)]#AIによりベストな家具配置を見つける

    furniture_position_list = []
    name_counter = {}
    for furniture in furniture_list:
        furniture_postion = FurniturePosition(
            name=furniture.name,
            width=furniture.width,
            length=furniture.length,
            if name not in name_counter:
                name_counter[name] = 1
            else:
                name_counter[name] += 1
            x, y = get_position(name, name_counter, df)
            # とりあえず原点に配置
            x=0,
            y=0
        )
        furniture_position_list.append(furniture_postion)
    
    return FloorPlanOutputSchema(floor=floor_info.floor, furniture=furniture_position_list)

# 家具のリストを取得
@router.get("/floor/furnitures")
def get_furnitures() -> list[Furniture]:
    return furniture_list_all