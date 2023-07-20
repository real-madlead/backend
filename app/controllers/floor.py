from fastapi import APIRouter
from app.schemas import FloorPlanInputSchema, FloorPlanOutputSchema, FurniturePlace, Furniture
from app.furniture_data import furniture_list_all
router = APIRouter()

# 家具のリストを受け取り、床の上に配置した家具のリストを返す
@router.post("/floor/generate")
def generate_floor_plan(
    floor_info: FloorPlanInputSchema,
) -> FloorPlanOutputSchema:
    """
    ### 間取り生成用のAPI
    #### リクエスト
    - ***floor***: 床面積の情報
    - ***furnitures***: 家具のリスト
    家具の数をquantityで指定することで、同じ家具を複数個配置することができる
    ある家具の数が1個以上のときに含める

    #### レスポンス
    - ***floor***: 床面積の情報
    - ***furnitures***: 家具のリスト (家具の位置情報を含む)
    """
    # 配置する家具のリスト{name, width, length}
    furniture_list = floor_info.furnitures
    # 部屋の縦横の長さ
    floor_width = floor_info.floor.width
    floor_length = floor_info.floor.length

    furniture_position_list = []
    for furniture in furniture_list:
        furniture_postion = FurniturePlace(
            name=furniture.name,
            width=furniture.width,
            length=furniture.length,
            # とりあえず原点に配置
            x=0,
            y=0
        )
        furniture_position_list.append(furniture_postion)
    
    return FloorPlanOutputSchema(floor=floor_info.floor, furniture=furniture_position_list)

# 家具のリストを取得
@router.get("/floor/furnitures")
def get_furnitures() -> list[Furniture]:
    """
    ### 使用できる家具のリストを取得するAPI
    #### レスポンス
    [id, name, width, length]をカラムに持つオブジェクトが複数個入った配列が返ってくる
    """
    return furniture_list_all