from pydantic import BaseModel, Field

# 家具
class Furniture(BaseModel):
    id: int = Field(example=0)
    name: str = Field(example="sofa")
    width: float = Field(example=1.4)
    length: float = Field(example=0.5)
    restriction: str | None = Field(example="alongwall")
    rand_rotation: list = Field(example=[0, 90, 180]) 
    

# 家具の位置　家具を継承
class FurniturePlace(Furniture):
    x: float = Field(example=0)
    y: float = Field(example=0)
    rotation: float = Field(example=0)
    

# 床
class Floor(BaseModel):
    width: float = Field(example=5)
    length: float = Field(example=5)

class FurnitureInput(BaseModel):
    id: int = Field(example=0)
    quantity: int = Field(example=1)

# 間取り生成の入力
class FloorPlanInputSchema(BaseModel):
    floor: Floor
    furnitures: list[FurnitureInput]



# 間取り生成の出力
class FloorPlanOutputSchema(BaseModel):
    floor: Floor
    furnitures: list[FurniturePlace]
    scoring_of_room_layout_using_AI: float