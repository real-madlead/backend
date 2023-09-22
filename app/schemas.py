from pydantic import BaseModel, Field

# 家具
class Furniture(BaseModel):
    id: int = Field(example=0)
    name: str = Field(example="bed")
    width: float = Field(example=1.95)
    length: float = Field(example=1.0)
    restriction: str | None = Field(example="alongwall")
    rand_rotation: list = Field(example=[0, 90, 180, 270]) 
    

# 家具の位置　家具を継承
class FurniturePlace(Furniture):
    x: float = Field(example=2)
    y: float = Field(example=2)
    rotation: float = Field(example=0)
    color_code: str = Field(example="#FF6000")

# 床
class Floor(BaseModel):
    width: float = Field(example=7)
    length: float = Field(example=7)

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
    score_of_room_layout_using_AI: float = Field(example=0.5)

class FloorPlanOutputSchemaPlusText(FloorPlanOutputSchema):
    colorcodetext: str = Field(example="#ad5c5b")

"""
class FloorPlanOutputSchemaPlusText(BaseModel):
    floor_plan_output_schema: FloorPlanOutputSchema
    colorcodetext: str = Field(example="ad5c5b")
"""