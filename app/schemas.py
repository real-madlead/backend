from pydantic import BaseModel, Field

# 家具
class Furniture(BaseModel):
    name: str = Field(example="Chair")
    width: float = Field(example=0.5)
    length: float = Field(example=0.5)

# 家具の位置　家具を継承
class FurniturePosition(Furniture):
    x: float = Field(example=0)
    y: float = Field(example=0)

# 床
class Floor(BaseModel):
    width: float = Field(example=5)
    length: float = Field(example=5)

# 間取り生成の入力
class FloorPlanInputSchema(BaseModel):
    floor: Floor
    furnitures: list[Furniture]

# 間取り生成の出力
class FloorPlanOutputSchema(BaseModel):
    floor: Floor
    furnitures: list[FurniturePosition]