from app.schemas import Furniture

furniture_list_all: list[Furniture] = [
    Furniture(id=0, name="bed", width=1.95, length=1.0, rand_rotation=[0, 90, 180, 270], restriction=["alongwall"]),
    Furniture(id=1, name="desk", width=1.2, length=0.6, rand_rotation=[0, 90, 180, 270], restriction=["alongwall"]),
    Furniture(id=2, name="chair", width=0.5, length=0.5, rand_rotation=[0, 90, 180, 270], restriction=["set"]),
    #Furniture(id=3, name="TV", width=1.2, length=0.05, rand_rotation=[0, 45, 90, 135, 180, 225, 270, 315], restriction="alongwall"),
    Furniture(id=3, name="TV&Stand", width=0.4, length=1.8, rand_rotation=[0, 45, 90, 135, 180, 225, 270, 315], restriction=["alongwall direction center"]),
    Furniture(id=4, name="sofa", width=0.5, length=1.4, rand_rotation=[0, 45, 90, 135, 180, 225, 270, 315], restriction=["facing"]),
    Furniture(id=5, name="light", width=0.2, length=0.2, rand_rotation=[0, 90, 180, 270], restriction=["alongwall"]),
    Furniture(id=6, name="plant", width=0.2, length=0.2, rand_rotation=[0, 90, 180, 270], restriction=["alongwall"]),
    Furniture(id=7, name="shelf", width=0.4, length=0.3, rand_rotation=[0, 90, 180, 270], restriction=["alongwall direction center"]),
    Furniture(id=8, name="chest", width=0.5, length=1, rand_rotation=[0, 90, 180, 270], restriction=["alongwall direction center"]),
]


# 家具	縦	横
# ソファ	0.5	1.4
# デスク	0.6	1.2
# 椅子	0.5	0.5
# テレビ	0.05	1.2
# テレビスタンド&TV	0.4	1.8
# 照明	0.2	0.2
# ラグ、カーペット		
# 観葉植物	0.2	0.2
# 棚	0.3	0.4
# タンス	0.5	1