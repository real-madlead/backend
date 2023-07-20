from app.schemas import Furniture

furniture_list_all: list[Furniture] = [
    Furniture(id=0, name="sofa", width=1.4, length=0.5),
    Furniture(id=1, name="desk", width=1.2, length=0.6),
    Furniture(id=2, name="chair", width=0.5, length=0.5),
    Furniture(id=3, name="TV", width=1.2, length=0.05),
    Furniture(id=4, name="TV&Stand", width=1.8, length=0.4),
    Furniture(id=5, name="light", width=0.2, length=0.2),
    Furniture(id=6, name="plant", width=0.2, length=0.2),
    Furniture(id=7, name="shelf", width=0.4, length=0.3),
    Furniture(id=8, name="drawer", width=1, length=0.5),
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