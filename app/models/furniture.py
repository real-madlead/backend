class Furniture():
    def __init__(self, id: int, name: str, width: float, length: float, restriction: str, rand_rotation: list) -> None:
        self.id = id
        self.name = name
        self.width = width
        self.length = length
        self.restriction = restriction
        self.rand_rotation = rand_rotation
        
class FurniturePlacement():
    def __init__(self, furniture: Furniture, x: float, y: float, rotation: float) -> None:
        self.furniture = furniture
        self.x = x
        self.y = y
        self.rotation = rotation
