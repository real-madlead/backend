from .floor import Floor
from .furniture import FurniturePlacement
class Room:
    def __init__(self, floor: Floor, Furnitures: list[FurniturePlacement]):
        self.floor = floor
        self.Furnitures = Furnitures
