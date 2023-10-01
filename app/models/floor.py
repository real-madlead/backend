import math
from shapely.geometry import LineString

class Floor:
    def __init__(self, edges: list, windows: list = None, doors: list = None):
        """
        Parameters
        ----------
        edges : list
            部屋の角の座標(左下を始点に時計周りで記述)[[x座標, y座標], [float, float], ]
        windows : list[windows]
        doors : list[doors]
        """
        if len(edges) != 4:
            raise ValueError("edges must be 4 points")
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("edge must have x, y coordinate")
        
        self.edges = edges
        self.windows = windows
        self.doors = doors
        self.line_objects = []
        self.plot_floor()
    
    def plot_floor(self):
        coords = self.edges
        if self.windows:
            window_start_coords = [window.start for window in self.windows]
            window_end_coords = [window.end for window in self.windows]
            windows = window_start_coords + window_end_coords
            coords.extend(windows)
            
        if self.doors:
            door_start_coords = [door.start for door in self.doors]
            door_end_coords = [door.end for door in self.doors]
            doors = door_start_coords + door_end_coords
            coords.extend(doors)
            
        coords = self.sort_coords_by_angle(coords)
        last_coords = [coords[-1], coords[0]]
        last_line = {
            "x": [last_coords[0][0], last_coords[1][0]],
            "y": [last_coords[0][1], last_coords[1][1]],
        }
        if (self.windows != None) and (last_coords[0] in windows) and (last_coords[1] in windows):
            last_line["color"] = "blue"
        elif (self.doors != None) and (last_coords[0] in doors) and (last_coords[1] in doors):
            last_line["color"] = "red"
        else:
            last_line["color"] = "k"
        lines = [last_line]
        for i in range(len(coords) - 1):
            ps = coords[i : i + 2]
            line = {"x": [ps[0][0], ps[1][0]], "y": [ps[0][1], ps[1][1]]}
            if (self.windows != None) and (ps[0] in windows) and (ps[1] in windows):
                line["color"] = "blue"
            elif (self.doors != None) and (ps[0] in doors) and (ps[1] in doors):
                line["color"] = "red"
            else:
                line["color"] = "k"
            lines.append(line)
        for line in lines:
            pseudogeometric_line_for_calculation = self.create_floor_rectangle(
                start=[line["x"][0], line["y"][0]],
                end=[line["x"][1], line["y"][1]],
                color=line["color"],
            )
            self.line_objects.append(pseudogeometric_line_for_calculation)
        
    @staticmethod
    def sort_coords_by_angle(coords: list[list]):
        """各点を角度に基づいてソートする関数"""
        center = Floor.get_center_coord(coords)
        return sorted(coords, key=lambda coord: Floor.get_angle(coord, center))
    
    @staticmethod
    def get_center_coord(coords):
        """中心を見つけるための関数"""
        x_coords = [p[0] for p in coords]
        y_coords = [p[1] for p in coords]
        center_x = sum(x_coords) / len(coords)
        center_y = sum(y_coords) / len(coords)
        return (center_x, center_y)
    
    @staticmethod
    def get_angle(coord, center):
        """角度を計算する関数"""
        return math.atan2(coord[1] - center[1], coord[0] - center[0])
    
    @staticmethod
    def create_floor_rectangle(start: list, end: list):
        """部屋の枠を表す線を作成する関数
        Returns
        -------
        floor_rectangle : shapely.geometry.LineString
            あたり判定を計算するために仮想的に作成された四角形
        """
        floor_rectangle = LineString([(start[0], start[1]), (end[0], end[1])])
        return floor_rectangle
        
class Window:
    def __init__(self, start: list, end: list):
        """
        Parameters
        ----------
        start : list
            窓の始点[x座標, y座標]
        end : list
            窓の終点[x座標, y座標]
        """
        self.start = start
        self.end = end
        
class Door:
    def __init__(self, start: list, end: list):
        """
        Parameters
        ----------
        start : list
            ドアの始点[x座標, y座標]
        end : list
            ドアの終点[x座標, y座標]
        """
        self.start = start
        self.end = end
        