# spatial_hash.py
from collections import defaultdict
from typing import Iterable, Tuple, Dict, List
import math

Cell = Tuple[int, int]

class SpatialHash:
    """Uniform-grid spatial hash for broadphase collision & picking."""

    def __init__(self, cell_size: float = 64.0):
        self.cell_size = float(cell_size)
        self.cells: Dict[Cell, List[int]] = defaultdict(list)
        self.aabbs: Dict[int, Tuple[float, float, float, float]] = {}

    def _cell_coords(self, x: float, y: float) -> Cell:
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def _cells_for_aabb(self, aabb: Tuple[float, float, float, float]):
        x0, y0, x1, y1 = aabb
        cx0, cy0 = self._cell_coords(x0, y0)
        cx1, cy1 = self._cell_coords(x1, y1)
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                yield (cx, cy)

    def clear(self):
        self.cells.clear()
        self.aabbs.clear()

    def insert(self, id_: int, aabb: Tuple[float, float, float, float]):
        self.aabbs[id_] = aabb
        for c in self._cells_for_aabb(aabb):
            self.cells[c].append(id_)

    def rebuild(self, items: Iterable[Tuple[int, Tuple[float, float, float, float]]]):
        self.clear()
        for id_, aabb in items:
            self.insert(id_, aabb)

    def query(self, aabb: Tuple[float, float, float, float]) -> List[int]:
        ids = []
        seen = set()
        for c in self._cells_for_aabb(aabb):
            for id_ in self.cells.get(c, ()):
                if id_ not in seen:
                    seen.add(id_)
                    ids.append(id_)
        return ids

    def neighbors_of_point(self, x: float, y: float, radius: float) -> List[int]:
        r = float(radius)
        return self.query((x - r, y - r, x + r, y + r))
