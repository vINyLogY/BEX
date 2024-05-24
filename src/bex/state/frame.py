# coding: utf-8
r"""Data structure for topology of tensors in a network

"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Optional
from weakref import WeakValueDictionary

from bex.libs.utils import iter_round_visitor, iter_visitor


class Point:
    __cache = WeakValueDictionary()  # type: WeakValueDictionary[str, Point]

    def __new__(cls, name: Optional[str] = None):
        if name is not None and name in cls.__cache:
            return cls.__cache[name]

        obj = object.__new__(cls)

        if name is not None:
            cls.__cache[name] = obj

        return obj

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = str(hex(id(self))) if name is None else str(name)
        return


class Node(Point):

    def __repr__(self) -> str:
        return f'({self.name})'


class End(Point):

    def __repr__(self) -> str:
        return f'<{self.name}>'


class Frame:

    def __init__(self):
        self._neighbor = dict()  # type: dict[Point, list[Point]]
        self._duality = dict(
        )  # type: dict[tuple[Point, int], tuple[Point, int]]
        return

    def merge(self, other: Frame):
        connectors = self.ends & other.ends

        if len(connectors) != 1:
            raise RuntimeError('Can only merge two frame with one connector')
        connector = connectors.pop()

        node_m, i_m = self.dual(connector, 0)
        node_b, i_b = other.dual(connector, 0)
        new_frame = Frame()
        new_frame._neighbor.update(self._neighbor)
        new_frame._neighbor.update(other._neighbor)
        new_frame._duality.update(self._duality)
        new_frame._duality.update(other._duality)

        new_frame._neighbor[node_m][i_m] = node_b
        new_frame._neighbor[node_b][i_b] = node_m
        new_frame._duality[node_m, i_m] = node_b, i_b
        new_frame._duality[node_b, i_b] = node_m, i_m
        del new_frame._duality[connector, 0]
        del new_frame._neighbor[connector]
        return new_frame

    def contract(self, node: Node, axis: int) -> Frame:
        other, j = self._duality[node, axis]
        n1 = list(self._neighbor[node])
        n1.pop(axis)
        n2 = list(self._neighbor[other])
        n2.pop(j)
        new_neighbor = n1 + n2
        new_dict = dict(self._neighbor)
        new_dict[node] = new_neighbor
        del new_dict[other]
        for m in new_dict[node]:
            new_dict[m] = [
                _m if _m is not other else node for _m in self._neighbor[m]
            ]

        new_frame = Frame()
        visited = set()

        for n, ms in new_dict.items():
            visited.add(n)
            for m in ms:
                if m not in visited:
                    new_frame.add_link(n, m)
        return new_frame

    def add_link(self, p: Point, q: Point) -> None:
        for _p in [p, q]:
            if _p not in self._neighbor:
                self._neighbor[_p] = []
            # elif isinstance(_p, End):
            #     raise RuntimeError(
            #         f'End {_p} cannot link to more than one points.')

        i = len(self._neighbor[p])
        j = len(self._neighbor[q])
        self._duality[(p, i)] = (q, j)
        self._duality[(q, j)] = (p, i)

        self._neighbor[p].append(q)
        self._neighbor[q].append(p)
        return

    @property
    def world(self) -> set[Point]:
        return set(self._neighbor.keys())

    @property
    def links(self) -> set[tuple[Point, int, Point, int]]:
        return {(p, i, q, j) for (p, i), (q, j) in self._duality.items()}

    @property
    def nodes(self) -> set[Node]:
        return {p for p in self._neighbor.keys() if isinstance(p, Node)}

    @property
    def ends(self) -> set[End]:
        return {p for p in self._neighbor.keys() if isinstance(p, End)}

    def order(self, p: Node):
        return len(self._neighbor[p])

    def near_points(self, key: Point) -> list[Point]:
        return list(self._neighbor[key])

    def near_nodes(self, key: Point) -> list[Node]:
        return [n for n in self._neighbor[key] if isinstance(n, Node)]

    def dual(self, p: Point, i: int) -> tuple[Point, int]:
        return self._duality[(p, i)]

    def find_axes(self, p: Point, q: Point) -> tuple[int, int]:
        i = self._neighbor[p].index(q)
        j = self._neighbor[q].index(p)
        return i, j

    def node_link_visitor(self,
                          start: Node) -> list[tuple[Node, int, Node, int]]:
        nodes = [n for n in iter_round_visitor(start, self.near_nodes)]
        axes_list = [self.find_axes(n1, n2) for n1, n2 in pairwise(nodes)]
        return [
            (p, i, q, j) for (p, q), (i, j) in zip(pairwise(nodes), axes_list)
        ]

    def point_link_visitor(self,
                           start: Node) -> list[tuple[Point, int, Point, int]]:
        pts = [n for n in iter_round_visitor(start, self.near_points)]
        axes_list = [self.find_axes(n1, n2) for n1, n2 in pairwise(pts)]
        return [(p, i, q, j) for (p, q), (i, j) in zip(pairwise(pts), axes_list)
               ]

    def node_visitor(self,
                     start: Node,
                     method: Literal['DFS', 'BFS'] = 'DFS') -> list[Node]:
        return list(iter_visitor(start, self.near_nodes, method=method))

    def point_visitor(self,
                      start: Node,
                      method: Literal['DFS', 'BFS'] = 'DFS') -> list[Node]:
        return list(iter_visitor(start, self.near_points, method=method))

    # def visitor(self, start: Point, method: Literal['DFS', 'BFS'] = 'DFS'):
    #     return list(iter_visitor(start, self.near_points, method=method))
