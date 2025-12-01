from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import geopandas as gpd
from shapely.geometry import LineString
import math, heapq, os, tempfile, zipfile
from collections import deque

app = FastAPI(title="UB Routing API (FastAPI)")

# Graph storage
nodes_coord_to_id: Dict[Tuple[float, float], int] = {}
nodes_id_to_coord: Dict[int, Tuple[float, float]] = {}
adj: Dict[int, List[Tuple[int, float, int, dict]]] = (
    {}
)  # u -> list of (v, weight, edge_id, props)
edges: Dict[int, dict] = {}
_next_edge_id = 0


def haversine(coord1, coord2):
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _add_node(coord: Tuple[float, float]) -> int:
    key = (round(coord[0], 6), round(coord[1], 6))
    if key in nodes_coord_to_id:
        return nodes_coord_to_id[key]
    nid = len(nodes_coord_to_id)
    nodes_coord_to_id[key] = nid
    nodes_id_to_coord[nid] = key
    adj[nid] = []
    return nid


def build_graph_from_gdf(gdf):
    global _next_edge_id
    nodes_coord_to_id.clear()
    nodes_id_to_coord.clear()
    adj.clear()
    edges.clear()
    _next_edge_id = 0
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        lines = []
        if geom.geom_type == "LineString":
            lines = [geom]
        else:
            try:
                for part in geom:
                    if part.geom_type == "LineString":
                        lines.append(part)
            except Exception:
                continue
        for line in lines:
            coords = list(line.coords)
            prev_nid = None
            for c in coords:
                lon, lat = c[0], c[1]
                nid = _add_node((lon, lat))
                if prev_nid is not None and prev_nid != nid:
                    # weight based on haversine; optionally adjust by speed/class
                    w = haversine(nodes_id_to_coord[prev_nid], nodes_id_to_coord[nid])
                    props = {}
                    try:
                        props = dict(row)
                    except Exception:
                        props = {}
                    eid = _next_edge_id
                    _next_edge_id += 1
                    edges[eid] = {"u": prev_nid, "v": nid, "weight": w, "props": props}
                    # handle one-way
                    oneway = str(props.get("oneway", "")).lower()
                    if oneway in (
                        "yes",
                        "true",
                        "1",
                        "-1",
                    ):  # treat "-1" later as reversed
                        adj[prev_nid].append((nid, w, eid, props))
                        # if -1 indicates reversed direction (some OSM variations) handle:
                        if oneway == "-1":
                            # only add reversed edge (v->u) as well? Commonly "-1" means one-way reverse
                            adj[nid].append((prev_nid, w, eid, props))
                    else:
                        adj[prev_nid].append((nid, w, eid, props))
                        adj[nid].append((prev_nid, w, eid, props))
                prev_nid = nid
    return {"nodes": len(nodes_id_to_coord), "edges": len(edges)}


# --- Algorithms ---
def dijkstra_shortest(start: int, goal: int):
    if start not in nodes_id_to_coord or goal not in nodes_id_to_coord:
        return None
    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u == goal:
            break
        for v, w, eid, props in adj.get(u, []):
            nd = d + (w if w is not None else 1.0)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if goal not in dist:
        return None
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return {"path": path, "distance_m": dist[goal]}


def bfs_all_paths(start: int, goal: int, max_paths: int = 100):
    paths = []
    q = deque()
    q.append([start])
    while q and len(paths) < max_paths:
        p = q.popleft()
        u = p[-1]
        if u == goal:
            paths.append(p)
            continue
        for v, w, eid, props in adj.get(u, []):
            if v not in p:
                q.append(p + [v])
    return paths


def dfs_all_paths(start: int, goal: int, max_paths: int = 100):
    paths = []
    stack = [(start, [start])]
    while stack and len(paths) < max_paths:
        u, p = stack.pop()
        if u == goal:
            paths.append(p)
            continue
        for v, w, eid, props in adj.get(u, []):
            if v not in p:
                stack.append((v, p + [v]))
    return paths


def min_steps_bfs(start: int, goal: int):
    parent = {start: None}
    q = deque([start])
    found = False
    while q:
        u = q.popleft()
        if u == goal:
            found = True
            break
        for v, w, eid, props in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)
    if not found:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return {"path": path, "steps": len(path) - 1}


# --- API models
class LoadRequest(BaseModel):
    shapefile_path: Optional[str] = None


class PathRequest(BaseModel):
    start: int
    goal: int
    max_paths: Optional[int] = 100


@app.post("/load")
async def load_shapefile(req: LoadRequest):
    path = req.shapefile_path
    if not path or not os.path.exists(path):
        raise HTTPException(400, "shapefile_path missing or not found on server")
    try:
        gdf = gpd.read_file(path)
        info = build_graph_from_gdf(gdf)
        return {"status": "ok", "info": info}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/load-zip")
async def upload_shp_zip(file: UploadFile = File(...)):
    # accept a .zip containing .shp/.dbf/.shx and build graph
    tmpdir = tempfile.mkdtemp()
    zp = os.path.join(tmpdir, "upload.zip")
    with open(zp, "wb") as f:
        f.write(await file.read())
    with zipfile.ZipFile(zp, "r") as z:
        z.extractall(tmpdir)
    # find .shp
    shp = None
    for fname in os.listdir(tmpdir):
        if fname.endswith(".shp"):
            shp = os.path.join(tmpdir, fname)
            break
    if not shp:
        raise HTTPException(400, "No .shp file found in zip")
    gdf = gpd.read_file(shp)
    info = build_graph_from_gdf(gdf)
    return {"status": "ok", "info": info}


@app.get("/nodes")
def get_nodes():
    return [
        {"id": nid, "coord": nodes_id_to_coord[nid]}
        for nid in sorted(nodes_id_to_coord.keys())
    ]


@app.get("/edges")
def get_edges():
    return [{"id": eid, **edges[eid]} for eid in sorted(edges.keys())]


@app.post("/shortest-path")
def api_shortest(req: PathRequest):
    r = dijkstra_shortest(req.start, req.goal)
    if r is None:
        raise HTTPException(404, "no path")
    return r


@app.post("/all-paths/bfs")
def api_bfs(req: PathRequest):
    ps = bfs_all_paths(req.start, req.goal, max_paths=req.max_paths)
    return {"count": len(ps), "paths": ps}


@app.post("/all-paths/dfs")
def api_dfs(req: PathRequest):
    ps = dfs_all_paths(req.start, req.goal, max_paths=req.max_paths)
    return {"count": len(ps), "paths": ps}


@app.post("/min-steps")
def api_min_steps(req: PathRequest):
    r = min_steps_bfs(req.start, req.goal)
    if r is None:
        raise HTTPException(404, "no path")
    return r

