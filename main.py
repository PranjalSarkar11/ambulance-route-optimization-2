from pathlib import Path
import heapq
import json
import math
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PATHS: supports either a project root with data/ or CSV files
# placed beside main.py.
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").is_dir() else ROOT
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NODES_FILE = DATA_DIR / "tezpur_map_nodes.csv"
ROADS_FILE = DATA_DIR / "map_road_ways.csv"
HOSPITALS_FILE = DATA_DIR / "tezpur_map_hospitals.csv"

for path in (NODES_FILE, ROADS_FILE, HOSPITALS_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")

nodes_df = pd.read_csv(NODES_FILE, dtype={"node_id": str}, keep_default_na=False)
roads_df = pd.read_csv(ROADS_FILE, dtype={"osm_way_id": str}, keep_default_na=False)
hospitals_df = pd.read_csv(HOSPITALS_FILE, keep_default_na=False)

# ------------------------------------------------------------
# DISTANCE AND SPEED MODEL
# ------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    earth_radius = 6_371_000
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return earth_radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Academic assumptions only; these are not measured or live traffic speeds.
ROAD_SPEEDS_KMH = {
    "motorway": 50, "trunk": 40, "primary": 35, "secondary": 30,
    "tertiary": 25, "residential": 20, "unclassified": 20,
    "service": 10, "living_street": 10,
}

coordinates = {
    str(row.node_id): (float(row.latitude), float(row.longitude))
    for row in nodes_df.itertuples()
}

# ------------------------------------------------------------
# BUILD DIRECTED, WEIGHTED MULTIGRAPH
# ------------------------------------------------------------
G = nx.MultiDiGraph()
node_road_names = {}
for row in roads_df.to_dict("records"):
    node_ids = [part.strip() for part in str(row["osm_node_ids_in_order"]).split(";") if part.strip()]
    if len(node_ids) < 2:
        continue
    oneway = str(row.get("oneway", "")).strip().lower()
    if oneway == "-1":
        node_ids.reverse()
    road_type = str(row.get("road_type", "unclassified"))
    road_name = str(row.get("road_name", "Unnamed road"))
    speed = ROAD_SPEEDS_KMH.get(road_type, 20)
    for start, end in zip(node_ids, node_ids[1:]):
        if start not in coordinates or end not in coordinates:
            continue
        distance_m = haversine(*coordinates[start], *coordinates[end])
        travel_time_min = (distance_m / 1000) / speed * 60
        edge_data = {
            "distance_m": distance_m,
            "speed_kmh": speed,
            "travel_time_min": travel_time_min,
            "osm_way_id": str(row.get("osm_way_id", "")),
            "road_name": road_name,
            "road_type": road_type,
        }
        G.add_edge(start, end, **edge_data)
        # OSM -1 means the node order was reversed above; other explicit
        # one-way values prohibit a reverse edge. Unspecified means two-way.
        if oneway not in ("yes", "1", "true", "-1"):
            G.add_edge(end, start, **edge_data)
        if road_name and road_name.casefold() != "unnamed road":
            node_road_names.setdefault(start, set()).add(road_name)
            node_road_names.setdefault(end, set()).add(road_name)

if G.number_of_nodes() == 0:
    raise RuntimeError("The road graph is empty. Check the road-way and node CSV files.")

# ------------------------------------------------------------
# HOSPITAL AND ACCESS NODE
# ------------------------------------------------------------
hospital_rows = hospitals_df[
    hospitals_df["name"].astype(str).str.strip().str.casefold().eq("kanaklata civil hospital")
]
if hospital_rows.empty:
    raise RuntimeError("Exact hospital row 'Kanaklata Civil Hospital' was not found.")
hospital = hospital_rows.iloc[0]
hospital_name = str(hospital["name"])
hospital_lat, hospital_lon = float(hospital["latitude"]), float(hospital["longitude"])

stored_access = str(hospital.get("graph_access_node_id", "")).strip()
if stored_access and stored_access in G:
    hospital_access_node = stored_access
elif "5622971534" in G:
    hospital_access_node = "5622971534"
else:
    hospital_access_node = min(
        (node for node in G.nodes if node in coordinates),
        key=lambda node: haversine(hospital_lat, hospital_lon, *coordinates[node]),
    )
hospital_access_distance_m = haversine(
    hospital_lat, hospital_lon, *coordinates[hospital_access_node]
)

# Multi-destination hospital catalogue. Nearby road nodes are candidates;
# the browser checks directed reachability from the selected start.
hospital_catalogue = []
for hospital_row in hospitals_df.to_dict("records"):
    name = str(hospital_row.get("name", "")).strip()
    tags = str(hospital_row.get("all_tags", "")).lower()
    if not name or "veterinary" in tags or "blood_bank" in tags or "blood bank" in name.lower():
        continue
    if "amenity=hospital" not in tags and "hospital" not in name.lower():
        continue
    try:
        h_lat, h_lon = float(hospital_row["latitude"]), float(hospital_row["longitude"])
    except (KeyError, TypeError, ValueError):
        continue
    nearest_nodes = sorted(G.nodes, key=lambda n: haversine(h_lat, h_lon, *coordinates[n]))[:8]
    candidates = [{
        "node_id": str(n), "lat": float(coordinates[n][0]), "lon": float(coordinates[n][1]),
        "access_distance_m": float(haversine(h_lat, h_lon, *coordinates[n]))
    } for n in nearest_nodes]
    hospital_catalogue.append({
        "id": str(hospital_row.get("osm_id", name)), "name": name,
        "address": str(hospital_row.get("address", "")), "lat": h_lat, "lon": h_lon,
        "access_candidates": candidates
    })
if not hospital_catalogue:
    raise RuntimeError("No general hospital destinations were found in the hospital CSV.")

# ------------------------------------------------------------
# DIJKSTRA: edge weights are non-negative estimated travel times.
# MultiDiGraph parallel edges are all considered.
# ------------------------------------------------------------
def dijkstra(graph, start, destination):
    start, destination = str(start), str(destination)
    if start not in graph or destination not in graph:
        return None, math.inf
    distances = {node: math.inf for node in graph.nodes}
    previous = {}
    distances[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distances[current]:
            continue
        if current == destination:
            break
        for _, neighbour, key, attrs in graph.out_edges(current, keys=True, data=True):
            candidate = current_distance + float(attrs["travel_time_min"])
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = current
                heapq.heappush(queue, (candidate, neighbour))
    if not math.isfinite(distances[destination]):
        return None, math.inf
    path = [destination]
    while path[-1] != start:
        parent = previous.get(path[-1])
        if parent is None:
            return None, math.inf
        path.append(parent)
    path.reverse()
    return path, distances[destination]

# ------------------------------------------------------------
# LOAD THE EIGHT CURATED DEMONSTRATION START LOCATIONS FROM CSV
# ------------------------------------------------------------
required_node_columns = {"is_demo_start", "direction", "display_name"}
if not required_node_columns.issubset(nodes_df.columns):
    raise RuntimeError("The updated node CSV is missing demo-start metadata. Use all three updated CSV files.")
start_rows = nodes_df[nodes_df["is_demo_start"].astype(str).str.lower().eq("true")]
start_locations = []
for row in start_rows.to_dict("records"):
    node_id = str(row["node_id"])
    if node_id not in G:
        continue
    route, route_time = dijkstra(G, node_id, hospital_access_node)
    if route is None:
        print(f"WARNING: {node_id} ({row['direction']}) has no directed route to the hospital; skipped.")
        continue
    start_locations.append({
        "id": node_id,
        "label": str(row["display_name"]),
        "direction": str(row["direction"]),
        "lat": coordinates[node_id][0],
        "lon": coordinates[node_id][1],
        "distance_from_hospital_m": float(row.get("distance_from_hospital_m") or 0),
        "nearest_landmark": str(row.get("nearest_landmark", "")),
        "road_name": str(row.get("associated_road_names", "Unnamed road")),
        "route_time_min": float(route_time),
    })

compass_order = {name: i for i, name in enumerate(
    ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
)}
start_locations.sort(key=lambda item: compass_order.get(item["direction"], 99))
if len(start_locations) < 4:
    raise RuntimeError(f"Only {len(start_locations)} demo start nodes are reachable. Check the updated CSV metadata and graph.")

print("\n========== TEZPUR AMBULANCE ROUTE OPTIMIZER ==========")
print(f"Graph nodes: {G.number_of_nodes()} | Directed edges: {G.number_of_edges()}")
print(f"Hospital: {hospital_name} | Road access node: {hospital_access_node}")
print(f"Hospital-to-road access distance: {hospital_access_distance_m:.1f} m (not included in route time)")
print("\nAvailable directional starting locations:")
for index, item in enumerate(start_locations, 1):
    print(f"{index}. {item['label']} | {item['distance_from_hospital_m']:.0f} m straight-line | "
          f"{item['route_time_min']:.2f} min estimated")

# Pick a start location for the command-line run; Enter selects the northern option.
try:
    answer = input(f"\nChoose a starting location (1-{len(start_locations)}; Enter for 1): ").strip()
except EOFError:
    answer = ""
choice = int(answer) if answer else 1
if not 1 <= choice <= len(start_locations):
    raise ValueError("Invalid starting-location number.")
selected_start = start_locations[choice - 1]
start_node = selected_start["id"]
shortest_path, total_time = dijkstra(G, start_node, hospital_access_node)
if shortest_path is None:
    raise RuntimeError("The selected start location unexpectedly has no route to the hospital.")

route_distance_m = 0.0
for u, v in zip(shortest_path, shortest_path[1:]):
    edge_options = G.get_edge_data(u, v) or {}
    if edge_options:
        route_distance_m += min(float(a["distance_m"]) for a in edge_options.values())

print("\n========== SHORTEST AMBULANCE ROUTE ==========")
print("Starting location:", selected_start["label"])
print("Internal start node:", start_node)
print("Hospital access node:", hospital_access_node)
print("Path:", " -> ".join(shortest_path))
print(f"Road distance: {route_distance_m / 1000:.3f} km")
print(f"Estimated travel time: {total_time:.3f} minutes")
print(f"Road segments: {len(shortest_path) - 1}")

# ------------------------------------------------------------
# SAVE STATIC VISUALIZATIONS
# ------------------------------------------------------------
positions = {node: (coordinates[node][1], coordinates[node][0]) for node in G.nodes if node in coordinates}
plt.figure(figsize=(12, 10))
nx.draw_networkx_edges(G, positions, edge_color="lightgray", width=0.6, arrows=False)
nx.draw_networkx_nodes(G, positions, node_size=5, node_color="steelblue")
nx.draw_networkx_nodes(G, positions, nodelist=[start_node], node_color="green", node_size=65, label="Ambulance start")
nx.draw_networkx_nodes(G, positions, nodelist=[hospital_access_node], node_color="red", node_size=65, label="Hospital road access")
nx.draw_networkx_edges(G, positions, edgelist=list(zip(shortest_path, shortest_path[1:])), edge_color="crimson", width=2.5, arrows=False)
plt.title(f"Ambulance route: {selected_start['direction']} approach to {hospital_name}")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.axis("equal"); plt.grid(alpha=0.2); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shortest_ambulance_route.png", dpi=250, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# CONGESTION AND ROAD-CLOSURE SIMULATIONS (safe for short paths)
# ------------------------------------------------------------
print("\n========== CONGESTION SIMULATION ==========")
if len(shortest_path) < 2:
    print("Route has no road segment; congestion simulation skipped.")
else:
    mid = max(0, (len(shortest_path) - 2) // 2)
    a, b = shortest_path[mid], shortest_path[mid + 1]
    edge_map = G.get_edge_data(a, b) or {}
    saved = []
    for key, attrs in edge_map.items():
        saved.append((key, dict(attrs)))
        attrs["travel_time_min"] *= 10
    congested_path, congested_time = dijkstra(G, start_node, hospital_access_node)
    print(f"Congestion applied to {a} -> {b} (10x edge travel time).")
    if congested_path:
        print(f"New estimated time: {congested_time:.3f} min; alternative route: {congested_path != shortest_path}")
    else:
        print("No route remains under the congestion scenario.")
    for key, attrs in saved:
        G[a][b][key].update(attrs)

print("\n========== ROAD CLOSURE SIMULATION ==========")
if len(shortest_path) < 2:
    print("Route has no road segment; closure simulation skipped.")
else:
    a, b = shortest_path[0], shortest_path[1]
    edge_data = G.get_edge_data(a, b) or {}
    if not edge_data:
        print("Selected road segment is missing; closure simulation skipped.")
    else:
        selected_way = str(min(edge_data.values(), key=lambda x: x["travel_time_min"]).get("osm_way_id", ""))
        removed = []
        # Close this physical OSM way segment in both directions, including parallel edges.
        for u, v in ((a, b), (b, a)):
            for key, attrs in list((G.get_edge_data(u, v) or {}).items()):
                if str(attrs.get("osm_way_id", "")) == selected_way:
                    removed.append((u, v, key, dict(attrs)))
                    G.remove_edge(u, v, key)
        alternate_path, alternate_time = dijkstra(G, start_node, hospital_access_node)
        print(f"Closed OSM way {selected_way} between {a} and {b}.")
        if alternate_path:
            print(f"New estimated time: {alternate_time:.3f} min; alternative route: {alternate_path != shortest_path}")
        else:
            print("No route remains after this road closure.")
        for u, v, key, attrs in removed:
            G.add_edge(u, v, key=key, **attrs)
        print("Original graph restored after simulation.")

# ------------------------------------------------------------
# EXPORT DATA FOR index.html
# ------------------------------------------------------------
website_data = {
    "nodes": [
        {"id": str(node), "lat": float(coordinates[node][0]), "lon": float(coordinates[node][1])}
        for node in G.nodes if node in coordinates
    ],
    "edges": [
        {
            "source": str(u), "target": str(v),
            "travel_time_min": float(attrs["travel_time_min"]),
            "distance_m": float(attrs["distance_m"]),
            "osm_way_id": str(attrs.get("osm_way_id", "")),
            "road_name": str(attrs.get("road_name", "Unnamed road")),
            "road_type": str(attrs.get("road_type", "unknown")),
        }
        for u, v, key, attrs in G.edges(keys=True, data=True)
        if u in coordinates and v in coordinates
    ],
    "hospital": {"name": hospital_name, "lat": hospital_lat, "lon": hospital_lon},
    "hospitals": hospital_catalogue,
    "hospital_access_node": str(hospital_access_node),
    "hospital_access_distance_m": float(hospital_access_distance_m),
    "default_start_node": start_node,
    "default_start_label": selected_start["label"],
    "suggested_start_nodes": start_locations,
    "default_route": shortest_path,
    "default_route_time_min": float(total_time),
    "default_route_distance_km": float(route_distance_m / 1000),
    "model_note": "Estimated travel times use assumed road speeds; no live traffic data is used.",
}
with open(OUTPUT_DIR / "road_network.json", "w", encoding="utf-8") as f:
    json.dump(website_data, f, indent=2)

nx.write_graphml(G, OUTPUT_DIR / "road_network.graphml")
print("\nWebsite data exported to:", OUTPUT_DIR / "road_network.json")
print("Open index.html through a local web server (not file://) to use the map.")
