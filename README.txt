TEZPUR AMBULANCE ROUTE OPTIMIZER — MULTI-HOSPITAL DESTINATIONS

1. Place this folder wherever you want.
2. Install dependencies if needed:
   python3 -m pip install pandas networkx matplotlib
3. Generate the graph and hospital destination catalogue:
   python3 main.py
4. Start the local website server from this folder:
   python3 -m http.server 8000
5. Open http://localhost:8000/index.html

Use the starting-direction selector and destination-hospital selector.
The browser runs Dijkstra from the selected start to each of the selected
hospital's nearby road-node candidates and chooses the fastest reachable
candidate. The hospital marker remains at the actual hospital coordinates.
The road-access connector distance is reported separately and is not included
in the road-network travel-time estimate.

The traffic simulation remains available: choose a road from the route,
increase its weight, and recalculate. Congestion is a simulated multiplier,
not live traffic data.
