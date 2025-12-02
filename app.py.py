import streamlit as st
import pandas as pd
import heapq
import math
import csv

# --- Page Configuration and Styling ---
st.set_page_config(page_title="EV Route AI", page_icon="⚡", layout="wide")

# --- 1. Data Definition & Loading ---
# Approximate Delhi coordinates (Lat, Long) for A* heuristic and map display
COORDS = {
    "Janakpuri": (28.62, 77.09),
    "Tilak Nagar": (28.63, 77.09),
    "Rajouri Garden": (28.65, 77.12),
    "Punjabi Bagh": (28.67, 77.13),
    "Pitampura": (28.70, 77.12),
    "Rohini": (28.74, 77.11),
    "Karol Bagh": (28.65, 77.19),
    "Connaught Place": (28.63, 77.22),
    "Paschim Vihar": (28.67, 77.09),
    "Shalimar Bagh": (28.72, 77.16)
}

@st.cache_data
def load_graph_data(file_path="delhi_routes.csv"):
    """
    Loads graph data from CSV and builds the adjacency list.
    This function is cached to prevent reloading data on every user interaction.
    """
    graph = {}
    try:
        # NOTE: This assumes 'delhi_routes.csv' is in the same directory for deployment.
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                src, dest, dist = row['From'], row['To'], float(row['Distance'])
                # Undirected graph: add edge in both directions
                graph.setdefault(src, []).append((dest, dist))
                graph.setdefault(dest, []).append((src, dist))
    except FileNotFoundError:
        # In a deployed environment, this might indicate the data file is missing
        st.error(f"❌ Required data file '{file_path}' not found. Please ensure it is in the same directory.")
        return {}
    return graph

GRAPH = load_graph_data()
LOCATIONS = sorted(list(COORDS.keys()))
BATTERY_RANGE = 15.0 # Default battery range in km for the slider

# --- 2. Algorithms (Dijkstra and A*) ---

def heuristic(a, b):
    """
    Calculates Euclidean distance (straight-line distance on a flat plane) 
    as the A* heuristic from node 'a' to node 'b'.
    """
    if a not in COORDS or b not in COORDS: return 0
    (x1, y1), (x2, y2) = COORDS[a], COORDS[b]
    # Simple calculation for a heuristic, as the coordinate difference is proportional to distance
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def dijkstra(graph, start, goal):
    """Implementation of Dijkstra's algorithm to find the shortest path."""
    # Priority Queue stores tuples: (cost, node, path_list_so_far)
    pq = [(0, start, [])] 
    distances = {node: float('inf') for node in LOCATIONS}
    distances[start] = 0
    visited = set()

    while pq:
        (cost, node, path) = heapq.heappop(pq)

        if node in visited: continue
        visited.add(node)
        path = path + [node]

        if node == goal:
            return cost, path

        for neighbor, weight in graph.get(node, []):
            new_cost = cost + weight
            if new_cost < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path))
    return float("inf"), []

def astar(graph, start, goal):
    """Implementation of A* Search algorithm, using a heuristic for faster results."""
    # Priority Queue stores tuples: (f = g+h, g, node, path_list_so_far)
    pq = [(heuristic(start, goal), 0, start, [])]
    g_scores = {node: float('inf') for node in LOCATIONS}
    g_scores[start] = 0
    visited = set()

    while pq:
        (f, g, node, path) = heapq.heappop(pq)
        
        if node in visited: continue
        visited.add(node)
        path = path + [node]

        if node == goal:
            return g, path

        for neighbor, weight in graph.get(node, []):
            tentative_g = g + weight
            if tentative_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g
                h = heuristic(neighbor, goal)
                f_score = tentative_g + h
                heapq.heappush(pq, (f_score, tentative_g, neighbor, path))
    return float("inf"), []

# --- 3. Streamlit UI Layout and Interaction ---

st.title("⚡ EV Route Optimization AI")
st.markdown("Use this tool to find the shortest path between locations in Delhi and check if the route is within the EV's battery range.")

# SideBar for User Input
with st.sidebar:
    st.header("Trip Planner")
    start_loc = st.selectbox("📍 Start Location", LOCATIONS, index=0)
    end_loc = st.selectbox("🏁 Destination", LOCATIONS, index=min(len(LOCATIONS) - 1, 7))
    
    selected_algorithm = st.radio("Pathfinding Algorithm", ["Dijkstra", "A* Star"], horizontal=True)
    
    current_battery_range = st.slider("Current Battery Range (km)", 5, 50, BATTERY_RANGE, step=1.0)
    
    find_button = st.button("Calculate Best Route", type="primary")

# Main Content Columns
col1, col2 = st.columns([1, 1])

if find_button:
    if start_loc == end_loc:
        st.warning("Start and Destination cannot be the same.")
    elif not GRAPH:
        st.warning("Cannot run search, graph data is missing.")
    else:
        # Run selected algorithm
        if selected_algorithm == "Dijkstra":
            cost, path = dijkstra(GRAPH, start_loc, end_loc)
        else:
            cost, path = astar(GRAPH, start_loc, end_loc)

        if cost == float("inf"):
            st.error("No path found between these two locations.")
        else:
            with col1:
                st.subheader("✅ Route Found")
                
                # Display Route Path
                st.markdown(f"**Path ({len(path)} stops):**")
                st.code(" → ".join(path), language='python')
                
                # Metrics
                m1, m2 = st.columns(2)
                m1.metric("Distance", f"{cost:.2f} km")
                m2.metric("Algorithm", selected_algorithm)
                
                # Battery Check
                st.write("---")
                st.subheader("🔋 Battery Check")
                
                if cost > current_battery_range:
                    st.error(f"❌ **FAIL:** Trip distance ({cost:.2f} km) exceeds battery range ({current_battery_range} km). A recharge stop is required.")
                else:
                    st.success(f"✅ **PASS:** Trip distance ({cost:.2f} km) is within battery range.")

            with col2:
                st.subheader("📍 Route Visualization")
                # Prepare data for map visualization
                map_data_list = []
                for city in path:
                    if city in COORDS:
                        lat, lon = COORDS[city]
                        # Use city name as a label for the point
                        map_data_list.append({"lat": lat, "lon": lon, "Location": city}) 
                
                df_map = pd.DataFrame(map_data_list)
                
                # Show the map centered on the first location
                st.map(df_map, zoom=11)
                st.caption(f"Showing the {len(df_map)} stops on the map.")

else:
    # Initial view (all locations)
    with col1:
        st.info("Select your starting and destination locations from the sidebar, then click 'Calculate Best Route'.")
    with col2:
        st.subheader("All Available Locations")
        # DataFrame for map showing all locations
        all_points = [{"lat": v[0], "lon": v[1]} for k,v in COORDS.items()]
        st.map(pd.DataFrame(all_points), zoom=10)