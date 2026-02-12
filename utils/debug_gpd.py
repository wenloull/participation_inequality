
import geopandas as gpd
import traceback

try:
    path = gpd.datasets.get_path('naturalearth_lowres')
    print(f"Path: {path}")
    world = gpd.read_file(path)
    print("Loaded successfully")
    print(world.head())
except Exception:
    traceback.print_exc()
