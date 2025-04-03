import osmnx as ox
import pandas as pd
import folium

# Define Istanbul districts
districts = [
    "Başakşehir, Istanbul, Turkey",
    "Güngören, Istanbul, Turkey",
    "Esenler, Istanbul, Turkey",
    "Bağcılar, Istanbul, Turkey",
    "Bahçelievler, Istanbul, Turkey",
    "Bakırköy, Istanbul, Turkey"
]

# Define broader tags to get more data
tags = {"building": ["commercial", "retail", "industrial"],
        "amenity": True,
        "shop": True,
        "office": True,
        "tourism": True,
        "leisure": True}  # Added other tags to ensure data is fetched.


# Fetch data for all districts
pois = []
for district in districts:
    try:
        data = ox.features_from_place(district, tags)
        pois.append(data)
    except Exception as e:
        print(f"Error fetching data for {district}: {e}")

# Combine all data
pois_df = pd.concat(pois)

# Keep only useful columns
pois_df = pois_df[["name", "geometry", "amenity", "shop", "office", "tourism", "leisure", "building"]].dropna(how="all", subset=["name", "amenity", "shop", "office", "tourism", "leisure", "building"]) # Modified dropna to drop if all are null.

# Extract latitude & longitude
pois_df["latitude"] = pois_df["geometry"].apply(lambda x: x.y if hasattr(x, 'y') else None)
pois_df["longitude"] = pois_df["geometry"].apply(lambda x: x.x if hasattr(x, 'x') else None)

# Remove rows without coordinates
pois_df = pois_df.dropna(subset=["latitude", "longitude"])

# Save as CSV
pois_df.to_csv("real_istanbul_osm_data.csv", index=False)

# Create interactive map
m = folium.Map(location=[41.05, 28.87], zoom_start=12, tiles="cartodbpositron")

# Add locations to the map
for _, row in pois_df.iterrows():
    try:
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,  # Adjust the radius for dot size
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=row["name"]
        ).add_to(m)
    except (TypeError, ValueError) as e:
        print(f"Error adding marker: {e}, for row {row}")


# Save interactive map
m.save("istanbul_real_osm_map.html")

print(f"Extracted {len(pois_df)} real locations from OSM!")