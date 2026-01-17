import matplotlib.pyplot as plt


def plot_route_ga_zoom(
    route,
    city_names,
    latitudes,
    longitudes,
    title,
    min_lon,
    max_lon,
    min_lat,
    max_lat
):
    """
    Plottet eine TSP-Route in einem explizit vorgegebenen Zoom-Bereich.
    Städte und Kanten außerhalb des Bereichs werden nicht dargestellt.
    """

    # Route schließen
    lat = [latitudes[i] for i in route] + [latitudes[route[0]]]
    lon = [longitudes[i] for i in route] + [longitudes[route[0]]]

    plt.figure(figsize=(10, 7))

    # Pfeile zeichnen NUR wenn beide Punkte im Sichtfenster liegen
    for i in range(len(route)):
        x1, y1 = lon[i], lat[i]
        x2, y2 = lon[i + 1], lat[i + 1]

        if (
            min_lon <= x1 <= max_lon and
            min_lon <= x2 <= max_lon and
            min_lat <= y1 <= max_lat and
            min_lat <= y2 <= max_lat
        ):
            plt.arrow(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                length_includes_head=True,
                head_width=0.005,
                head_length=0.005,
                linewidth=1.2,
                color="black"
            )

    # Städte plotten NUR wenn sie im Zoom liegen
    for idx in route:
        lon_i = longitudes[idx]
        lat_i = latitudes[idx]

        if min_lon <= lon_i <= max_lon and min_lat <= lat_i <= max_lat:
            plt.scatter(lon_i, lat_i, color="black", s=30)
            plt.text(lon_i, lat_i, city_names[idx], fontsize=8)

    # Startpunkt (falls im Zoom)
    start_idx = route[0]
    if (
        min_lon <= longitudes[start_idx] <= max_lon and
        min_lat <= latitudes[start_idx] <= max_lat
    ):
        plt.scatter(
            longitudes[start_idx],
            latitudes[start_idx],
            color="purple",
            s=80,
            label="Start"
        )

    plt.xlim(min_lon, max_lon)
    plt.ylim(min_lat, max_lat)

    plt.title(title + " (Zoom)")
    plt.xlabel("Längengrad")
    plt.ylabel("Breitengrad")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
