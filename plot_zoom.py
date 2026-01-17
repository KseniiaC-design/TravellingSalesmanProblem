import matplotlib.pyplot as plt


def clip_line_to_box(x1, y1, x2, y2, min_x, max_x, min_y, max_y):
    """
    Schneidet eine Linie auf ein Rechteck (Liang-Barsky-Algorithmus).
    Gibt (cx1, cy1, cx2, cy2) oder None zurück.
    """
    dx = x2 - x1
    dy = y2 - y1

    p = [-dx, dx, -dy, dy]
    q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]

    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t

    cx1 = x1 + u1 * dx
    cy1 = y1 + u1 * dy
    cx2 = x1 + u2 * dx
    cy2 = y1 + u2 * dy

    return cx1, cy1, cx2, cy2


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
    Zoom-Plot mit abgeschnittenen Kanten am Rand.
    Städte außerhalb werden nicht angezeigt,
    Kanten laufen bis zum Plot-Rand.
    """

    lat = [latitudes[i] for i in route] + [latitudes[route[0]]]
    lon = [longitudes[i] for i in route] + [longitudes[route[0]]]

    plt.figure(figsize=(10, 7))

    # ----- KANTEN MIT CLIPPING -----
    for i in range(len(route)):
        x1, y1 = lon[i], lat[i]
        x2, y2 = lon[i + 1], lat[i + 1]

        clipped = clip_line_to_box(
            x1, y1, x2, y2,
            min_lon, max_lon,
            min_lat, max_lat
        )

        if clipped is not None:
            cx1, cy1, cx2, cy2 = clipped
            plt.plot([cx1, cx2], [cy1, cy2], color="black", linewidth=1.2)

    # ----- STÄDTE IM ZOOM -----
    for idx in route:
        lon_i = longitudes[idx]
        lat_i = latitudes[idx]

        if min_lon <= lon_i <= max_lon and min_lat <= lat_i <= max_lat:
            plt.scatter(lon_i, lat_i, color="black", s=30)
            plt.text(lon_i, lat_i, city_names[idx], fontsize=8)

    # ----- STARTPUNKT -----
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
