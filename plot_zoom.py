import matplotlib.pyplot as plt

def plot_route_ga_zoom(
    route,
    city_names,
    latitudes,
    longitudes,
    title,
    lon_min,
    lon_max,
    lat_min,
    lat_max
):
    lat = [latitudes[i] for i in route] + [latitudes[route[0]]]
    lon = [longitudes[i] for i in route] + [longitudes[route[0]]]

    plt.figure(figsize=(8, 6))

    for i in range(len(route)):
        plt.arrow(
            lon[i],
            lat[i],
            lon[i + 1] - lon[i],
            lat[i + 1] - lat[i],
            length_includes_head=True,
            head_width=0.01,
            head_length=0.01,
            linewidth=1.2,
            color="black"
        )

    for idx in route:
        if lon_min <= longitudes[idx] <= lon_max and lat_min <= latitudes[idx] <= lat_max:
            plt.scatter(longitudes[idx], latitudes[idx], color="black", s=30)
            plt.text(
                longitudes[idx],
                latitudes[idx],
                city_names[idx],
                fontsize=8
            )

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)

    plt.title(title)
    plt.xlabel("Längengrad")
    plt.ylabel("Breitengrad")
    plt.grid(True)
    plt.show()

