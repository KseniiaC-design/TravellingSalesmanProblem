import matplotlib.pyplot as plt

def plot_route_ga(route, city_names, latitudes, longitudes, title):
    # Route schließen (zurück zum Start)
    lat = [latitudes[i] for i in route] + [latitudes[route[0]]]
    lon = [longitudes[i] for i in route] + [longitudes[route[0]]]

    plt.figure(figsize=(10, 7))

    # Pfeile zeichnen
    for i in range(len(route)):
        plt.arrow(
            lon[i],
            lat[i],
            lon[i + 1] - lon[i],
            lat[i + 1] - lat[i],
            length_includes_head=True,
            head_width=0.02,
            head_length=0.02,
            linewidth=1.5,
            color="black"
        )

    start_idx = route[0]

    # Schwarze Punkte für alle Städte außer Start
    for idx in route[1:]:
        plt.scatter(
            longitudes[idx],
            latitudes[idx],
            color="black",
            s=30
        )
        plt.text(
            longitudes[idx],
            latitudes[idx],
            city_names[idx],
            fontsize=9
        )

    # Startpunkt: kleiner & lila
    plt.scatter(
        longitudes[start_idx],
        latitudes[start_idx],
        color="purple",
        s=80,
        label="Start"
    )
    plt.text(
        longitudes[start_idx],
        latitudes[start_idx],
        city_names[start_idx],
        fontsize=10,
        fontweight="bold"
    )

    plt.title(title)
    plt.xlabel("Längengrad")
    plt.ylabel("Breitengrad")
    plt.legend()
    plt.grid(True)
    plt.show()





