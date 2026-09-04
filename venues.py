"""WC2026 venue lookup: city -> (lat, lng, altitude_m).

Only includes the 16 host cities plus a few common cities that appear in the
martj42 dataset for altitude feature consistency. The altitude flag is what
the model actually uses; lat/long are for future weather/distance work.
"""

WC2026_VENUES = {
    # USA (11). ESPN/FIFA use the local-municipality names (Inglewood, Foxborough,
    # East Rutherford, Arlington, Miami Gardens, Santa Clara) rather than the
    # metro names, so we include both variants.
    "Atlanta":            (33.755, -84.401,  315),
    "Boston":             (42.361, -71.057,   43),
    "Foxborough":         (42.062, -71.247,   55),  # Gillette Stadium (Boston metro)
    "Dallas":             (32.776, -96.797,  131),
    "Arlington":          (32.736, -97.108,  168),  # AT&T Stadium (Dallas metro)
    "Houston":            (29.760, -95.369,   12),
    "Kansas City":        (39.099, -94.578,  274),
    "Los Angeles":        (34.052, -118.244,   71),
    "Inglewood":          (33.961, -118.353,   28), # SoFi Stadium (LA metro)
    "Miami":              (25.761, -80.192,    2),
    "Miami Gardens":      (25.957, -80.239,    4),  # Hard Rock Stadium
    "New York":           (40.713, -74.006,   10),
    "East Rutherford":    (40.812, -74.075,    7),  # MetLife Stadium (NY/NJ)
    "Philadelphia":       (39.953, -75.165,   12),
    "San Francisco":      (37.775, -122.419,   16),
    "Santa Clara":        (37.355, -121.955,   18), # Levi's Stadium
    "Seattle":            (47.606, -122.332,   53),
    # Mexico (3) - notable altitude.
    "Mexico City":        (19.433, -99.133, 2240),
    "Guadalajara":        (20.677, -103.348, 1566),
    "Zapopan":            (20.720, -103.391, 1567), # Estadio Akron (Guadalajara metro)
    "Monterrey":          (25.687, -100.317,  540),
    "Guadalupe":          (25.677, -100.252,  500), # Estadio BBVA (Monterrey metro)
    # Canada (2) - sea level.
    "Toronto":            (43.651, -79.347,   76),
    "Vancouver":          (49.283, -123.121,   70),
}

# Cities elsewhere that show up in martj42's historical data with non-trivial
# altitude. Used so the `venue_altitude_m` feature is well-populated for
# training, not just the WC venues. Add more as needed.
HISTORICAL_HIGH_ALTITUDE = {
    "La Paz":             (-16.500, -68.150, 3640),
    "Quito":              (-0.180,  -78.467, 2850),
    "Bogota":             (4.711,   -74.072, 2640),
    "Bogotá":             (4.711,   -74.072, 2640),
    "Cusco":              (-13.531, -71.967, 3399),
    "Addis Ababa":        (9.030,   38.740,  2355),
    "Asmara":             (15.333,  38.933,  2325),
    "Sana'a":             (15.369,  44.191,  2250),
    "Johannesburg":       (-26.205, 28.050,  1753),
    "Pretoria":           (-25.745, 28.188,  1339),
    "Nairobi":            (-1.286,  36.817,  1795),
    "Kampala":            (0.347,   32.583,  1190),
    "Lhasa":              (29.652,  91.172,  3650),
}

ALTITUDE_LOOKUP = {
    **{city: meta[2] for city, meta in WC2026_VENUES.items()},
    **{city: meta[2] for city, meta in HISTORICAL_HIGH_ALTITUDE.items()},
}


def altitude_of(city: str | float | None) -> int:
    if not isinstance(city, str):
        return 0
    return ALTITUDE_LOOKUP.get(city, 0)
