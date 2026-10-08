# -*- coding: utf-8 -*-
"""
Reference geography for the LIRR access network.

STATIONS: real LIRR stations, real branch assignment, coordinates accurate to
roughly +/-0.005 deg (~400 m). Replace with the MTA GTFS stops.txt for exact
positions -- schema is (name, lat, lon, fare_zone, county).

ANCHORS: real Long Island villages / CDPs / hamlets with approximate 2020
populations, used as kernel centers for the synthetic census-tract layer.
Replace with TIGER/Line tract centroids to get the real layer.
"""

# ---------------------------------------------------------------- stations
# (name, lat, lon, fare_zone, county)
STATIONS = [
    # --- City Terminal Zone ---
    ("Penn Station",            40.7506, -73.9935, 1, "New York"),
    ("Grand Central Madison",   40.7527, -73.9772, 1, "New York"),
    ("Hunterspoint Ave",        40.7424, -73.9540, 1, "Queens"),
    ("Long Island City",        40.7440, -73.9560, 1, "Queens"),
    ("Atlantic Terminal",       40.6840, -73.9770, 1, "Kings"),
    ("Nostrand Ave",            40.6800, -73.9500, 1, "Kings"),
    ("East New York",           40.6780, -73.8990, 1, "Kings"),
    ("Woodside",                40.7455, -73.9030, 1, "Queens"),
    ("Forest Hills",            40.7210, -73.8450, 3, "Queens"),
    ("Kew Gardens",             40.7110, -73.8300, 3, "Queens"),
    ("Jamaica",                 40.7005, -73.8077, 3, "Queens"),
    # --- Port Washington Branch ---
    ("Mets-Willets Point",      40.7540, -73.8455, 3, "Queens"),
    ("Flushing-Main Street",    40.7595, -73.8300, 3, "Queens"),
    ("Murray Hill",             40.7635, -73.8175, 3, "Queens"),
    ("Broadway",                40.7660, -73.8110, 3, "Queens"),
    ("Auburndale",              40.7610, -73.7920, 3, "Queens"),
    ("Bayside",                 40.7690, -73.7710, 3, "Queens"),
    ("Douglaston",              40.7690, -73.7450, 3, "Queens"),
    ("Little Neck",             40.7700, -73.7320, 3, "Queens"),
    ("Great Neck",              40.7995, -73.7280, 4, "Nassau"),
    ("Manhasset",               40.7980, -73.6950, 4, "Nassau"),
    ("Plandome",                40.8050, -73.7010, 4, "Nassau"),
    ("Port Washington",         40.8180, -73.6980, 4, "Nassau"),
    # --- Far Rockaway Branch ---
    ("St. Albans",              40.6920, -73.7600, 3, "Queens"),
    ("Locust Manor",            40.6690, -73.7710, 3, "Queens"),
    ("Laurelton",               40.6700, -73.7460, 3, "Queens"),
    ("Rosedale",                40.6650, -73.7350, 3, "Queens"),
    ("Valley Stream",           40.6640, -73.7080, 4, "Nassau"),
    ("Gibson",                  40.6560, -73.7180, 4, "Nassau"),
    ("Hewlett",                 40.6400, -73.6960, 4, "Nassau"),
    ("Woodmere",                40.6320, -73.7130, 4, "Nassau"),
    ("Cedarhurst",              40.6230, -73.7250, 4, "Nassau"),
    ("Lawrence",                40.6160, -73.7300, 4, "Nassau"),
    ("Inwood",                  40.6180, -73.7440, 4, "Nassau"),
    ("Far Rockaway",            40.6010, -73.7480, 4, "Queens"),
    # --- Long Beach Branch ---
    ("Lynbrook",                40.6580, -73.6720, 4, "Nassau"),
    ("Centre Avenue",           40.6540, -73.6660, 4, "Nassau"),
    ("East Rockaway",           40.6420, -73.6690, 4, "Nassau"),
    ("Oceanside",               40.6360, -73.6400, 4, "Nassau"),
    ("Island Park",             40.6070, -73.6540, 4, "Nassau"),
    ("Long Beach",              40.5900, -73.6620, 4, "Nassau"),
    # --- West Hempstead Branch ---
    ("Westwood",                40.6720, -73.7030, 4, "Nassau"),
    ("Malverne",                40.6740, -73.6740, 4, "Nassau"),
    ("Lakeview",                40.6870, -73.6720, 4, "Nassau"),
    ("Hempstead Gardens",       40.6970, -73.6640, 4, "Nassau"),
    ("West Hempstead",          40.7010, -73.6520, 4, "Nassau"),
    # --- Hempstead Branch ---
    ("Hollis",                  40.7110, -73.7620, 3, "Queens"),
    ("Queens Village",          40.7180, -73.7420, 3, "Queens"),
    ("Bellerose",               40.7250, -73.7150, 4, "Nassau"),
    ("Floral Park",             40.7220, -73.7030, 4, "Nassau"),
    ("Stewart Manor",           40.7190, -73.6900, 4, "Nassau"),
    ("Nassau Boulevard",        40.7130, -73.6690, 4, "Nassau"),
    ("Garden City",             40.7220, -73.6390, 4, "Nassau"),
    ("Country Life Press",      40.7160, -73.6280, 4, "Nassau"),
    ("Hempstead",               40.7060, -73.6180, 4, "Nassau"),
    # --- Babylon Branch ---
    ("Rockville Centre",        40.6590, -73.6410, 4, "Nassau"),
    ("Baldwin",                 40.6560, -73.6100, 4, "Nassau"),
    ("Freeport",                40.6570, -73.5830, 4, "Nassau"),
    ("Merrick",                 40.6630, -73.5510, 7, "Nassau"),
    ("Bellmore",                40.6690, -73.5290, 7, "Nassau"),
    ("Wantagh",                 40.6800, -73.5090, 7, "Nassau"),
    ("Seaford",                 40.6720, -73.4920, 7, "Nassau"),
    ("Massapequa",              40.6790, -73.4710, 7, "Nassau"),
    ("Massapequa Park",         40.6810, -73.4520, 7, "Nassau"),
    ("Amityville",              40.6790, -73.4180, 9, "Suffolk"),
    ("Copiague",                40.6800, -73.3980, 9, "Suffolk"),
    ("Lindenhurst",             40.6870, -73.3740, 9, "Suffolk"),
    ("Babylon",                 40.6970, -73.3260, 9, "Suffolk"),
    # --- Montauk Branch (east of Babylon) ---
    ("Bay Shore",               40.7250, -73.2470, 9, "Suffolk"),
    ("Islip",                   40.7300, -73.2100, 10, "Suffolk"),
    ("Great River",             40.7250, -73.1660, 10, "Suffolk"),
    ("Oakdale",                 40.7350, -73.1370, 10, "Suffolk"),
    ("Sayville",                40.7420, -73.0790, 10, "Suffolk"),
    ("Patchogue",               40.7650, -73.0130, 10, "Suffolk"),
    ("Bellport",                40.7660, -72.9440, 12, "Suffolk"),
    ("Mastic-Shirley",          40.8060, -72.8420, 12, "Suffolk"),
    ("Speonk",                  40.8210, -72.7080, 12, "Suffolk"),
    ("Westhampton",             40.8280, -72.6480, 14, "Suffolk"),
    ("Hampton Bays",            40.8700, -72.5170, 14, "Suffolk"),
    ("Southampton",             40.8890, -72.3880, 14, "Suffolk"),
    ("Bridgehampton",           40.9370, -72.3010, 14, "Suffolk"),
    ("East Hampton",            40.9600, -72.1900, 14, "Suffolk"),
    ("Amagansett",              40.9760, -72.1440, 14, "Suffolk"),
    ("Montauk",                 41.0350, -71.9600, 14, "Suffolk"),
    # --- Main Line / Ronkonkoma Branch ---
    ("New Hyde Park",           40.7350, -73.6870, 4, "Nassau"),
    ("Merillon Avenue",         40.7400, -73.6540, 4, "Nassau"),
    ("Mineola",                 40.7470, -73.6410, 4, "Nassau"),
    ("Carle Place",             40.7510, -73.6110, 4, "Nassau"),
    ("Westbury",                40.7550, -73.5880, 7, "Nassau"),
    ("Hicksville",              40.7660, -73.5290, 7, "Nassau"),
    ("Bethpage",                40.7440, -73.4830, 7, "Nassau"),
    ("Farmingdale",             40.7330, -73.4460, 7, "Nassau"),
    ("Pinelawn",                40.7690, -73.3980, 9, "Suffolk"),
    ("Wyandanch",               40.7530, -73.3760, 9, "Suffolk"),
    ("Deer Park",               40.7620, -73.3230, 9, "Suffolk"),
    ("Brentwood",               40.7830, -73.2560, 9, "Suffolk"),
    ("Central Islip",           40.7900, -73.2000, 10, "Suffolk"),
    ("Ronkonkoma",              40.8140, -73.1130, 10, "Suffolk"),
    # --- Greenport Branch ---
    ("Medford",                 40.8180, -72.9860, 12, "Suffolk"),
    ("Riverhead",               40.9170, -72.6620, 12, "Suffolk"),
    ("Mattituck",               40.9930, -72.5350, 14, "Suffolk"),
    ("Southold",                41.0640, -72.4260, 14, "Suffolk"),
    ("Greenport",               41.1020, -72.3630, 14, "Suffolk"),
    # --- Port Jefferson Branch ---
    ("Syosset",                 40.8250, -73.5020, 7, "Nassau"),
    ("Cold Spring Harbor",      40.8480, -73.4530, 9, "Suffolk"),
    ("Huntington",              40.8570, -73.4090, 9, "Suffolk"),
    ("Greenlawn",               40.8690, -73.3630, 9, "Suffolk"),
    ("Northport",               40.8830, -73.3290, 9, "Suffolk"),
    ("Kings Park",              40.8880, -73.2530, 10, "Suffolk"),
    ("Smithtown",               40.8580, -73.1900, 10, "Suffolk"),
    ("St. James",               40.8720, -73.1560, 10, "Suffolk"),
    ("Stony Brook",             40.9110, -73.1300, 10, "Suffolk"),
    ("Port Jefferson",          40.9450, -73.0640, 10, "Suffolk"),
    # --- Oyster Bay Branch ---
    ("East Williston",          40.7580, -73.6350, 4, "Nassau"),
    ("Albertson",               40.7710, -73.6470, 4, "Nassau"),
    ("Roslyn",                  40.7930, -73.6480, 4, "Nassau"),
    ("Greenvale",               40.8100, -73.6280, 7, "Nassau"),
    ("Glen Head",               40.8330, -73.6180, 7, "Nassau"),
    ("Sea Cliff",               40.8420, -73.6440, 7, "Nassau"),
    ("Glen Street",             40.8620, -73.6300, 7, "Nassau"),
    ("Glen Cove",               40.8640, -73.6230, 7, "Nassau"),
    ("Locust Valley",           40.8760, -73.5960, 7, "Nassau"),
    ("Oyster Bay",              40.8740, -73.5320, 7, "Nassau"),
]

# ---------------------------------------------------------------- branches
# Ordered station sequences. peak_tph = AM peak westbound trains/hour,
# cars = typical consist length, color = MTA LIRR branch color.
BRANCHES = {
    "City Terminal": dict(
        color="#4D5357", peak_tph=26, cars=10,
        stops=["Jamaica", "Kew Gardens", "Forest Hills", "Woodside",
               "Penn Station"]),
    "Atlantic": dict(
        color="#4D5357", peak_tph=6, cars=10,
        stops=["Jamaica", "East New York", "Nostrand Ave",
               "Atlantic Terminal"]),
    "GCM": dict(
        color="#4D5357", peak_tph=18, cars=10,
        stops=["Woodside", "Grand Central Madison"]),
    "LIC": dict(
        color="#4D5357", peak_tph=2, cars=10,
        stops=["Woodside", "Hunterspoint Ave", "Long Island City"]),
    "Port Washington": dict(
        color="#C60C30", peak_tph=8, cars=10,
        stops=["Port Washington", "Plandome", "Manhasset", "Great Neck",
               "Little Neck", "Douglaston", "Bayside", "Auburndale",
               "Broadway", "Murray Hill", "Flushing-Main Street",
               "Mets-Willets Point", "Woodside", "Penn Station"]),
    "Far Rockaway": dict(
        color="#6E3219", peak_tph=4, cars=8,
        stops=["Far Rockaway", "Inwood", "Lawrence", "Cedarhurst", "Woodmere",
               "Hewlett", "Gibson", "Valley Stream", "Rosedale", "Laurelton",
               "Locust Manor", "St. Albans", "Jamaica"]),
    "Long Beach": dict(
        color="#FF6319", peak_tph=5, cars=10,
        stops=["Long Beach", "Island Park", "Oceanside", "East Rockaway",
               "Centre Avenue", "Lynbrook", "Valley Stream", "Jamaica"]),
    "West Hempstead": dict(
        color="#00A1DE", peak_tph=2, cars=6,
        stops=["West Hempstead", "Hempstead Gardens", "Lakeview", "Malverne",
               "Westwood", "Valley Stream", "Jamaica"]),
    "Hempstead": dict(
        color="#CE8E00", peak_tph=4, cars=8,
        stops=["Hempstead", "Country Life Press", "Garden City",
               "Nassau Boulevard", "Stewart Manor", "Floral Park",
               "Bellerose", "Queens Village", "Hollis", "Jamaica"]),
    "Babylon": dict(
        color="#00985F", peak_tph=10, cars=12,
        stops=["Babylon", "Lindenhurst", "Copiague", "Amityville",
               "Massapequa Park", "Massapequa", "Seaford", "Wantagh",
               "Bellmore", "Merrick", "Freeport", "Baldwin",
               "Rockville Centre", "Lynbrook", "Jamaica"]),
    "Montauk": dict(
        color="#00B2A9", peak_tph=2, cars=6,
        stops=["Montauk", "Amagansett", "East Hampton", "Bridgehampton",
               "Southampton", "Hampton Bays", "Westhampton", "Speonk",
               "Mastic-Shirley", "Bellport", "Patchogue", "Sayville",
               "Oakdale", "Great River", "Islip", "Bay Shore", "Babylon"]),
    "Ronkonkoma": dict(
        color="#A626AA", peak_tph=8, cars=12,
        stops=["Ronkonkoma", "Central Islip", "Brentwood", "Deer Park",
               "Wyandanch", "Pinelawn", "Farmingdale", "Bethpage",
               "Hicksville", "Westbury", "Carle Place", "Mineola",
               "Merillon Avenue", "New Hyde Park", "Floral Park",
               "Queens Village", "Jamaica"]),
    "Greenport": dict(
        color="#A626AA", peak_tph=1, cars=4,
        stops=["Greenport", "Southold", "Mattituck", "Riverhead", "Medford",
               "Ronkonkoma"]),
    "Port Jefferson": dict(
        color="#006EC7", peak_tph=6, cars=10,
        stops=["Port Jefferson", "Stony Brook", "St. James", "Smithtown",
               "Kings Park", "Northport", "Greenlawn", "Huntington",
               "Cold Spring Harbor", "Syosset", "Hicksville"]),
    "Oyster Bay": dict(
        color="#00AF3F", peak_tph=2, cars=6,
        stops=["Oyster Bay", "Locust Valley", "Glen Cove", "Glen Street",
               "Sea Cliff", "Glen Head", "Greenvale", "Roslyn", "Albertson",
               "East Williston", "Mineola"]),
}

# --------------------------------------------------------------- NYC zones
# (name, lat, lon, share of LI->NYC AM work trips, walk-served terminals)
NYC_ZONES = [
    ("Midtown West / Penn",   40.7520, -73.9910, 0.30, ["Penn Station"]),
    ("Midtown East / GCM",    40.7550, -73.9760, 0.25, ["Grand Central Madison"]),
    ("Lower Manhattan",       40.7080, -74.0100, 0.12, []),
    ("Downtown Brooklyn",     40.6910, -73.9840, 0.10, ["Atlantic Terminal"]),
    ("Long Island City",      40.7460, -73.9430, 0.09, ["Hunterspoint Ave", "Long Island City"]),
    ("Central Queens/Jamaica",40.7040, -73.8080, 0.14, ["Jamaica"]),
]

# ---------------------------------------------- population kernel anchors
# (name, lat, lon, approx 2020 population, county)
ANCHORS = [
    # ---- Nassau ----
    ("Hempstead Village",  40.706, -73.618,  59000, "Nassau"),
    ("Freeport",           40.657, -73.583,  43700, "Nassau"),
    ("Long Beach",         40.588, -73.658,  35000, "Nassau"),
    ("Valley Stream",      40.664, -73.708,  40400, "Nassau"),
    ("North Valley Stream",40.686, -73.709,  17000, "Nassau"),
    ("Rockville Centre",   40.658, -73.641,  25000, "Nassau"),
    ("Garden City",        40.727, -73.634,  23300, "Nassau"),
    ("Levittown",          40.725, -73.514,  52000, "Nassau"),
    ("Hicksville",         40.768, -73.525,  41500, "Nassau"),
    ("Massapequa",         40.681, -73.474,  22600, "Nassau"),
    ("Massapequa Park",    40.683, -73.452,  17900, "Nassau"),
    ("Plainview",          40.777, -73.468,  26200, "Nassau"),
    ("Syosset",            40.827, -73.502,  19700, "Nassau"),
    ("Oyster Bay Hamlet",  40.866, -73.532,   6700, "Nassau"),
    ("Glen Cove",          40.862, -73.633,  28400, "Nassau"),
    ("Port Washington",    40.826, -73.698,  17000, "Nassau"),
    ("Great Neck",         40.800, -73.729,  41000, "Nassau"),
    ("Manhasset",          40.798, -73.694,   8400, "Nassau"),
    ("Roslyn Heights",     40.799, -73.651,  12000, "Nassau"),
    ("New Hyde Park",      40.735, -73.688,  30000, "Nassau"),
    ("Floral Park",        40.722, -73.703,  16000, "Nassau"),
    ("Elmont",             40.702, -73.712,  35300, "Nassau"),
    ("Franklin Square",    40.700, -73.677,  30000, "Nassau"),
    ("West Hempstead",     40.704, -73.650,  20000, "Nassau"),
    ("Uniondale",          40.700, -73.593,  32000, "Nassau"),
    ("Roosevelt",          40.678, -73.588,  16300, "Nassau"),
    ("Baldwin",            40.657, -73.610,  24000, "Nassau"),
    ("Merrick",            40.662, -73.552,  22000, "Nassau"),
    ("Bellmore",           40.668, -73.527,  16900, "Nassau"),
    ("Wantagh",            40.683, -73.510,  18900, "Nassau"),
    ("Seaford",            40.667, -73.489,  15300, "Nassau"),
    ("East Meadow",        40.714, -73.558,  38000, "Nassau"),
    ("Westbury",           40.756, -73.588,  15400, "Nassau"),
    ("Mineola",            40.749, -73.641,  20800, "Nassau"),
    ("Carle Place",        40.751, -73.611,   5100, "Nassau"),
    ("Jericho",            40.792, -73.540,  14500, "Nassau"),
    ("Bethpage",           40.744, -73.483,  17000, "Nassau"),
    ("Farmingdale Village",40.732, -73.445,   8200, "Nassau"),
    ("Old Bethpage",       40.756, -73.454,   5500, "Nassau"),
    ("Woodmere",           40.632, -73.713,  20000, "Nassau"),
    ("Lynbrook",           40.658, -73.672,  20300, "Nassau"),
    ("Malverne",           40.675, -73.673,   8800, "Nassau"),
    ("Oceanside",          40.638, -73.640,  32100, "Nassau"),
    ("Island Park",        40.604, -73.653,   4900, "Nassau"),
    ("Williston Park",     40.760, -73.646,  15000, "Nassau"),
    ("Glen Head",          40.840, -73.630,  12000, "Nassau"),
    ("Locust Valley",      40.888, -73.593,  12000, "Nassau"),
    ("Brookville",         40.840, -73.560,   6000, "Nassau"),
    ("Hewlett",            40.639, -73.696,   7100, "Nassau"),
    ("Cedarhurst",         40.618, -73.729,  12000, "Nassau"),
    ("Inwood",             40.617, -73.746,   9800, "Nassau"),
    ("Stewart Manor",      40.713, -73.690,   8000, "Nassau"),
    ("Wheatley/Old Westbury", 40.788, -73.600, 6000, "Nassau"),
    # ---- Suffolk ----
    ("Huntington",         40.868, -73.426,  41000, "Suffolk"),
    ("Huntington Station", 40.853, -73.411,  33000, "Suffolk"),
    ("Melville",           40.793, -73.415,  19000, "Suffolk"),
    ("Dix Hills",          40.803, -73.348,  27000, "Suffolk"),
    ("Commack",            40.843, -73.293,  36000, "Suffolk"),
    ("Kings Park",         40.886, -73.245,  17700, "Suffolk"),
    ("Smithtown",          40.856, -73.201,  26000, "Suffolk"),
    ("Nesconset",          40.845, -73.152,  13700, "Suffolk"),
    ("Hauppauge",          40.825, -73.207,  21000, "Suffolk"),
    ("St. James",          40.878, -73.157,  13300, "Suffolk"),
    ("Stony Brook",        40.917, -73.132,  14000, "Suffolk"),
    ("East Setauket",      40.930, -73.101,  15600, "Suffolk"),
    ("Port Jefferson Sta", 40.925, -73.043,  25000, "Suffolk"),
    ("Coram",              40.874, -72.999,  40000, "Suffolk"),
    ("Selden",             40.870, -73.045,  22500, "Suffolk"),
    ("Centereach",         40.865, -73.083,  32000, "Suffolk"),
    ("Lake Grove",         40.858, -73.116,  11700, "Suffolk"),
    ("Ronkonkoma",         40.815, -73.121,  21000, "Suffolk"),
    ("Holbrook",           40.795, -73.077,  27000, "Suffolk"),
    ("Holtsville",         40.815, -73.045,  19700, "Suffolk"),
    ("Farmingville",       40.837, -73.032,  16500, "Suffolk"),
    ("Medford",            40.816, -72.986,  24100, "Suffolk"),
    ("Patchogue",          40.762, -73.015,  12000, "Suffolk"),
    ("East Patchogue",     40.769, -72.977,  22000, "Suffolk"),
    ("Bellport",           40.756, -72.939,  10000, "Suffolk"),
    ("Shirley",            40.799, -72.878,  28000, "Suffolk"),
    ("Mastic",             40.809, -72.849,  15500, "Suffolk"),
    ("Mastic Beach",       40.766, -72.852,  12000, "Suffolk"),
    ("Center Moriches",    40.800, -72.788,   8000, "Suffolk"),
    ("East Moriches",      40.809, -72.762,   5300, "Suffolk"),
    ("Eastport",           40.828, -72.734,   6000, "Suffolk"),
    ("Westhampton Beach",  40.804, -72.643,   5000, "Suffolk"),
    ("Hampton Bays",       40.869, -72.518,  14000, "Suffolk"),
    ("Southampton",        40.884, -72.390,  12000, "Suffolk"),
    ("Bridgehampton",      40.966, -72.293,   9000, "Suffolk"),
    ("East Hampton",       40.964, -72.184,  15000, "Suffolk"),
    ("Montauk",            41.036, -71.955,   4000, "Suffolk"),
    ("Riverhead",          40.917, -72.662,  35000, "Suffolk"),
    ("Mattituck",          40.993, -72.535,   4700, "Suffolk"),
    ("Southold",           41.065, -72.426,   5700, "Suffolk"),
    ("Greenport",          41.103, -72.362,   4200, "Suffolk"),
    ("Shelter Island",     41.070, -72.340,   3000, "Suffolk"),
    ("Babylon Village",    40.696, -73.326,  12600, "Suffolk"),
    ("West Babylon",       40.712, -73.354,  43000, "Suffolk"),
    ("North Babylon",      40.729, -73.323,  17900, "Suffolk"),
    ("Deer Park",          40.762, -73.329,  28000, "Suffolk"),
    ("Wyandanch",          40.749, -73.365,  12000, "Suffolk"),
    ("Wheatley Heights",   40.762, -73.371,   5500, "Suffolk"),
    ("Amityville",         40.679, -73.417,   9400, "Suffolk"),
    ("Copiague",           40.680, -73.398,  23000, "Suffolk"),
    ("Lindenhurst",        40.687, -73.373,  27000, "Suffolk"),
    ("Bay Shore",          40.725, -73.245,  27000, "Suffolk"),
    ("Brentwood",          40.781, -73.247,  62000, "Suffolk"),
    ("Central Islip",      40.790, -73.201,  36000, "Suffolk"),
    ("Islip",              40.730, -73.210,  19000, "Suffolk"),
    ("East Islip",         40.729, -73.185,  14000, "Suffolk"),
    ("Great River",        40.721, -73.155,   1500, "Suffolk"),
    ("Oakdale",            40.737, -73.139,   8000, "Suffolk"),
    ("Sayville",           40.735, -73.082,  17000, "Suffolk"),
    ("Bohemia",            40.769, -73.115,  10000, "Suffolk"),
    ("North Bay Shore",    40.749, -73.259,  21000, "Suffolk"),
    ("Northport",          40.900, -73.343,   7400, "Suffolk"),
    ("East Northport",     40.876, -73.325,  20000, "Suffolk"),
    ("Greenlawn",          40.868, -73.365,  14000, "Suffolk"),
    ("Cold Spring Harbor", 40.858, -73.454,   5000, "Suffolk"),
    ("South Huntington",   40.822, -73.400,  10000, "Suffolk"),
    ("Elwood",             40.846, -73.335,  12000, "Suffolk"),
    ("Islandia",           40.803, -73.166,   6000, "Suffolk"),
    ("Yaphank",            40.836, -72.918,   6000, "Suffolk"),
    ("Manorville",         40.860, -72.798,  15000, "Suffolk"),
    ("Calverton",          40.909, -72.760,   6500, "Suffolk"),
    ("Wading River",       40.949, -72.840,   8000, "Suffolk"),
    ("Rocky Point",        40.947, -72.936,  14000, "Suffolk"),
    ("Miller Place",       40.943, -72.988,  12700, "Suffolk"),
    ("Mount Sinai",        40.933, -73.019,  12000, "Suffolk"),
    ("Sound Beach",        40.958, -72.968,   7600, "Suffolk"),
    ("Ridge",              40.899, -72.892,  13700, "Suffolk"),
    ("Middle Island",      40.885, -72.947,  10500, "Suffolk"),
]

# Control totals (2020 Census county populations)
COUNTY_POP = {"Nassau": 1395774, "Suffolk": 1525920}
# 2020 tract counts
COUNTY_TRACTS = {"Nassau": 285, "Suffolk": 373}
COUNTY_FIPS = {"Nassau": "36059", "Suffolk": "36103"}

# Stations that historically carry outsized park-and-ride catchments.
MAJOR_STATIONS = {
    "Hicksville": 2.4, "Ronkonkoma": 2.3, "Mineola": 1.8, "Huntington": 2.0,
    "Babylon": 1.7, "Freeport": 1.3, "Great Neck": 1.4, "Port Washington": 1.4,
    "Massapequa": 1.3, "Rockville Centre": 1.3, "Valley Stream": 1.3,
    "Bay Shore": 1.3, "Lindenhurst": 1.3, "Bethpage": 1.2, "Farmingdale": 1.4,
    "Syosset": 1.4, "Merrick": 1.2, "Wantagh": 1.2, "Baldwin": 1.1,
    "Westbury": 1.2, "Deer Park": 1.3, "Stony Brook": 1.2, "Smithtown": 1.3,
    "Patchogue": 1.2, "Northport": 1.1, "Long Beach": 1.2, "Hempstead": 1.2,
    "Garden City": 1.1, "Amityville": 1.1, "Copiague": 1.1, "Seaford": 1.1,
    "Bellmore": 1.1, "Central Islip": 1.2, "Brentwood": 1.1, "Riverhead": 1.1,
}

# Share of AM-peak westbound Nassau+Suffolk boardings, for the stations whose
# relative standing is well known. Remaining stations are filled from the
# catchment model and the residual share.
ANCHOR_SHARES = {
    "Hicksville": .062, "Ronkonkoma": .043, "Mineola": .034, "Huntington": .033,
    "Babylon": .030, "Freeport": .024, "Massapequa": .023, "Rockville Centre": .022,
    "Great Neck": .021, "Port Washington": .020, "Bay Shore": .019,
    "Lindenhurst": .018, "Farmingdale": .018, "Bethpage": .017, "Syosset": .017,
    "Merrick": .016, "Wantagh": .015, "Baldwin": .014, "Westbury": .014,
    "Valley Stream": .014, "Hempstead": .013, "Bellmore": .013, "Long Beach": .013,
    "Seaford": .012, "Deer Park": .012, "Stony Brook": .011, "Smithtown": .011,
    "Patchogue": .010, "Amityville": .010, "Copiague": .010, "Northport": .009,
    "Garden City": .009, "Lynbrook": .009, "Oceanside": .009, "Central Islip": .009,
    "Brentwood": .008, "Manhasset": .008, "Cedarhurst": .007, "Woodmere": .007,
    "Sayville": .007,
}


# ===========================================================================
#  Published control totals -- MTA LIRR 2025 Annual Ridership Report
#  https://www.mta.info/document/205596   (retrieved 2026-08)
#  These replace the figures previously assumed in this file.  The build has
#  no network access, so they are transcribed here rather than fetched.
# ===========================================================================
MTA_2025 = dict(
    annual_customers      = 82_000_000,   # 90.0% of 2019
    weekday_avg           = 256_000,      # 2024: 245,000;  2019 base: 284,110
    weekday_offpeak       = 111_740,      # 2019 base: 98,340
    commutation_annual    = 31_000_000,   # non-commutation 51.0M
    saturday_avg          = 137_000,
    sunday_avg            = 112_000,
    source                = "MTA LIRR 2025 Annual Ridership Report",
)
# Peak weekday = weekday - off-peak, covering AM and PM together.
AM_PEAK_BOARDINGS = int(round(
    (MTA_2025["weekday_avg"] - MTA_2025["weekday_offpeak"]) / 2))   # ~72,130

# "GCM stabilized at 40% of AM Peak travel to Manhattan, while 60% used Penn
# Station" -- 2025 report, 2026 Outlook.  Constrains terminal choice among
# Manhattan-bound AM peak rail riders.
AM_PEAK_MANHATTAN_TERMINAL = {"Penn Station": 0.60, "Grand Central Madison": 0.40}

# 2025 branch ridership, annual customers.  The report names the top three
# explicitly; the remaining branches are not broken out in the public summary,
# so only these three are treated as hard anchors.
BRANCH_2025 = {
    "Port Jefferson": 15_100_000,   # includes Huntington branch customers
    "Babylon":        13_460_000,
    "City Terminal":  13_430_000,   # +12.8% YoY, largest increase
}
# City Zone trips originate at Queens/Brooklyn stations, which this instance
# treats as destinations only, so they are outside the Long Island -> NYC
# market modelled here.
LI_ORIGIN_ANNUAL = MTA_2025["annual_customers"] - BRANCH_2025["City Terminal"]

# NOT AVAILABLE.  Station-level weekday counts exist in the 2012-2014 LIRR
# Origin and Destination Report, Volume I, sheets 197-199 ("WEEKDAY East/West
# Total By Station"), MTA, 23 Aug 2016.  web.mta.info disallows automated
# access and the Wayback copy is blocked, so the within-branch station split
# in ANCHOR_SHARES below remains synthesized.  Anyone with that PDF can
# replace ANCHOR_SHARES wholesale and the rest of the build is unchanged.


# ===========================================================================
#  Station-level AM peak westbound boardings, from the MTA 2012-2014 LIRR
#  Origin and Destination Report, Volume I (23 Aug 2016), sheets 197-199,
#  "2012-2014 LIRR OD COUNTS: WEEKDAY, East/West Total By Station in
#  Numerical Order", WESTBOUND / AM Peak / On column.
#
#  These are observed counts, not estimates.  The 125 station rows sum to the
#  report's own TOTAL of 111,328; the 97 Long Island stations below account
#  for 87,101 of that, the remainder being Queens, Brooklyn and the
#  Manhattan terminals.
#
#  The survey predates Grand Central Madison (2023) and the post-COVID
#  ridership shift, so the LEVEL is stale; the build rescales the total to the
#  2025 control derived from MTA_2025 and keeps only the SPATIAL DISTRIBUTION.
# ===========================================================================
AM_PEAK_BOARDINGS_2012_2014 = {
    "Hicksville": 5854,
    "Ronkonkoma": 5510,
    "Huntington": 3725,
    "Mineola": 3039,
    "Great Neck": 2909,
    "Babylon": 2903,
    "Merrick": 2718,
    "Port Washington": 2664,
    "Bellmore": 2628,
    "Baldwin": 2428,
    "Rockville Centre": 2409,
    "Valley Stream": 2189,
    "Syosset": 2139,
    "Deer Park": 2108,
    "Wantagh": 2089,
    "Manhasset": 1844,
    "Massapequa": 1737,
    "Long Beach": 1579,
    "Seaford": 1433,
    "Wyandanch": 1427,
    "Massapequa Park": 1386,
    "Freeport": 1368,
    "Bethpage": 1338,
    "New Hyde Park": 1333,
    "Lynbrook": 1286,
    "Central Islip": 1235,
    "Cold Spring Harbor": 1230,
    "Westbury": 1149,
    "Floral Park": 1088,
    "Lindenhurst": 1057,
    "Oceanside": 1014,
    "Farmingdale": 956,
    "Northport": 954,
    "Copiague": 819,
    "Brentwood": 816,
    "Island Park": 717,
    "Stewart Manor": 676,
    "Amityville": 662,
    "Hewlett": 661,
    "Merillon Avenue": 636,
    "Kings Park": 633,
    "Woodmere": 632,
    "Hempstead": 515,
    "Nassau Boulevard": 513,
    "Plandome": 486,
    "Bay Shore": 474,
    "Bellerose": 469,
    "Gibson": 466,
    "Cedarhurst": 439,
    "Port Jefferson": 434,
    "Stony Brook": 408,
    "Garden City": 408,
    "Greenlawn": 406,
    "East Rockaway": 394,
    "Centre Avenue": 380,
    "Sayville": 379,
    "Islip": 379,
    "Malverne": 362,
    "Roslyn": 352,
    "Smithtown": 307,
    "St. James": 301,
    "Carle Place": 299,
    "Patchogue": 290,
    "Country Life Press": 287,
    "Glen Head": 261,
    "Inwood": 258,
    "East Williston": 252,
    "Lakeview": 244,
    "Westwood": 243,
    "Lawrence": 215,
    "Locust Valley": 201,
    "Mastic-Shirley": 200,
    "Albertson": 192,
    "Sea Cliff": 190,
    "Glen Street": 165,
    "Hempstead Gardens": 157,
    "Glen Cove": 148,
    "Oakdale": 139,
    "West Hempstead": 115,
    "Great River": 104,
    "Greenvale": 66,
    "Speonk": 49,
    "Oyster Bay": 47,
    "Hampton Bays": 23,
    "Riverhead": 20,
    "Bellport": 18,
    "East Hampton": 15,
    "Bridgehampton": 12,
    "Medford": 10,
    "Mattituck": 8,
    "Southampton": 7,
    "Amagansett": 5,
    "Montauk": 5,
    "Southold": 4,
    "Westhampton": 2,
    "Pinelawn": 0,
    "Greenport": 0,
}
