"""
material_profile.py
===============================================================
140 GHz Radio Material Library for IEEE 6G Digital Twin

Compatible with:
    - Sionna RT 2.0.1
    - TensorFlow 2.21

Reference:
    ITU-R P.2040
    ITU-R P.676
    3GPP TR 38.901
===============================================================
"""

from sionna.rt import RadioMaterial

# ============================================================
# MATERIAL DATABASE (140 GHz)
# ============================================================

# (Relative Permittivity, Conductivity [S/m])

MATERIAL_DATABASE = {

    # --------------------------------------------------------
    # Building Materials
    # --------------------------------------------------------
    "Concrete_140GHz": (5.31, 0.75),

    "Brick_140GHz": (3.75, 0.38),

    "Marble_140GHz": (7.10, 0.18),

    "Granite_140GHz": (5.90, 0.22),

    "Glass_140GHz": (6.27, 0.55),

    "Wood_140GHz": (1.99, 0.05),

    "Plasterboard_140GHz": (2.94, 0.02),

    # --------------------------------------------------------
    # Ground Materials
    # --------------------------------------------------------
    "Asphalt_140GHz": (3.30, 0.18),

    "Soil_140GHz": (4.20, 0.12),

    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------
    "Vegetation_140GHz": (2.80, 0.35),

    # --------------------------------------------------------
    # Metallic Surfaces
    # --------------------------------------------------------
    "Metal_140GHz": (1.00, 5.8e7),
}

# ============================================================
# XML / ITU MATERIAL NAME MAPPING
# ============================================================

ITU_TO_140GHZ = {

    # Building
    "concrete": "Concrete_140GHz",
    "itu_concrete": "Concrete_140GHz",

    "brick": "Brick_140GHz",
    "itu_brick": "Brick_140GHz",

    "marble": "Marble_140GHz",

    "granite": "Granite_140GHz",

    "glass": "Glass_140GHz",
    "itu_glass": "Glass_140GHz",

    "wood": "Wood_140GHz",
    "itu_wood": "Wood_140GHz",

    "plasterboard": "Plasterboard_140GHz",

    # Ground
    "asphalt": "Asphalt_140GHz",
    "road": "Asphalt_140GHz",
    "street": "Asphalt_140GHz",
    "ground": "Asphalt_140GHz",
    "floor": "Asphalt_140GHz",

    # Nature
    "tree": "Vegetation_140GHz",
    "vegetation": "Vegetation_140GHz",
    "plant": "Vegetation_140GHz",
    "leaf": "Vegetation_140GHz",
    "foliage": "Vegetation_140GHz",

    # Metal
    "metal": "Metal_140GHz",
    "itu_metal": "Metal_140GHz",
    "steel": "Metal_140GHz",
    "iron": "Metal_140GHz",
}

# ============================================================
# OBJECT KEYWORD FALLBACK MAPPING
# ============================================================

OBJECT_KEYWORDS = {

    "Vegetation_140GHz": [
        "tree",
        "vegetation",
        "plant",
        "leaf",
        "foliage",
        "grass",
        "bush",
    ],

    "Glass_140GHz": [
        "glass",
        "window",
        "windscreen",
    ],

    "Metal_140GHz": [
        "metal",
        "car",
        "vehicle",
        "pole",
        "traffic",
        "lamp",
        "sign",
        "streetlight",
        "rail",
    ],

    "Wood_140GHz": [
        "wood",
        "bench",
        "door",
        "table",
    ],

    "Asphalt_140GHz": [
        "road",
        "street",
        "ground",
        "floor",
        "pavement",
        "crosswalk",
        "asphalt",
    ],

    "Concrete_140GHz": [
        "wall",
        "building",
        "bridge",
        "roof",
        "stairs",
        "column",
        "concrete",
    ],
}

# ============================================================
# REGISTER CUSTOM MATERIALS
# ============================================================

def setup_140ghz_materials(scene):
    """
    Register all custom materials into Sionna scene.

    Returns
    -------
    dict
        Dictionary of RadioMaterial objects.
    """

    materials = {}

    for name, (eps_r, sigma) in MATERIAL_DATABASE.items():

        # Material already exists
        if name in scene._radio_materials:
            materials[name] = scene._radio_materials[name]
            continue

        material = RadioMaterial(
            name=name,
            relative_permittivity=eps_r,
            conductivity=sigma,
        )

        # Sionna RT 2.0.1 API
        scene.add(material)

        materials[name] = material

    return materials

# ============================================================
# MATERIAL CLASSIFIER
# ============================================================

def classify_material(material_name, object_name=""):
    """
    Determine the appropriate 140 GHz material.

    Parameters
    ----------
    material_name : str
        Original material name.

    object_name : str
        Mesh/object name.

    Returns
    -------
    str
        Custom material name.
    """

    mat = str(material_name).lower()
    obj = str(object_name).lower()

    # --------------------------------------------------------
    # First priority: ITU material mapping
    # --------------------------------------------------------
    for key, value in ITU_TO_140GHZ.items():
        if key in mat:
            return value

    # --------------------------------------------------------
    # Second priority: Object keywords
    # --------------------------------------------------------
    for material, keywords in OBJECT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in obj:
                return material

    # --------------------------------------------------------
    # Default fallback
    # --------------------------------------------------------
    return "Concrete_140GHz"

# ============================================================
# REPORT FUNCTION
# ============================================================

def print_material_statistics(counter):
    """
    Pretty print material statistics.
    """

    print("-" * 60)
    print("140 GHz MATERIAL DISTRIBUTION")
    print("-" * 60)

    total = sum(counter.values())

    for name in sorted(counter.keys()):

        value = counter[name]

        if value == 0:
            continue

        ratio = value / total * 100

        print(f"{name:<24}: {value:>5} ({ratio:>5.1f}%)")

    print("-" * 60)
    print(f"Total Objects : {total}")
    print("-" * 60)