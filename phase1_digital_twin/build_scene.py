"""
build_scene.py
===============================================================
IEEE Phase 1 – Digital Twin Initialization

Compatible with:
    Sionna RT 2.0.1
    TensorFlow 2.21

Functions
---------
1. Load local Etoile XML.
2. Register 140 GHz materials.
3. Replace ALL ITU materials before frequency update.
4. Map every mesh to Sub-THz material.
5. Produce IEEE experimental setup report.
===============================================================
"""

import os
import xml.etree.ElementTree as ET
from collections import Counter

from sionna.rt import load_scene

from phase1_digital_twin import config
from phase1_digital_twin.material_profile import (
    setup_140ghz_materials,
    classify_material,
    print_material_statistics
)

# ===============================================================
# XML ANALYSIS
# ===============================================================

def inspect_xml(scene_file):
    """
    Read XML directly to count meshes/material references.
    This is used only for reporting.
    """

    if not os.path.exists(scene_file):
        return 0, Counter()

    tree = ET.parse(scene_file)
    root = tree.getroot()

    shapes = root.findall("shape")

    material_counter = Counter()

    for shape in shapes:

        ref = shape.find("ref")

        if ref is not None:
            name = ref.attrib.get("id", "").lower()
            material_counter[name] += 1

    return len(shapes), material_counter


# ===============================================================
# MATERIAL REPLACEMENT
# ===============================================================

def replace_material_registry(scene, materials, verbose=True):
    """
    Replace unsupported ITU materials BEFORE frequency update.
    """

    registry = scene._radio_materials
    objects = scene._scene_objects

    replaced_registry = 0
    replaced_objects = 0

    # ----------------------------------------------------------
    # Replace registry entries
    # ----------------------------------------------------------

    registry_names = list(registry.keys())

    for name in registry_names:

        new_name = classify_material(name)

        if new_name in materials:

            registry[name] = materials[new_name]
            replaced_registry += 1

    # ----------------------------------------------------------
    # Replace object materials
    # ----------------------------------------------------------

    for obj_name, obj in objects.items():

        rm = obj.radio_material

        if rm is None:

            obj.radio_material = materials["Concrete_140GHz"]
            replaced_objects += 1
            continue

        mat_name = rm if isinstance(rm, str) else rm.name

        new_name = classify_material(mat_name, obj_name)

        obj.radio_material = materials[new_name]
        replaced_objects += 1

    if verbose:

        print(f"      -> Unsupported registry materials : {replaced_registry}")
        print(f"      -> Unsupported object materials   : {replaced_objects}")

    return replaced_registry, replaced_objects


# ===============================================================
# MATERIAL COVERAGE REPORT
# ===============================================================

def generate_material_report(scene):
    """
    Count final mapped materials.
    """

    counter = Counter()

    for obj in scene._scene_objects.values():

        rm = obj.radio_material

        if rm is None:
            counter["None"] += 1
        else:
            counter[rm.name] += 1

    return counter


# ===============================================================
# VERIFY MATERIALS
# ===============================================================

def verify_material_coverage(scene):
    """
    Ensure every object owns a valid RadioMaterial.
    """

    uncovered = []

    for name, obj in scene._scene_objects.items():

        if obj.radio_material is None:
            uncovered.append(name)

    if uncovered:

        raise RuntimeError(
            f"{len(uncovered)} objects have no RadioMaterial:\n"
            + "\n".join(uncovered[:10])
        )

    return True


# ===============================================================
# MAIN INITIALIZATION
# ===============================================================

def init_and_map_scene(verbose=True):
    """
    IEEE Digital Twin initialization.

    Returns
    -------
    Scene
    """

    scene_file = config.SCENE_NAME

    if verbose:
        print(f"[1/3] Loading Digital Twin: '{scene_file}'...")

    scene = load_scene(scene_file)

    # ----------------------------------------------------------
    # XML statistics
    # ----------------------------------------------------------

    xml_shapes, xml_materials = inspect_xml(scene_file)

    scene_objects = scene._scene_objects
    scene_materials = scene._radio_materials

    if verbose:

        print(f"      -> Scene objects loaded : {len(scene_objects)}")
        print(f"      -> Scene materials       : {len(scene_materials)}")
        print(f"      -> XML shapes            : {xml_shapes}")

        if xml_shapes > len(scene_objects):

            print("      -> WARNING: XML contains merged meshes.")
            print("         Ray tracing remains valid but mesh count is reduced.")

    # ----------------------------------------------------------
    # Register Sub-THz materials
    # ----------------------------------------------------------

    materials = setup_140ghz_materials(scene)

    # ----------------------------------------------------------
    # Replace unsupported ITU materials
    # ----------------------------------------------------------

    replace_material_registry(scene, materials, verbose)

    # ----------------------------------------------------------
    # Synthetic Array
    # ----------------------------------------------------------

    scene.synthetic_array = config.SYNTHETIC_ARRAY

    # ----------------------------------------------------------
    # Safe Frequency Update
    # ----------------------------------------------------------

    scene.frequency = config.CARRIER_FREQ

    if verbose:

        freq_ghz = config.CARRIER_FREQ / 1e9
        if verbose:
         print(f"[2/3] Carrier Frequency set to: {freq_ghz:.1f} GHz")

    # ----------------------------------------------------------
    # Final Mapping Pass
    # ----------------------------------------------------------

    counter = Counter()

    for obj_name, obj in scene_objects.items():

        rm = obj.radio_material

        if rm is None:

            obj.radio_material = materials["Concrete_140GHz"]
            counter["Concrete_140GHz"] += 1
            continue

        mapped = classify_material(rm.name, obj_name)

        obj.radio_material = materials[mapped]

        counter[mapped] += 1

    # ----------------------------------------------------------
    # Verification
    # ----------------------------------------------------------

    verify_material_coverage(scene)

    if verbose:

        print("[3/3] Material Mapping Completed.\n")

        print("=" * 60)
        print("EXPERIMENTAL SETUP REPORT")
        print("=" * 60)

        print(f"Loaded Scene Objects      : {len(scene_objects)}")
        print(f"XML Shape Count           : {xml_shapes}")

        coverage = sum(counter.values())
        print(f"Mapped Objects            : {coverage}")

        print("=" * 60)

        print_material_statistics(counter)

        print("XML Material References")

        for name, value in xml_materials.items():
            print(f"  {name:<20} : {value}")

        print("=" * 60)

    return scene