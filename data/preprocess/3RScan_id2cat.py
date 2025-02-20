import json

cats = set()
rels = []
with open("/cluster/project/cvg/students/shangwu/graphto3d_mani/GT/relationships.txt", "r") as relfile:
    # Iterate over each line in the file
    for line in relfile:
        # Remove leading/trailing whitespace
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        # Replace spaces with underscores and add to the list
        rel = line.replace(" ", "_")
        rels.append(rel)

with open("/cluster/project/cvg/students/shangwu/graphto3d_mani/GT/relationships_train_filtered.json", "r") as infile:
    input_data = json.load(infile)
    for scan in input_data.get("scans", []):
        objects = scan.get("objects", {})
        # Use relationships to populate the category set.
        for rel in scan.get("relationships", []):
            # Expecting each relationship to have at least 4 elements:
            # [source_object_id, target_object_id, <other_info>, <relationship_type>]
            if len(rel) < 4:
                continue
            rel_type = rel[3].replace(" ", "_")
            if rel_type not in rels:
                continue
            source_id = str(rel[0])
            target_id = str(rel[1])
            # Optional: Check that the keys exist in objects to avoid KeyError
            if source_id in objects and target_id in objects:
                cats.add(objects[source_id])
                cats.add(objects[target_id])
                
cat_dict = {}
for idx, cat in enumerate(cats):
    cat_dict[idx] = cat.replace(" ", "_")

# Open the output file in write mode ('w') to dump the JSON
with open("/cluster/project/cvg/students/shangwu/3DIndoor-SceneGraphNet/data/preprocess/TRAIN_id2cat_3RScan.json", "w") as jfile:
    json.dump(cat_dict, jfile, indent=2)
