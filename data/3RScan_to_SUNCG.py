import json

def convert_dataset(input_data, rel_types):
    rooms = []
    
    for scan in input_data.get("scans", []):
        # Use the scan id as room_model_id and generate a room_scene_id (e.g., by appending the split number)
        room_model_id = scan.get("scan")
        split = scan.get("split")
        room_scene_id = f"{room_model_id}_{split}"
        # Set a default room_type (adjust if needed)
        room_type = "3RScan"

        # Create node_list: for each object, the key is "objectType_objectID"
        node_list = {}
        objects = scan.get("objects", {})
        all_obj_id = set()
            
        for obj_id, obj_name in objects.items():
            # Replace spaces with underscores in the object name to form the key
            key = f"{obj_name.replace(' ', '_')}_{obj_id}"
                
            node_list[key] = {
                "type": obj_name.replace(' ', '_'),
                "self_info": {
                    # Default identity rotation (3x3 flattened), zero translation and dimensions
                    # "rotation": [1.0, 0.0, 0.0,
                    #              0.0, 1.0, 0.0,
                    #              0.0, 0.0, 1.0],
                    # "translation": [0.0, 0.0, 0.0],
                    # Using the unique object id as the node_model_id
                    # "node_model_id": obj_id,
                    # "dim": [0.0, 0.0, 0.0]
                },
            }
            # Initialize each relationship type key as an empty list
            for rel in rel_types:
                node_list[key][rel] = []
            

        # Use relationships to populate each node's relationship list.
        for rel in scan.get("relationships", []):
            # Expecting each relationship to have at least 4 elements:
            # [source_object_id, target_object_id, <other_info>, <relationship_type>]
            if len(rel) < 4:
                continue

            source_id = str(rel[0])
            target_id = str(rel[1])
            all_obj_id.add(source_id)
            all_obj_id.add(target_id)
            rel_type = rel[3].replace(" ", "_")
            if rel_type not in rel_types:
                continue
            # Only proceed if both source and target exist in the objects dictionary
            if source_id in objects and target_id in objects:
                source_key = f"{objects[source_id].replace(' ', '_')}_{source_id}"
                target_key = f"{objects[target_id].replace(' ', '_')}_{target_id}"
                # Check if the relationship type exists in the source node
                if rel_type in node_list[source_key]:
                    if target_key not in node_list[source_key][rel_type]:
                        node_list[source_key][rel_type].append(target_key)
                # Uncomment the following block if you want to add the relationship bidirectionally:
                """
                if rel_type in node_list[target_key]:
                    if source_key not in node_list[target_key][rel_type]:
                        node_list[target_key][rel_type].append(source_key)
                """
        for obj_id, obj_name in objects.items():
            if obj_id not in all_obj_id:
                node_list.pop(f"{obj_name.replace(' ', '_')}_{obj_id}")

        # Build the final room dictionary for this scan
        room = {
            "room_model_id": room_model_id,
            "used_nodes": {},
            "room_scene_id": room_scene_id,
            "room_type": room_type,
            "node_list": node_list
        }
        rooms.append(room)

    return rooms

def main():
    # Read the original JSON dataset from "input.json"
    with open("/cluster/project/cvg/students/shangwu/graphto3d_mani/GT/relationships_train_filtered.json", "r") as infile:
        input_data = json.load(infile)
    
    rels = []
    with open("/cluster/project/cvg/students/shangwu/3DIndoor-SceneGraphNet/data/relationships.txt", "r") as relfile:
        # Iterate over each line in the file
        for line in relfile:
            # Remove leading/trailing whitespace
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            # Replace spaces with underscores and add to the list
            rel = line.replace(" ", "_")
            rels.append(rel)
    
    # Convert the dataset into the desired format
    converted_data = convert_dataset(input_data, rels)

    # Write the converted data to "output.json"
    with open("/cluster/project/cvg/students/shangwu/3DIndoor-SceneGraphNet/data/3RScan_data.json", "w") as outfile:
        json.dump(converted_data, outfile, indent=2)

if __name__ == "__main__":
    main()
