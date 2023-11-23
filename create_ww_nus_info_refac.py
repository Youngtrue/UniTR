import os
import mmcv
import numpy as np
import json
import pdb
from typing import List, Dict

def get_calib(datacali: str) -> Dict:
    calib_dict = dict()
    Ks = dict()
    Es = dict()

    for camera_type in ["frontleft", "frontmid", "frontright", "rearleft", "rearright", "topleft"]:
        calib_path = os.path.join(datacali, f"{camera_type}.json")
        with open(calib_path, 'r') as f:
            calib = json.load(f)
            calib["intrinsic"] = np.array(calib["intrinsic"]).reshape(3, 3)
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = calib["intrinsic"]
            Ks[camera_type] = camera_intrinsics.astype(np.float32)

            calib["extrinsic"] = np.array(calib["extrinsic"]).reshape(4, 4)
            Es[camera_type] = calib["extrinsic"].astype(np.float32)

            K = np.vstack([calib["intrinsic"], [0, 0, 0]])
            K = np.hstack([K, [[0], [0], [0], [1]]])
            lidar2img = np.dot(K, calib["extrinsic"])
            calib_dict[camera_type] = lidar2img.astype(np.float32)

    return calib_dict, Ks, Es

def in_range_bev(bboxes_array,xy_range):
    """Check whether the boxes are in the given range.
    lidar_box3d中仅仅按照中心x,y是否在range内,也按此操作过滤
    Args:
        xy_range (list | torch.Tensor): the range of box
            (x_min, y_min, x_max, y_max)

    Note:
        The original implementation of SECOND checks whether boxes in
        a range by checking whether the points are in a convex
        polygon, we reduce the burden for simpler cases.

    Returns:
        torch.Tensor: Whether each box is inside the reference range.
    """
    # pdb.set_trace()
    if bboxes_array.shape[0] == 0:
        return False,False
    in_range_flags = (
        (bboxes_array[:, 0] > xy_range[0])
        & (bboxes_array[:, 1] > xy_range[1])
        & (bboxes_array[:, 0] < xy_range[2])
        & (bboxes_array[:, 1] < xy_range[3])
    )
    return True,in_range_flags

def load_annotation(ann_json_path: str, xy_range: List[float]) -> List[Dict]:
    # Load annotation from JSON file
    with open(ann_json_path, 'r') as f:
        ann_json = json.load(f)

    bboxes = []
    labels = []
    num_lidar_pts = []

    for meta in ann_json["dataList"]:
        # Extract relevant information from the annotation
        box_xyz = [meta["center"]["x"], meta["center"]["y"], meta["center"]["z"]]
        box_wlh = [meta["dimensions"]["width"], meta["dimensions"]["length"], meta["dimensions"]["height"]]
        yaw = [-meta["rotation"]["z"] - np.pi / 2]
        label = meta["label"]

        if label == "self_tray":
            continue  # Skip self_tray label

        labels.append(label)
        bboxes.append(box_xyz + box_wlh + yaw + [0, 0])
        num_lidar_pts.append(meta["point"])

    # Convert lists to numpy arrays
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    num_lidar_pts = np.array(num_lidar_pts)

    # Filter by xy range
    filter_success, filter_flag = in_range_bev(bboxes, xy_range)
    if filter_success:
        bboxes = bboxes[filter_flag]
        labels = labels[filter_flag]

    return bboxes, labels, num_lidar_pts

def create_info_dict(lidar_path: str, ann_json_path: str, scene_datapath_raw: str,
                     calib_dict: Dict, Ks: Dict, Es: Dict) -> Dict:
    timestamp = ann_json_path.split("/")[-1].split(".")[0]

    cams_dict = dict()
    for k, camera_type in enumerate(["frontleft", "frontmid", "frontright", "rearleft", "rearright", "topleft"]):
        img_path = os.path.join(scene_datapath_raw, "camera", camera_type, f"{timestamp}.jpg")

        cams_dict[camera_type] = {
            "data_path": img_path,
            "type": camera_type,
            "camera_intrinsics": Ks[camera_type],
            "lidar2camera": Es[camera_type],
            "camera2lidar": np.linalg.inv(Es[camera_type]).astype(np.float32),
            "camera2ego": np.linalg.inv(Es[camera_type]).astype(np.float32),
            "lidar2image": calib_dict[camera_type],
            "timestamp": timestamp
        }

    token = ann_json_path.replace("/", "_")[1:]
    info = {
        "lidar_path": lidar_path,
        "token": token,
        "sweeps": "ww do not have,seq pre and next frame",
        "cams": cams_dict.copy(),
        "lidar2ego": np.eye(4).astype(np.float32),
        "ego2global": np.eye(4).astype(np.float32),
        "timestamp": timestamp,
        "location": "taiguo_ww",
        "gt_boxes": bboxes,
        "gt_names": labels,
        "num_lidar_pts": num_lidar_pts
    }

    return info

if __name__ == "__main__":
    # ...
    

    for scene_datepath in scenes:
        # ...

        # get calib
        datacali_path = os.path.join(scene_datapath_raw, "calib", "camera")
        calib_dict, Ks, Es = get_calib(datacali=datacali_path)

        ann_paths = glob.glob(os.path.join(scene_datepath, "lidar", "*json"))
        ann_paths.sort()

        for frames_cur_scene, ann_json_path in enumerate(ann_paths):
            # ...

            bboxes, labels, num_lidar_pts = load_annotation(ann_json_path, xy_range)

            if bboxes.shape[0] == 0:
                print("No targets in the BEV range of this frame:", xy_range)
                continue

            lidar_path = ann_json_path.replace(data_ann_root_train, dataname)
            lidar_path = lidar_path.replace("json", "pcd")
            use_dim7 = False
            if use_dim7:
                lidar_path = lidar_path.replace("0622_data_4528frames_147scenes", "0622_data_4528frames_147scenes_withRGB")
                lidar_path = lidar_path.replace("pcd", "npy")

            info = create_info_dict(lidar_path, ann_json_path, scene_datapath_raw, calib_dict, Ks, Es)

            if scene_datename in val_scene_list:
                data_val["infos"].append(info)
                frames_val += 1
                for label in labels:
                    object_classes_dict_val[label] += 1
            else:
                data_train["infos"].append(info)
                frames_train += 1
                for label in labels:
                    object_classes_dict_train[label] += 1

            frames += 1

    # ...

    mmcv.dump(data_train, os.path.join(root, train_pkl_save_name))
    mmcv.dump(data_val, os.path.join(root, val_pkl_save_name))
    mmcv.dump(data_val, os.path.join(root, test_pkl_save_name))

    # ... (print statements)
