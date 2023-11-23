import numpy as np
import cv2
import sys
import mmcv
import os.path as osp
import json
import glob
"""
create datsaet ww : 
1. 按照annotation 遍历所有数据，添加lidar和6个camera的paths 和 gt_bboxes ， names，
2. 以下脚本使用的外参calib: data/ww/calib，
ww_dataset
info = {
        "lidar_path": lidar_path,
        "token": ann_json_path,
        "sweeps": "ww do not have,seq pre and next frame",
        "cams": cams_dict.copy(),
        "lidar2ego": np.eye(4).astype(np.float32),
        "ego2global":  np.eye(4).astype(np.float32),# 先留着,训练不使用
        "timestamp": timestamp,
        "location": "taiguo_ww", # 地图
        "gt_boxes": bboxes,
        "gt_names":labels,
        "num_lidar_pts":np.ones(labels.shape[0]),
        }

nuscenes: 
info = {
    "lidar_path": lidar_path,
    "token": sample["token"],
    "sweeps": [],
    "cams": dict(),
    "lidar2ego_translation": cs_record["translation"],
    "lidar2ego_rotation": cs_record["rotation"],
    "timestamp": sample["timestamp"],
    "location": "taiguo_ww", # 地图
    "gt_boxes": "",
    "gt_names":""
    
    "ego2global_translation": pose_record["translation"],
    "ego2global_rotation": pose_record["rotation"],
    info["gt_boxes"] = gt_boxes
    info["gt_names"] = names
    info["gt_velocity"] = velocity.reshape(-1, 2)
    info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
    info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
    info["valid_flag"] = valid_flag
    }

"""

def get_calib(datacali="./data/ww/calib/camera"):
    calib_dict = dict()
    Ks = dict()
    Es = dict()
    for camera_type in ["frontleft","frontmid","frontright","rearleft","rearright","topleft","topright"]:
        #get calib
        calib_path = osp.join(datacali,camera_type+".json")
        # print(calib_path)
        with open(calib_path, 'r') as f:
            calib = json.load(f)
            calib["intrinsic"] = np.array(calib["intrinsic"]).reshape(3,3)
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = calib["intrinsic"]
            Ks[camera_type]=camera_intrinsics.astype(np.float32)
            
            calib["extrinsic"] = np.array(calib["extrinsic"]).reshape(4,4)
            Es[camera_type]=calib["extrinsic"].astype(np.float32)

            # calib_dict[camera_type]=calib
            # pdb.set_trace()
            K = np.vstack([calib["intrinsic"],[0,0,0]])
            K = np.hstack([K,[[0],[0],[0],[1]]])
            lidar2img = np.dot(K,calib["extrinsic"])
            calib_dict[camera_type]=lidar2img.astype(np.float32)
            
    return calib_dict,Ks,Es


def showImg(img,wati_time=0,win_name="r"):
    # print(img.shape)
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.imshow(win_name,img)
    key = cv2.waitKey(wati_time) & 0xFF
    if key == ord('q'):
        sys.exit()
    return key


def in_range_bev(bboxes_array,xy_range):
    """Check whether the boxes are in the given range.
    lidar_box3d中仅仅按照中心x，y是否在range内，也按此操作过滤
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


object_classes = ["car","truck_head","RTG","tray_w_container","tray_wo_container","QC",
                  "lock_station","other_vehicle","pedestrian","street_lamp"]
"""
  - car
  - truck_head
  - RTG
  - tray_w_container
  - tray_wo_container
  - QC
  - lock_station
  - other_vehicle
  - pedestrian
  - stree_lamp
"""
object_classes_dict_train=dict()
object_classes_dict_val=dict()
for name in object_classes:
    object_classes_dict_train[name]=0
    object_classes_dict_val[name]=0

# camera_type_ww = ["frontleft","frontmid","frontright","rearleft","rearright","topleft","topright"]
camera_type_ww = ["frontleft","frontmid","frontright","rearleft","rearright","topleft"]
import pdb
if __name__ == "__main__":
    # zjh add: obj filter by range, do need filte by range, you can add obj filter by range in train test pipeline
    point_cloud_range=[-140.0, -160.0, -1.0, 180.0, 160.0, 10.0]
    # point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] # old range 
    xy_range = [point_cloud_range[0],point_cloud_range[1],point_cloud_range[3],point_cloud_range[4]]

    # dataset_split = sys.argv[1]
    # print("please input 'train' or 'val' ")        
    # dataset_split = input()
    # if dataset_split == "train":
    #     create_train = True
    # elif dataset_split == "val":
    #     create_train = False
    # else:
    #     print("please input 'train' or 'val ")        
    # create_train = False # else create_val , info dict is same

    root = "./data/ww/"
    # datacali_path = root + "calib/camera"
    dataname = "0622_data_4528frames_147scenes"
    dataroot = root + dataname
    
    
    data_ann_root_train = "0622_data_4528frames_147scenes_label"
    # data_ann_root_val = "0622_data_4528frames_147scenes_label_val"
    test_pkl_save_name = "ww_infos_test.pkl" 
    # if create_train:
    train_pkl_save_name = "ww_infos_train.pkl"
    data_ann_root = root+ data_ann_root_train # 132个scenes
    data_train={"infos":[],"metadata":{"version":"train"}}

    # else:
    val_pkl_save_name = "ww_infos_val.pkl"
    # data_ann_root_val = root+ data_ann_root_val #14个scenes
    val_scene_list ={
        "1687261817_scene_0",
        "1687279284_scene_1",
        "1687280927_scene_0",
        "1687285633_scene_0",
        "1687288375_scene_3",
        "1687291469_scene_1",
        "1687294974_scene_1",
        "1687298168_scene_0",
        "1687301814_scene_0",
        "1687303518_scene_1",
        "1687309007_scene_0",
        "1687311834_scene_2",
        "1687313930_scene_0",
        "1687314354_scene_2"
    }
    data_val={"infos":[],"metadata":{"version":"val"}}
        
    # pdb.set_trace()
    # calib_dict,Ks,Es = get_calib(datacali=datacali_path)

    scenes = glob.glob(osp.join(data_ann_root,"*scene*"))
    scenes.sort()
    
    frames=0
    frames_train=0
    scenes_num_train = 0
    instance_train={}

    frames_val=0
    scenes_num_val = 0
    instance_val=0

    for scene_datepath in scenes:
        scene_datename = scene_datepath.split("/")[-1]
        scene_datapath_raw = osp.join(dataroot,scene_datename)
        print(scene_datename)
        if scene_datename in val_scene_list:
            scenes_num_val+=1
        else:
            scenes_num_train+=1
        # pdb.set_trace()
        # get calib: one calib file for each scene
        datacali_path = osp.join(scene_datapath_raw,"calib","camera")
        calib_dict,Ks,Es = get_calib(datacali=datacali_path)

        ann_paths = glob.glob(osp.join(scene_datepath,"lidar","*json")) # lidar path 
        ann_paths.sort()
        
        # start show single frame 
        for frames_cur_scene,ann_json_path in enumerate(ann_paths):
            
            # print(ann_json_path)
            # 1 get lidar 
            # if create_train:
            lidar_path = ann_json_path.replace(data_ann_root_train,dataname)
            # else:
            #     lidar_path = ann_json_path.replace(data_ann_root_val,dataname)
            lidar_path = lidar_path.replace("json","pcd")
            use_dim7=False
            if use_dim7:
                lidar_path = lidar_path.replace("0622_data_4528frames_147scenes","0622_data_4528frames_147scenes_withRGB")
                lidar_path = lidar_path.replace("pcd","npy")
            
            # 2 get ann
            if not osp.exists(ann_json_path):
                print(ann_json_path,"not exit!")
                continue
            # load ann boxes
            with open(ann_json_path, 'r') as f:
                ann_json = json.load(f)
            # print(ann_json_path)
            # print(ann_json)
            bboxes = []
            labels = [] # ind
            num_lidar_pts =[]
            for meta in ann_json["dataList"]:
                box_xyz = [meta["center"]["x"],meta["center"]["y"],meta["center"]["z"]]
                # box_lwh = [meta["dimensions"]["length"],meta["dimensions"]["width"],meta["dimensions"]["height"]]
                box_wlh = [meta["dimensions"]["width"],meta["dimensions"]["length"],meta["dimensions"]["height"]]
                yaw = meta["rotation"]["z"]
                yaw = [-yaw-np.pi/2]

                label = meta["label"]
                if "self_tray" == label:
                    # label = "other_vehicle"
                    continue
                # if meta["point"] < 1:
                #     print(label,meta["point"],"too few pts!")
                #     # pdb.set_trace()
                #     continue
                use_one_tray = True
                if use_one_tray:
                    if "tray_wo_container" == label: # 将所有的托架都定义为 tray_w_container
                        label = "tray_w_container"                  
                labels.append(label)
                bboxes.append(box_xyz+box_wlh+yaw+[0,0])
                num_lidar_pts.append(meta["point"])

            bboxes = np.array(bboxes)
            labels = np.array(labels)
            num_lidar_pts = np.array(num_lidar_pts)

            # filter by xy range
            filter_success, filter_flag = in_range_bev(bboxes,xy_range)
            if filter_success:
                bboxes = bboxes[filter_flag]
                labels = labels[filter_flag]
            else:
                print("there is no target in this frame: ",xy_range)
                continue
            if bboxes.shape[0] == 0:
                print("there is no target in bev range of this frame: ",xy_range)
                continue
            
            # get img
            timestamp = ann_json_path.split("/")[-1].split(".")[0]
            img_dict=dict()
            img_list=[]

            cams_dict=dict()
            """
            与 ww-dataset.py 中 get_data_info 对齐
            """
            for k,camera_type in enumerate(camera_type_ww):
                img_path = osp.join(scene_datapath_raw,"camera",camera_type,timestamp+".jpg")

                cams_dict[camera_type]={"data_path":img_path,"type":camera_type,
                                        "camera_intrinsics":Ks[camera_type],
                                        "lidar2camera":Es[camera_type],
                                        "camera2lidar":np.linalg.inv(Es[camera_type]).astype(np.float32),
                                        "camera2ego":np.linalg.inv(Es[camera_type]).astype(np.float32),
                                        "lidar2image":calib_dict[camera_type] ,
                                        "timestamp":timestamp}

                # image = mmcv.imread(img_path)
            token = ann_json_path.replace("/","_")[1:]
            info = {
                "lidar_path": lidar_path,
                "token": token,
                "sweeps": "ww do not have,seq pre and next frame",
                "cams": cams_dict.copy(),
                "lidar2ego": np.eye(4).astype(np.float32),
                "ego2global":  np.eye(4).astype(np.float32),# 先留着
                "timestamp": timestamp,
                "location": "taiguo_ww", # 地图
                "gt_boxes": bboxes,
                "gt_names":labels,
                "num_lidar_pts":num_lidar_pts #np.ones(labels.shape[0]),
                }
                # "ann_json_path":ann_json_path
                # "valid_flag":True if bboxes.shape[0]!=0 else False # no obj then false
                # 'valid_flag': array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
                # True,  True,  True,  True,  True,  True,  True, False,  True,
                # True,  True,  True])}]
            
            if scene_datename in val_scene_list:
                data_val["infos"].append(info)
                frames_val+=1
                for label in labels:
                    object_classes_dict_val[label]+=1
                
            else:
                data_train["infos"].append(info)
                frames_train+=1
                for label in labels:
                    object_classes_dict_train[label]+=1

            frames+=1

    mmcv.dump(data_train,osp.join(root,train_pkl_save_name))
    mmcv.dump(data_val,osp.join(root,val_pkl_save_name))
    mmcv.dump(data_val,osp.join(root,test_pkl_save_name))

    print(data_train["infos"][0])
    print("scenes_sum: ", len(scenes))
    print("frames_sum: ",frames)
    
    print("scenes_sum_train: ", scenes_num_train)
    print("frames_sum_train: ",frames_train)
    
    print("scenes_sum_val: ", scenes_num_val)
    print("frames_sum_val: ",frames_val)
    
    print("*******************train instance distribution*********")
    for k,v in object_classes_dict_train.items():
        print("{:<20}".format(k)," ",v)    
    print("*******************val   instance distribution*********")
    for k,v in object_classes_dict_val.items():
        print("{:<20}".format(k)," ",v)