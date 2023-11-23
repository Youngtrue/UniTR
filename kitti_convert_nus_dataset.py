      
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
import math
# from ..core import show_multi_modality_result, show_result
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose

# from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class KittiDataset(Custom3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    # CLASSES = ('car', 'twistlock_station', 'pedestrian', 'traffic_cone', 'igv', 'truck')
    CLASSES = ('car', 'twistlock_station', 'qc', 'pedestrian', 'traffic_cone', 'igv', 'truck')
    # CLASSES = (
    #     'car', 'twistlock_station', 'qc', 'pedestrian', 'traffic_cone', 'pole', 'igv', 'truck', 'box_truck',
    #     'box_igv', 'tray', 'truck_head'
    # )

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 map_classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
                 use_valid_flag=False,
                 lidar_only=True,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode
            # split=split
        )

        self.map_classes = map_classes
        self.use_valid_flag = use_valid_flag
        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix
        self.lidar_only = lidar_only
        self.data_infos = self.append_data_info()


    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        idx_str = str(idx)
        if len(idx_str) < 6:
            idx_str = f'{idx:06d}'
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx_str}.bin')
        # pts_filename = osp.join(self.root_split,
        #                         f'{idx}.bin')
        return pts_filename

    def append_data_info(self):
        data_info = []
        for i, info in enumerate(self.data_infos):
            this_info = {}
            x = self.get_data_info(i, True)
            this_info['lidar_path'] = x['lidar_path']
            this_info['token'] = x['token']
            this_info['sweeps'] = x['sweeps']
            cams_info = {}
            if not self.lidar_only:
                type_list = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]
                for index, type in enumerate(type_list):
                    cams = {}
                    cams['data_path'] = info['image']['image_path'][index]
                    cams['type'] = type
                    cams['sample_data_token'] = info['image']['image_idx']
                    cams['sensor2ego_translation'] = x['camera2ego'][index]
                    cams['sensor2ego_rotation'] = x['camera2ego'][index]
                    cams['ego2global_translation'] = x['ego2global']
                    cams['ego2global_rotation'] = x['ego2global']
                    cams['sensor2lidar_translation'] = x['camera2lidar_translation'][index]
                    cams['sensor2lidar_rotation'] =  x['camera2lidar_rotation'][index]
                    cams['camera_intrinsics'] = x['camera_intrinsics'][index]
                    cams_info[type] = cams
                this_info['cams'] = cams_info
                this_info['lidar2camera'] = x['lidar2camera']
                this_info['lidar2image'] = x['lidar2image']
                this_info['camera2lidar'] = x['camera2lidar']
            this_info['lidar2ego_translation'] = x['lidar2ego']
            this_info['lidar2ego_rotation'] = np.zeros((4,4))
            this_info['ego2global_translation'] = x['ego2global']
            this_info['ego2global_rotation'] = np.zeros((4,4))
            this_info['timestamp'] = x['timestamp']
            this_info['location'] = x['location']


            if not self.test_mode:
                gt_names = []
                this_info['gt_boxes'] = x['ann_info']['gt_bboxes_3d']
                labels = x['ann_info']['gt_labels_3d']
                num_boxes = len(x['ann_info']['gt_bboxes_3d'])
                this_info['num_lidar_pts'] = np.zeros(num_boxes)
                this_info['num_radar_pts'] = np.zeros(num_boxes)
                this_info['valid_flag'] = np.full(num_boxes, True)
                for label in labels:
                    gt_names.append(self.CLASSES[label])
                this_info['gt_names'] = gt_names
                this_info['gt_velocity'] = np.array([0, 0])
            else:
                this_info['gt_boxes'] = x['ann_info']['gt_bboxes_3d']

            this_info['annos'] = info
            data_info.append(this_info)
        return data_info


    def get_data_info(self, index, is_ini=False):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        if is_ini:
            info = self.data_infos[index]
        else:
            info = self.data_infos[index]['annos']
        sample_idx = info['image']['image_idx']
        if not self.lidar_only:
            calib_list = info['calib']
        #     # TODO: consider use torch.Tensor only
        #     rect = calib['R0_rect'].astype(np.float32)
        #     Trv2c = calib['Tr_velo_to_cam'].astype(np.float32)
        #     P2 = calib['P2'].astype(np.float32)
        #     lidar2img = P2 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        # pts_filename = info['point_cloud']['velodyne_path']
        data = dict(
            token=sample_idx,
            sample_idx=sample_idx,
            lidar_path=pts_filename,
            pts_filename=pts_filename,
            sweeps=[],
            timestamp=sample_idx,
            location='xxx')

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        # ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        # ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        # lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        # lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for idx, calib in enumerate(calib_list):
                img_filename = info['image']['image_path'][idx]
                data['image_paths'].append(img_filename)

                # TODO: consider use torch.Tensor only
                Trv2c = calib['Tr_velo_to_cam'].astype(np.float32)
                data["lidar2camera"].append(Trv2c)

                rect = calib['R0_rect'].astype(np.float32)
                P2 = calib['P2'].astype(np.float32)
                data["camera_intrinsics"].append(P2)

                lidar2img = P2 @ rect @ Trv2c
                data["lidar2image"].append(lidar2img)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                # camera2ego[:3, :3] = Quaternion(
                #     camera_info["sensor2ego_rotation"]
                # ).rotation_matrix
                # camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                lidar2camera_r = Trv2c[:3, :3]
                camera2lidar_r = np.linalg.inv(lidar2camera_r)
                data['camera2lidar_rotation'] = camera2lidar_r

                lidar2camera_t = Trv2c[:3, 3]
                camera2lidar_t = lidar2camera_t @ camera2lidar_r.T
                data['camera2lidar_translation'] = camera2lidar_t

                camera2lidar_rt = np.eye(4).astype(np.float32)

                camera2lidar_rt[:3, :3] = camera2lidar_r.T
                camera2lidar_rt[3, :3] = -camera2lidar_t
                data["camera2lidar"].append(camera2lidar_rt.T)

        if not self.test_mode and is_ini:
            annos = self.get_ann_info(index, True)
            data['ann_info'] = annos
        elif not self.test_mode:
            annos = self.get_ann_info(index)
            data['ann_info'] = annos
        else:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            annos = self.remove_dontcare(annos)
            loc = annos['location']
            dims = annos['dimensions']

            rots = annos['rotation_y']
            gt_names = annos['name']
            # if (dims < 0).any():
            #     print(info)
            #     print("dims=", dims)
            gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                          axis=1).astype(np.float32)
            gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,
                box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0))
            gt_bboxes = annos['bbox']
            selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
            gt_bboxes = gt_bboxes[selected].astype('float32')
            gt_names = gt_names[selected]

            gt_labels = []
            for cat in gt_names:
                if cat in self.CLASSES:
                    gt_labels.append(self.CLASSES.index(cat))
                else:
                    gt_labels.append(-1)
            gt_labels = np.array(gt_labels).astype(np.int64)
            gt_labels_3d = copy.deepcopy(gt_labels)

            annos = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                bboxes=gt_bboxes,
                labels=gt_labels,
                gt_names=gt_names)
            data['ann_info'] = annos
        return data


    def get_ann_info(self, index, is_ini = False):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        if is_ini:
            info = self.data_infos[index]
        else:
            info = self.data_infos[index]['annos']
        #info = self.data_infos[index]
        #print("index--------------------------",index)


        if 'plane' in info:
            rect = info['calib']['R0_rect'].astype(np.float32)
            Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
            # convert ground plane to velodyne coordinates
            reverse = np.linalg.inv(rect @ Trv2c)

            (plane_norm_cam,
             plane_off_cam) = (info['plane'][:3],
                               -info['plane'][:3] * info['plane'][3])
            plane_norm_lidar = \
                (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
            plane_off_lidar = (
                reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                reverse[:3, 3])
            plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
            plane_lidar[:3] = plane_norm_lidar
            plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
        else:
            plane_lidar = None

        difficulty = info['annos']['difficulty']
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']

        # rots = -1 * annos['rotation_y'] + math.pi/2
        rots = annos['rotation_y']
        gt_names = annos['name']

        if (dims < 0).any():
            print(info)
            print("dims=", dims)
        #     ddd = np.unique(np.where(dims<=0)[0])
        #     loc = np.delete(loc, ddd, axis=0)
        #     dims = np.delete(dims, ddd, axis=0)
        #     rots = np.delete(rots, ddd, axis=0)
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        #gt_bboxes_3d[..., 2] -= gt_bboxes_3d[..., 5] / 2
        #gt_bboxes_3d[..., 6] = -1 * gt_bboxes_3d[...,6] + math.pi/2
        # convert gt_bboxes_3d to velodyne coordinates
        # gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,
        #     box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d) #, np.linalg.inv(rect @ Trv2c)

        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,
                                            box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0))
        gt_bboxes = annos['bbox']
        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names,
            plane=plane_lidar,
            difficulty=difficulty)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'dontcare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti2d(outputs, self.CLASSES,
                                                    pklfile_prefix,
                                                    submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if 'img' in name:
                    result_files = self.bbox2result_kitti2d(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir


    def convert_gtannos2kitti(self, gt_annos):
        gt_infos = []
        for anno in gt_annos:
            anno_ = anno['annos']
            info = {k:v for k, v in anno_.items()}
            info['sample_idx'] = anno['image']['image_idx']
            gt_infos.append(info)
        return gt_infos


    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import kitti_eval
        gt_annos = [info['annos'] for info in self.data_infos]

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                if not self.lidar_only:
                    eval_types = ['bbox', 'bev', '3d']
                    if 'img' in name:
                        eval_types = ['bbox']
                else:
                    eval_types = ['3d']

                gt_annos = self.convert_gtannos2kitti(gt_annos)


                from nuscenes import NuScenes
                from nuscenes.eval.detection.evaluate import NuScenesEval


                ap_result_str, ap_dict_ = kitti_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)

                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            if metric == 'img_bbox':
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
                                                    self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            if not self.lidar_only:
                sample_idx = info['image']['image_idx']
                image_shape = info['image']['image_shape'][:2]
            else:
                sample_idx = info['image']['image_idx']
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if not self.lidar_only:
                if len(box_dict['bbox']) > 0:
                    box_2d_preds = box_dict['bbox']
                    box_preds = box_dict['box3d_camera']
                    scores = box_dict['scores']
                    box_preds_lidar = box_dict['box3d_lidar']
                    label_preds = box_dict['label_preds']

                    for box, box_lidar, bbox, score, label in zip(
                            box_preds, box_preds_lidar, box_2d_preds, scores,
                            label_preds):
                        bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                        bbox[:2] = np.maximum(bbox[:2], [0, 0])
                        anno['name'].append(class_names[int(label)])
                        anno['truncated'].append(0.0)
                        anno['occluded'].append(0)
                        anno['alpha'].append(
                            -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                        anno['bbox'].append(bbox)
                        anno['dimensions'].append(box[3:6])
                        anno['location'].append(box[:3])
                        anno['rotation_y'].append(box[6])
                        anno['score'].append(score)

                    anno = {k: np.stack(v) for k, v in anno.items()}
                    annos.append(anno)
                else:
                    anno = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    }
                    annos.append(anno)
            else:
                if len(box_dict['box3d_lidar']) > 0:
                    scores = box_dict['scores']
                    box_preds_lidar = box_dict['box3d_lidar']
                    label_preds = box_dict['label_preds']
                    for box_lidar, score, label in zip(box_preds_lidar, scores, label_preds):
                        # bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                        # bbox[:2] = np.maximum(bbox[:2], [0, 0])
                        anno['name'].append(class_names[int(label)])
                        anno['truncated'].append(0.0)
                        anno['occluded'].append(0)
                        anno['alpha'].append(0)
                        anno['bbox'].append([0,0,0,0])
                        anno['dimensions'].append(box_lidar[3:6])
                        anno['location'].append(box_lidar[:3])
                        anno['rotation_y'].append(box_lidar[6])
                        anno['score'].append(score)

                    anno = {k: np.stack(v) for k, v in anno.items()}
                    annos.append(anno)
                else:
                    anno = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    }
                    annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = self.data_infos[i]['image']['image_idx']

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(0.0)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            # save file in pkl format
            pklfile_path = (
                pklfile_prefix[:-4] if pklfile_prefix.endswith(
                    ('.pkl', '.pickle')) else pklfile_prefix)
            mmcv.dump(det_annos, pklfile_path)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.data_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        if not self.lidar_only:
            sample_idx = info['image']['image_idx']
        else:
            sample_idx = info['lidar_path']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            if not self.lidar_only:
                return dict(
                    bbox=np.zeros([0, 4]),
                    box3d_camera=np.zeros([0, 7]),
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx)
            else:
                return dict(
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx
                )

        if not self.lidar_only:
            rect = info['calib']['R0_rect'].astype(np.float32)
            Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
            P2 = info['calib']['P2'].astype(np.float32)
            img_shape = info['image']['image_shape']
            P2 = box_preds.tensor.new_tensor(P2)

            box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

            box_corners = box_preds_camera.corners
            box_corners_in_image = points_cam2img(box_corners, P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # Post-processing
            # check box_preds_camera
            image_shape = box_preds.tensor.new_tensor(img_shape)
            valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                              (box_2d_preds[:, 1] < image_shape[0]) &
                              (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))

        if not self.lidar_only:
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        else:
            valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            if not self.lidar_only:
                return dict(
                    bbox=box_2d_preds[valid_inds, :].numpy(),
                    box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                    box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                    scores=scores[valid_inds].numpy(),
                    label_preds=labels[valid_inds].numpy(),
                    sample_idx=sample_idx)
            else:
                return dict(
                    box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                    scores=scores[valid_inds].numpy(),
                    label_preds=labels[valid_inds].numpy(),
                    sample_idx=sample_idx)
        else:
            if not self.lidar_only:
                return dict(
                    bbox=np.zeros([0, 4]),
                    box3d_camera=np.zeros([0, 7]),
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx)
            else:
                return dict(
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx)



    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        return Compose(pipeline)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    