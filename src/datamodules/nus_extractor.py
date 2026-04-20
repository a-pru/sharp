#!/usr/bin/env python
# coding: utf-8

from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.eval.common.utils import quaternion_yaw

from pyquaternion import Quaternion
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import av2.geometry.interpolate as interp_utils
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os
import time

def getCatId(category_name):
    if "vehicle.car" in category_name:
        return 0
    elif "vehicle.ego" in category_name:
        return 0
    elif "vehicle.bicycle" in category_name:
        return 2
    elif "vehicle.other" in category_name or "vehicle.unknown" in category_name or "vehicle.bus.rigid" in category_name or "vehicle.construction" in category_name  or "vehicle.trailer" in category_name:
        return 3
    elif "human" in category_name:
        return 1
    elif "vehicle.truck" in category_name:
        return 3
    elif "vehicle.motorcycle" in category_name:
        return 3
    elif "animal" in category_name:
        return 4
    elif "object" in category_name or "static.manmade" in category_name:
        return 4
    else:
        return 5


def processAgentData(agent_data, data_record):
    xs = []
    pms = []
    hs = []
    ats = []
    i = 0

    for ad in agent_data:
        xs.append( torch.cat([ad["history_xy"], ad["future_xy"]]).unsqueeze(0) )
        pms.append( torch.cat([ad["history_padding_mask"], ad["future_padding_mask"]]).unsqueeze(0) )
        hs.append( torch.cat([ad["history_heading"], ad["future_heading"]]).unsqueeze(0) )
        ats.append(ad["type"])

    try:
        x_positions = torch.cat(xs, dim=0)
    except Exception as e:
        for x in xs: print(x.shape)
        for ad in agent_data: print( ad["history_xy"].shape, ad["future_xy"].shape )
        exit()
    try:
        padding_mask = torch.cat(pms, dim=0)
    except Exception as e:
        for p in pms: print(p.shape)
        for ad in agent_data: print( ad["history_mask"].shape, ad["future_mask"].shape )
        exit()
    try:
        x_angles = torch.cat(hs, dim=0)
    except Exception as e:
        for h in hs: print(h.shape)
        for ad in agent_data: print( ad["history_heading"].shape, ad["future_heading"].shape )
        exit()
    
    num_actors = x_positions.shape[0]
    x_attr = torch.zeros(num_actors, 3)
    for i in range(num_actors):
        x_attr[i, :] = getCatId(ats[i])

    tmp = {
        'x_attr': x_attr,
        'x_positions': x_positions,
        'x_angles': x_angles,
        'x_padding_mask': padding_mask,
    }
    for k in tmp.keys(): data_record[k] = tmp[k].float() 
    assert x_attr.shape[0] == x_positions.shape[0]
    return data_record


def get_map_features(map_api, map_center, radius, points_distance=1):
    ret = {}
    map_objs = map_api.get_records_in_radius(map_center[0], map_center[1], radius, ["lane", "lane_connector"])

    # normal lane
    for id in map_objs["lane"]:
        lane_info = map_api.get("lane", id)
        assert lane_info["token"] == id
        ret[id] = {
            "TYPE": "LANE",
            "POLYLINE": np.asarray(arcline_path_utils.discretize_lane(map_api.arcline_path_3[id], resolution_meters=points_distance)),
            "ENTRY": map_api.get_incoming_lane_ids(id),
            "EXIT": map_api.get_outgoing_lane_ids(id),
        }

    # intersection lane
    for id in map_objs["lane_connector"]:
        lane_info = map_api.get("lane_connector", id)
        assert lane_info["token"] == id
        ret[id] = {
            "TYPE": "LANE_INTERSECTION",
            "POLYLINE": np.asarray(arcline_path_utils.discretize_lane(map_api.arcline_path_3[id], resolution_meters=points_distance)),
            "ENTRY": map_api.get_incoming_lane_ids(id),
            "EXIT": map_api.get_outgoing_lane_ids(id)
        }
    return ret


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--plot", help="", action="store_true")
    args = parser.parse_args()
    plot = args.plot
    DATAROOT = '/TODO/nuscenes/'
    radius = 100
    splits = ["train", "train_val", "val"]
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    h_horizon = 2  
    f_horizon = 6

    nusc = NuScenes('v1.0-trainval', dataroot=DATAROOT)
    helper = PredictHelper(nusc) 
    nusc_maps = {}
    for nm in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']:
        nusc_maps[nm] = NuScenesMap(map_name=nm, dataroot=DATAROOT)
    
    for split in splits:
        data = get_prediction_challenge_split(split, dataroot=DATAROOT)
        print("process", split, " - samples:", len(data))
        stored = 0
        for i, scen in enumerate(tqdm(data)):
            instance_token, sample_token = scen.split("_")
            annotation = helper.get_sample_annotation(instance_token, sample_token)
            nusc_map = nusc_maps[helper.get_map_name_from_sample_token(sample_token)]

            origin = np.array([annotation["translation"][0], annotation["translation"][1]])
            theta = quaternion_yaw(Quaternion(annotation["rotation"]))

            lane_positions = [] 
            map_features = get_map_features(nusc_map, origin, radius)
            for elem in map_features.keys():
                map_xy_global = map_features[elem]["POLYLINE"][:, :2]
                map_xy_global = interp_utils.interp_arc(20, map_xy_global)                   
                
                lane_positions.append(map_xy_global)

            lane_positions = torch.from_numpy(np.array(lane_positions))

            all_instance_tokens = [instance_token]
            all_instance_tokens += [ann['instance_token'] for ann in nusc.sample_annotation if ann['sample_token'] == sample_token and ann['instance_token'] != instance_token]

            agent_data = []
            present = helper.get_annotations_for_sample(sample_token)
            for agent_token in all_instance_tokens:
                agent_data.append({"type": "none"})

                for pafu in ["history", "future"]:
                    func = helper.get_past_for_agent if pafu == "history" else helper.get_future_for_agent
                    horizon = h_horizon if pafu == "history" else f_horizon
                    data_global = func(agent_token, sample_token, seconds=horizon, in_agent_frame=False, just_xy=False)
                    horizon_steps = int(horizon*2) + (1 if pafu == "history" else 0)

                    if len(data_global) < 1:
                        agent_data[-1][pafu + "_padding_mask"] = torch.ones((horizon_steps), dtype=bool)
                        agent_data[-1][pafu + "_xy"] = torch.zeros((horizon_steps, 2))
                        agent_data[-1][pafu + "_heading"] = torch.zeros((horizon_steps))
                        continue

                    agent_data[-1]["type"] = data_global[-1]["category_name"]  

                    data_global = ([p for p in present if p["instance_token"] == agent_token] if pafu == "history" else []) +  [x for x in data_global]
                    xy_global = np.stack([x["translation"][:2] for x in data_global], axis=0)
                    yaw_global = np.stack( [quaternion_yaw(Quaternion(x["rotation"])) for x in data_global], axis=0)

                    num_observed = xy_global.shape[0]
                    t_mask = np.ones([horizon_steps], dtype=bool)
                    
                    if pafu == "history":
                        t_mask[-num_observed:] = 0
    
                        xy_global = np.flip(xy_global, axis=0)
                        xy_global = np.pad(
                            xy_global,
                            pad_width=((horizon_steps-num_observed, 0), (0, 0)),  # Pad only rows
                            mode='constant',
                            constant_values=0
                        )

                        yaw_global = np.flip(yaw_global, axis=0)
                        yaw_global = np.pad(
                            yaw_global,
                            pad_width=((horizon_steps-num_observed, 0)),  # Pad only rows
                            mode='constant',
                            constant_values=0
                        )
                    else:
                        t_mask[:num_observed] = 0

                        xy_global = np.pad(
                            xy_global,
                            pad_width=((0, horizon_steps-num_observed), (0, 0)),  # Pad only rows
                            mode='constant',
                            constant_values=0
                        )
                        yaw_global = np.pad(
                            yaw_global,
                            pad_width=((0, horizon_steps-num_observed)),  # Pad only rows
                            mode='constant',
                            constant_values=0
                        )

                    agent_data[-1][pafu + "_padding_mask"] = torch.from_numpy(t_mask)
                    agent_data[-1][pafu + "_xy"] = torch.from_numpy(xy_global)
                    agent_data[-1][pafu + "_heading"] = torch.from_numpy(yaw_global)

                if agent_data[-1]["history_xy"].shape[0] != h_horizon*2+1 or agent_data[-1]["future_xy"].shape[0] != f_horizon*2:
                    assert False
                if torch.all(agent_data[-1]["history_padding_mask"] == 1) or torch.all(agent_data[-1]["future_padding_mask"] == 1):
                    assert False
            
            data_record = {}
            data_record['scenario_id'] = "{}_{}".format(instance_token, sample_token)
            data_record['track_id'] = instance_token 
            data_record['origin'] = torch.tensor(origin) 
            data_record['theta'] = torch.tensor([theta]).float() 
            data_record = processAgentData(agent_data, data_record)
            data_record["lane_positions"] = lane_positions
            data_record["x_valid_mask"] = ~(data_record["x_padding_mask"].bool())
            data_record["agent_idx"] = 0    
            
            out_split = "train" if split == "train" or split == "train_val" else "val"
            out_dir = "/TODO/nuscenes/sharp_nus_processed_/{}/".format(out_split)
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            torch.save(data_record, "{}/{}_{}.pt".format(out_dir, instance_token, sample_token))
            stored += 1
        print("stored {} samples".format(stored))
    return



if __name__ == "__main__":
    main()