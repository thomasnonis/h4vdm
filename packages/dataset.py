import os
import random
import pickle
import json
from torch.utils.data import Dataset

from packages.video_utils import H264Extractor, Video
from packages.common import create_custom_logger

log = create_custom_logger('dataset.py')

def download(url, folder_path):
    filename = url.split('/')[-1]
    full_path = os.path.join(folder_path, filename)
    if os.path.exists(full_path):
        log.info(f'File {filename} already exists in {os.path.join(folder_path, filename)}, skipping download')
        return full_path
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    os.system(f'wget -P {folder_path} {url}')
    if not os.path.exists(full_path):
        raise RuntimeError(f'Error while downloading {url}')
    return full_path

# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================

class VisionDataset(Dataset):
    """VISION dataset (https://lesc.dinfo.unifi.it/VISION/)
    """

    log = create_custom_logger('VisionDataset')

    def __init__(self, root_path: str, download_on_init = False, shuffle = False, devices: list = [], media_types: list = [], properties: list = [], extensions: list = []):
        
        VisionDataset.log.debug(f'Building VISION dataset in {root_path}')
        self.root_path = root_path

        self.root = os.path.join(root_path, 'VISION')
        if not os.path.exists(self.root):
            try:
                os.makedirs(self.root)
            except FileNotFoundError:
                raise FileNotFoundError(f'VISION dataset folder {self.root} does not exist and cannot be created')
        
        self.length = 0
        self.dataset = {}
        self._build_dataset(devices, media_types, properties, extensions, download_on_init)

        # build a map that from a number finds the correct keys to the dict
        self.index_to_dict_map = self._build_index_to_dict_map(shuffle)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if index > self.length:
            raise RuntimeError(f'index {index} is greater than dataset length {self.length}')
        
        index_map = self.index_to_dict_map[index]
        video = self.dataset[index_map['device']][index_map['video']]

        if not os.path.exists(os.path.join(video['path'], video['filename'])):
            VisionDataset.log.debug(f'Downloading {video["url"]} to {video["path"]}')
            downloaded_full_path = download(video['url'], video['path'])
            if downloaded_full_path != os.path.join(video['path'], video['filename']):
                raise RuntimeError(f'Error while downloading {video["url"]}. Downloaded file is {downloaded_full_path} instead of {os.path.join(video["path"], video["filename"])}')
            
        return video
    
    def _parse_url(self, url):
        structure = url.split('https://lesc.dinfo.unifi.it/VISION/dataset/')[1].split('/')
        device = structure[0]
        media_type = structure[1]
        property = structure[2]
        filename = structure[3]

        return device, media_type, property, filename
    
    def _build_dataset(self, devices = [], media_types = [], properties = [], extensions = [], download_on_init: bool = False) -> dict:
        vision_files_path = download('https://lesc.dinfo.unifi.it/VISION/VISION_files.txt', self.root)
        with open(vision_files_path) as f:
            urls = f.readlines()

            for url in urls:
                url = url.split('\n')[0]
                device, media_type, property, filename = self._parse_url(url)

                # skip the frames_for_prnu folder
                if 'frames_for_prnu' in url:
                    continue

                # if the user specified a list of devices, media_types or properties, skip the element if it is not in the list
                # otherwise, add everything to the structure
                if len(devices) > 0 and device not in devices:
                    continue

                if len(media_types) > 0 and media_type not in media_types:
                    continue

                if len(properties) > 0 and property not in properties:
                    continue

                if len(extensions) > 0 and filename.split('.')[-1] not in extensions:
                    continue

                # build structure if it does not exist
                if device not in self.dataset.keys():
                    self.dataset[device] = []
                    
                # instantiate entry
                video = {}
                video['device'] = device
                video['media_type'] = media_type
                video['property'] = property
                video['url'] = url
                video['filename'] = filename
                video['name'] = filename.split('.')[0]
                video['extension'] = filename.split('.')[1]
                video['local_path'] = os.path.join(device, media_type, property)
                video['path'] = os.path.join(self.root, device, media_type, property)

                # add element to structure
                self.dataset[device].append(video)

                if download_on_init:
                    downloaded_full_path = download(video['url'], video['path'])
                    if downloaded_full_path != os.path.join(video['path'], video['filename']):
                        raise RuntimeError(f'Error while downloading {url}. Downloaded file is {downloaded_full_path} instead of {os.path.join(video["path"], video["filename"])}')

                self.length += 1
                    
        return self.dataset
    
    def _build_index_to_dict_map(self, shuffle: bool) -> list:
        index_to_dict_map = []

        for device in self.dataset.keys():
            for i, video in enumerate(self.dataset[device]):
                index_to_dict_map.append({'device': device, 'video': i})
        
        if len(index_to_dict_map) != self.length:
            raise RuntimeError(f'Index to dict map length {len(index_to_dict_map)} is different from dataset length {self.length}')
        
        if shuffle:
            random.shuffle(index_to_dict_map)
        return index_to_dict_map
    
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=4)

class VisionGOPDataset(VisionDataset):

    log = create_custom_logger('VisionGOPDataset')

    def __init__(
            self,
            root_path: str,
            devices: list = [],
            media_types: list = [],
            properties: list = [],
            extensions: list = [],
            gop_size: int = 8,
            frame_width: int = 224,
            frame_height: int = 224,
            gops_per_video: int = 1,
            build_on_init: bool = False,
            force_rebuild: bool = False,
            download_on_init = False,
            shuffle = False):
        
        VisionGOPDataset.log.debug(f'Building VISION GOP dataset in {root_path}')
        super().__init__(root_path, download_on_init, shuffle, devices, media_types, properties, extensions)
        
        self.gop_root = os.path.join(root_path, 'VISION_GOP')
        self.gop_size = gop_size
        self.gops_per_video = gops_per_video
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.force_rebuild = force_rebuild
        self.build_on_init = build_on_init

        if gop_size < 1:
            raise ValueError('gop_size must be greater than 0')
        
        if frame_width < 1 or frame_height < 1:
            raise ValueError('frame_width and frame_height must be greater than 0')
        
        if gops_per_video < 1:
            raise ValueError('gops_per_video must be greater than 0')

        if build_on_init:
            for i in range(len(self)):
                VisionGOPDataset.log.debug(f'Building video {i+1}/{len(self.dataset)}')
                video = self[i]

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        if index >= len(self):
            raise RuntimeError(f'index {index} is greater than dataset length {len(self)}')
        
        video_metadata = super().__getitem__(index)

        # if built on init with force rebuild, no need to forcibly rebuild it here
        if self.build_on_init == True and self.force_rebuild == True:
            rebuild = False
        else:
            rebuild = self.force_rebuild

        video_filename = os.path.join(video_metadata['path'], video_metadata['filename'])

        # saved_video_filename = os.path.join(self.gop_root, video_metadata['name'] + '.video')
        # if os.path.exists(saved_video_filename) and not rebuild:
        #     Video.log.info(f'Video file for {video_filename} already exists at {saved_video_filename}, skipping conversion')
        #     return Video.load(saved_video_filename)
        
        if 'object' not in video_metadata.keys() or rebuild:
            VisionGOPDataset.log.debug(f'Building video {index}: {video_metadata["filename"]}')
            video = Video(video_filename, video_metadata['device'], self.frame_width, self.frame_height, self.gops_per_video, self.gop_size, extract_gops_on_init=True)
            video_metadata['object'] = Video.save(video, os.path.join(self.gop_root, video_metadata['local_path']))
        else:
            Video.log.info(f'Video object for {video_filename} already exists, loading it from {video_metadata["object"]}')
            video = Video.load(video_metadata['object'])
        
        return video
    
    def _build_cartesian_product(self):
        cartesian_product = []

        for device1 in self.dataset.keys():
            for device2 in self.dataset.keys():
                cartesian_product.append((device1, device2))