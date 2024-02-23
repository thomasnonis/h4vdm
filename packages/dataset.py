import os
import random
import pickle
import json
import logging
from torch.utils.data import Dataset

from packages.video_utils import H264Extractor, Video, Gop
from packages.constants import DIFFERENT_DEVICE_LABEL, SAME_DEVICE_LABEL, N_GOPS_FROM_DIFFERENT_DEVICE, N_GOPS_FROM_SAME_DEVICE
from packages.common import create_custom_logger

log = create_custom_logger('dataset.py')

def download(url, folder_path):
    """Downloads a file from a given url into a given folder path if it does not exist already, otherwise it skips the download

    Args:
        url (str): The URL of the file to be downloaded
        folder_path (str): The path in which to save the file

    Raises:
        RuntimeError: If the file cannot be downloaded

    Returns:
        str: The full path of the downloaded file
    """
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

class VisionDataset():
    """VISION dataset class
     
    References the dataset https://lesc.dinfo.unifi.it/VISION/
    """

    JSON_NAME = 'dataset.json'
    log = create_custom_logger('VisionDataset', logging.WARNING)

    def __init__(self, root_path: str, download_on_init = False, shuffle = False, ignore_local_dataset: bool = False, devices: list = [], media_types: list = [], properties: list = [], extensions: list = []):
        """Constructor for the VisionDataset class

        Args:
            root_path (str): The path in which to save the dataset resources
            download_on_init (bool, optional): Choose whether to download the files on init or to wait until they are requested. Defaults to False.
            shuffle (bool, optional): Choose whether to shuffle the dataset. Defaults to False.
            ignore_local_dataset (bool, optional): Choose whether to create a new dataset regardless of the local cache. Defaults to False.
            devices (list, optional): The list of devices to include in the dataset. All are considered if not set. Defaults to [].
            media_types (list, optional): The list of media types to include in the dataset. All are considered if not set. Defaults to [].
            properties (list, optional): The list of properties to include in the dataset. All are considered if not set. Defaults to [].
            extensions (list, optional): The list of file extensions to include in the dataset. All are considered if not set. Defaults to [].

        Raises:
            FileNotFoundError: If the root_path does not exist and cannot be created
        """
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
        is_loaded = False
        if not ignore_local_dataset:
            is_loaded = self.load()

        if not is_loaded:
            self._build_dataset(devices, media_types, properties, extensions, download_on_init)

        # build a map that from a number finds the correct keys to the dict
        self.index_to_dict_map = self._build_index_to_dict_map(shuffle)

    def __len__(self):
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.length
    
    def __getitem__(self, index):
        """Returns the video metadata at the given index

        Args:
            index (int): The index of the video to retrieve

        Raises:
            RuntimeError: If the index is greater than the dataset length or if the video does not exist
        Returns:
            dict: The metadata structure object of the video
        """
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
        """Retrieves the video's properties from the URL

        Args:
            url (str): URL of the video. Must be in the format https://lesc.dinfo.unifi.it/VISION/dataset/device/media_type/property/filename

        Returns:
            tuple: Tuple containing the device, media_type, property and filename
        """
        structure = url.split('https://lesc.dinfo.unifi.it/VISION/dataset/')[1].split('/')
        device = structure[0]
        media_type = structure[1]
        property = structure[2]
        filename = structure[3]

        return device, media_type, property, filename
    
    def _build_dataset(self, devices = [], media_types = [], properties = [], extensions = [], download_on_init: bool = False) -> dict:
        """Builds the dataset structure given some filters

        Args:
            devices (list, optional): The list of devices to include in the dataset. All are considered if not set. Defaults to [].
            media_types (list, optional): The list of media types to include in the dataset. All are considered if not set. Defaults to [].
            properties (list, optional): The list of properties to include in the dataset. All are considered if not set. Defaults to [].
            extensions (list, optional): The list of file extensions to include in the dataset. All are considered if not set. Defaults to [].
            download_on_init (bool, optional): Choose whether to download the files on init or to wait until they are requested. Defaults to False.

        Raises:
            RuntimeError: If the downloaded file is not the expected one

        Returns:
            dict: The dataset structure
        """
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
                video_metadata = {}
                video_metadata['device'] = device
                video_metadata['media_type'] = media_type
                video_metadata['property'] = property
                video_metadata['url'] = url
                video_metadata['filename'] = filename
                video_metadata['name'] = filename.split('.')[0]
                video_metadata['extension'] = filename.split('.')[1]
                video_metadata['local_path'] = os.path.join(device, media_type, property)
                video_metadata['path'] = os.path.join(self.root, device, media_type, property)

                # add element to structure
                self.dataset[device].append(video_metadata)

                if download_on_init:
                    downloaded_full_path = download(video_metadata['url'], video_metadata['path'])
                    if downloaded_full_path != os.path.join(video_metadata['path'], video_metadata['filename']):
                        raise RuntimeError(f'Error while downloading {url}. Downloaded file is {downloaded_full_path} instead of {os.path.join(video_metadata["path"], video_metadata["filename"])}')

                self.length += 1
                    
        return self.dataset
    
    def _build_index_to_dict_map(self, shuffle: bool) -> list:
        """Builds a map that from a number finds the correct keys to the dict to be able to retrieve entries via an index

        Args:
            shuffle (bool): Choose whether to shuffle the dataset

        Raises:
            RuntimeError: If the index to dict map length is different from the dataset length

        Returns:
            list: The index to dict map
        """
        index_to_dict_map = []

        for device in self.dataset.keys():
            for i, video in enumerate(self.dataset[device]):
                index_to_dict_map.append({'device': device, 'video': i})
        
        if len(index_to_dict_map) != self.length:
            raise RuntimeError(f'Index to dict map length {len(index_to_dict_map)} is different from dataset length {self.length}')
        
        if shuffle:
            random.shuffle(index_to_dict_map)
        return index_to_dict_map
    
    def save(self):
        """Saves the dataset to a JSON file
        """
        filename = os.path.join(self.root, VisionDataset.JSON_NAME)
        VisionDataset.log.info(f'Saving VISION dataset to {filename}')
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=4)

    def load(self):
        """Loads the dataset from a JSON file

        Returns:
            bool: True if the dataset was loaded, False otherwise
        """
        filename = os.path.join(self.root, VisionDataset.JSON_NAME)
        if os.path.exists(filename):
            VisionDataset.log.info(f'Loading VISION dataset from {filename}')
        else:
            VisionDataset.log.warn(f'VISION dataset file {filename} does not exist, could not laod')
            return False

        with open(filename, 'r') as f:
            self.dataset = json.load(f)
            self.length = 0
            for device in self.dataset.keys():
                self.length += len(self.dataset[device])

        return True
        

class VisionGOPDataset(VisionDataset):

    log = create_custom_logger('VisionGOPDataset', logging.WARNING)

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
            download_on_init: bool = False,
            ignore_local_dataset: bool = False,
            shuffle: bool = False):
        """Constructor for the VisionGOPDataset class

        Args:
            root_path (str): The path in which to save the dataset resources
            shuffle (bool, optional): Choose whether to shuffle the dataset. Defaults to False.
            devices (list, optional): The list of devices to include in the dataset. All are considered if not set. Defaults to [].
            media_types (list, optional): The list of media types to include in the dataset. All are considered if not set. Defaults to [].
            properties (list, optional): The list of properties to include in the dataset. All are considered if not set. Defaults to [].
            extensions (list, optional): The list of file extensions to include in the dataset. All are considered if not set. Defaults to [].            
            gop_size (int, optional): The number of frames in a GOP. Defaults to 8.
            frame_width (int, optional): The width of the GOP frames. Defaults to 224.
            frame_height (int, optional): The height of the GOP frames. Defaults to 224.
            gops_per_video (int, optional): The number of GOPs extracted for each video. Defaults to 1.
            build_on_init (bool, optional): Choose whether to build the dataset on init or to lazy load it. Defaults to False.
            force_rebuild (bool, optional): Choose whether to forcibly rebuild the GOPs or whether to load the existing ones. Defaults to False.
            download_on_init (bool, optional): Choose whether to download the files on init or to wait until they are requested. Defaults to False.
            ignore_local_dataset (bool, optional): Choose whether to create a new dataset regardless of the local cache. Defaults to False.
            shuffle (bool, optional): Choose whether to shuffle the dataset. Defaults to False.

        Raises:
            ValueError: If the gop_size, frame_width, frame_height or gops_per_video are not valid
        """
        
        VisionGOPDataset.log.debug(f'Building VISION GOP dataset in {root_path}')
        super().__init__(root_path=root_path,
                         download_on_init=download_on_init,
                         ignore_local_dataset=ignore_local_dataset,
                         shuffle=shuffle,
                         devices=devices,
                         media_types=media_types,
                         properties=properties,
                         extensions=extensions)
        
        self.gop_root = os.path.join(root_path, 'VISION_GOP')
        self.gop_size = gop_size
        self.gops_per_video = gops_per_video
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.force_rebuild = force_rebuild
        self.build_on_init = build_on_init
        self.pair_dataset = []

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
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return super().__len__()
    
    def __getitem__(self, index):
        """Returns the video metadata at the given index

        Args:
            index (int): The index of the video to retrieve

        Raises:
            RuntimeError: If the index is greater than the dataset length

        Returns:
            Video: the Video object
        """
        if index >= len(self):
            raise RuntimeError(f'index {index} is greater than dataset length {len(self)}')
        
        video_metadata = super().__getitem__(index)

        return self._get_video_from_metadata(video_metadata)
    
    def get_devices(self):
        """Returns the list of devices in the dataset

        Returns:
            list: The list of devices in the dataset
        """
        return self.dataset.keys()
    
    def _get_video_from_metadata(self, video_metadata):
        """Returs the Video object from the corresponding metadata

        Args:
            video_metadata (dict): The video's metadata

        Returns:
            Video: The Video object
        """
        # if built on init with force rebuild, no need to forcibly rebuild it here
        if self.build_on_init == True and self.force_rebuild == True:
            rebuild = False
        else:
            rebuild = self.force_rebuild

        video_filename = os.path.join(video_metadata['path'], video_metadata['filename'])
      
        if 'object' not in video_metadata.keys() or rebuild:
            VisionGOPDataset.log.debug(f'Building {video_metadata["filename"]}')
            video = Video(video_filename, video_metadata['device'], self.frame_width, self.frame_height, self.gops_per_video, self.gop_size, extract_gops_on_init=True)
            video_metadata['object'] = Video.save(video, os.path.join(self.gop_root, video_metadata['local_path']))
            self.save()
   
        else:
            Video.log.info(f'Video object for {video_filename} already exists, loading it from {video_metadata["object"]}')
            video = Video.load(video_metadata['object'])
        
        return video
    

class GopPairDataset(Dataset):
    log = create_custom_logger('GopPairDataset', logging.CRITICAL)

    def __init__(
            self,
            gop_dataset: VisionGOPDataset,
            n_gops_from_same_device: int,
            n_gops_from_different_device: int,
            consider_devices: list = [],
            shuffle: bool = False
            ):
        """Constructor for the GopPairDataset class

        Args:
            gop_dataset (VisionGOPDataset): The reference to the GOP dataset
            n_gops_from_same_device (int): The number of GOPs to consider from the same device
            n_gops_from_different_device (int): The number of GOPs to consider from different devices
            consider_devices (list, optional): Filter for devices to be considered. Considers all devices if empty. Defaults to [].
            shuffle (bool, optional): Choose whether to shuffle the pairs. Defaults to False.
        """
        
        self.gop_dataset = gop_dataset
        self.pair_dataset = []
        self.n_gops_from_same_device = n_gops_from_same_device
        self.n_gops_from_different_device = n_gops_from_different_device
        self.consider_devices = consider_devices

        self._build_gop_pair_dataset()

        if shuffle:
            random.shuffle(self.pair_dataset)

    def __len__(self):
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.pair_dataset)
    
    def __getitem__(self, index: int):
        """Returns the pair of GOPs at the given index

        Args:
            index (int): The index

        Raises:
            RuntimeError: If the index is greater than the dataset length

        Returns:
            tuple: The pair of GOPs and the label
        """
        if index >= len(self.pair_dataset):
            raise RuntimeError(f'index {index} is greater than dataset length {len(self.pair_dataset)}')
        
        gop1 = Gop.load(self.pair_dataset[index][0][0])
        gop2 = Gop.load(self.pair_dataset[index][1][0])
        label = self.pair_dataset[index][2]
        return (gop1, gop2, label)

    def _build_gop_pair_dataset(self):
        """Builds the GOP pair dataset

        Returns:
            list: The pair dataset
        """
        GopPairDataset.log.debug(f'Building GOP pair dataset')
        for device1 in self.gop_dataset.get_devices():
            if len(self.consider_devices) > 0 and device1 not in self.consider_devices:
                    continue
            for device2 in self.gop_dataset.get_devices():
                if len(self.consider_devices) > 0 and device2 not in self.consider_devices:
                    continue

                current_n_gops_from_different_device = 0
                current_n_gops_from_same_device = 0
                if device1 != device2:
                    while current_n_gops_from_different_device < self.n_gops_from_different_device:
                        # select two random videos from the two devices
                        random_video1_metadata = random.choice(self.gop_dataset.dataset[device1])
                        random_video2_metadata = random.choice(self.gop_dataset.dataset[device2])
                        random_video1 = self.gop_dataset._get_video_from_metadata(random_video1_metadata)
                        random_video2 = self.gop_dataset._get_video_from_metadata(random_video2_metadata)
                        # retrieve the gops from the videos
                        random_video1_gops = random_video1.get_gops()
                        random_video2_gops = random_video2.get_gops()
                        # choose a random gop from each video
                        random_gop1_index = random.randint(0, len(random_video1_gops)- 1)
                        random_gop2_index = random.randint(0, len(random_video2_gops)- 1)
                        random_gop1 = random_video1_gops[random_gop1_index]
                        random_gop2 = random_video2_gops[random_gop2_index]

                        # check if the gops are already in the set
                        for gop1, gop2, label in self.pair_dataset:
                            if (random_gop1 == gop1 or random_gop1 == gop2) and (random_gop2 == gop1 or random_gop2 == gop2):
                                GopPairDataset.log.debug(f'GOPs already in set, skipping')
                                continue

                        random_gop1_path = random_video1.get_gops_paths()[random_gop1_index]
                        random_gop2_path = random_video2.get_gops_paths()[random_gop2_index]

                        self.pair_dataset.append(((random_gop1_path, random_video1_metadata), (random_gop2_path, random_video2_metadata), DIFFERENT_DEVICE_LABEL))
                        current_n_gops_from_different_device += 1
                        GopPairDataset.log.debug(f'GOP {current_n_gops_from_different_device}/{N_GOPS_FROM_DIFFERENT_DEVICE} from different device added to set')
                else:
                    while current_n_gops_from_same_device < self.n_gops_from_same_device:
                        device = device1
                        random_video1_metadata = random.choice(self.gop_dataset.dataset[device])
                        random_video2_metadata = random.choice(self.gop_dataset.dataset[device])
                        random_video1 = self.gop_dataset._get_video_from_metadata(random_video1_metadata)
                        random_video2 = self.gop_dataset._get_video_from_metadata(random_video2_metadata)
                        random_video1_gops = random_video1.get_gops()
                        random_video2_gops = random_video2.get_gops()

                        random_gop1_index = random.randint(0, len(random_video1_gops)- 1)
                        random_gop2_index = random.randint(0, len(random_video2_gops)- 1)
                        random_gop1 = random_video1_gops[random_gop1_index]
                        random_gop2 = random_video2_gops[random_gop2_index]

                        # check if the gops are already in the set
                        for gop1, gop2, label in self.pair_dataset:
                            if (random_gop1 == gop1 or random_gop1 == gop2) and (random_gop2 == gop1 or random_gop2 == gop2):
                                GopPairDataset.log.debug(f'GOPs already in set, skipping')
                                continue

                        random_gop1_path = random_video1.get_gops_paths()[random_gop1_index]
                        random_gop2_path = random_video2.get_gops_paths()[random_gop2_index]

                        self.pair_dataset.append(((random_gop1_path, random_video1_metadata), (random_gop2_path, random_video2_metadata), SAME_DEVICE_LABEL))
                        current_n_gops_from_same_device += 1
                        GopPairDataset.log.debug(f'GOP {current_n_gops_from_same_device}/{N_GOPS_FROM_SAME_DEVICE} from same device added to set')

        return self.pair_dataset
    
