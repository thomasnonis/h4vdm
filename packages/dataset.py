import os
import random
import pickle
from torch.utils.data import Dataset
from packages.video_utils import H264Extractor, VideoHandler, Gop

def download(url, folder_path):
    filename = url.split('/')[-1]
    full_path = os.path.join(folder_path, filename)
    if os.path.exists(full_path):
        # print(f'File {filename} already exists in {os.path.join(folder_path, filename)}, skipping download')
        return full_path
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    os.system(f'wget -P {folder_path} {url}')
    if not os.path.exists(full_path):
        raise RuntimeError(f'Error while downloading {url}')
    return full_path
class VisionDataset(Dataset):
    """VISION dataset (https://lesc.dinfo.unifi.it/VISION/)
    """

    def __init__(self, root: str, download_on_init = False, shuffle = False, devices: list = [], media_types: list = [], properties: list = [], extensions: list = []):
        if not os.path.exists(root):
            try:
                os.mkdir(root)
            except FileNotFoundError:
                raise FileNotFoundError(f'VISION dataset folder {root} does not exist and cannot be created')
        
        self.root = root
        self.length = 0
        self.dataset = self._build_dataset(devices, media_types, properties, extensions, download_on_init)

        # build a map that from a number finds the correct keys to the dict
        self.index_to_dict_map = self._build_index_to_dict_map(shuffle)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if index > self.length:
            raise RuntimeError(f'index {index} is greater than dataset length {self.length}')
        
        index_map = self.index_to_dict_map[index]
        element = self.dataset[index_map['device']][index_map['media_type']][index_map['property']][index_map['element']]

        if not os.path.exists(os.path.join(element['path'], element['filename'])):
            downloaded_full_path = download(element['url'], element['path'])
            if downloaded_full_path != os.path.join(element['path'], element['filename']):
                raise RuntimeError(f'Error while downloading {element["url"]}. Downloaded file is {downloaded_full_path} instead of {os.path.join(element["path"], element["filename"])}')
            
        return element
    
    def _parse_url(self, url):
        structure = url.split('https://lesc.dinfo.unifi.it/VISION/dataset/')[1].split('/')
        device = structure[0]
        media_type = structure[1]
        property = structure[2]
        filename = structure[3]

        return device, media_type, property, filename
    
    def _build_dataset(self, devices = [], media_types = [], properties = [], extensions = [], download_on_init: bool = False) -> dict:
        elements = {}
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
                if device not in elements:
                    elements[device] = {}
                if media_type not in elements[device]:
                    elements[device][media_type] = {}
                if property not in elements[device][media_type]:
                    elements[device][media_type][property] = {}
                if filename not in elements[device][media_type][property]:
                    elements[device][media_type][property][filename] = {}
                    
                # instantiate entry
                element = elements[device][media_type][property][filename]
                element['device'] = device
                element['media_type'] = media_type
                element['property'] = property
                element['url'] = url
                element['filename'] = filename
                element['name'] = filename.split('.')[0]
                element['extension'] = filename.split('.')[1]
                element['local_path'] = os.path.join(device, media_type, property)
                element['path'] = os.path.join(self.root, device, media_type, property)

                if download_on_init:
                    downloaded_full_path = download(element['url'], element['path'])
                    if downloaded_full_path != os.path.join(element['path'], element['filename']):
                        raise RuntimeError(f'Error while downloading {url}. Downloaded file is {downloaded_full_path} instead of {os.path.join(element["path"], element["filename"])}')

                self.length += 1
                    
        return elements
    
    def _build_index_to_dict_map(self, shuffle: bool) -> list:
        index_to_dict_map = []

        for device in self.dataset.keys():
            for media_type in self.dataset[device].keys():
                for property in self.dataset[device][media_type].keys():
                    for element in self.dataset[device][media_type][property].keys():
                        index_to_dict_map.append({'device': device, 'media_type': media_type, 'property': property, 'element': element})
        
        if len(index_to_dict_map) != self.length:
            raise RuntimeError(f'Index to dict map length {len(index_to_dict_map)} is different from dataset length {self.length}')
        
        if shuffle:
            random.shuffle(index_to_dict_map)
        return index_to_dict_map

class VisionGOPDataset(Dataset):
    def __init__(self, root: str, h264_extractor: H264Extractor, vision_dataset: VisionDataset, gop_size: int = 8, frame_width: int = 224, frame_height: int = 224, gops_per_video: int = 1, build_on_init: bool = False, force_rebuild: bool = False):
        self.root = root
        self.h264_extractor = h264_extractor
        self.vision_dataset = vision_dataset
        self.gop_size = gop_size
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.force_rebuild = force_rebuild

        if gop_size < 1:
            raise ValueError('gop_size must be greater than 0')
        
        if frame_width < 1 or frame_height < 1:
            raise ValueError('frame_width and frame_height must be greater than 0')
        
        if gops_per_video < 1:
            raise ValueError('gops_per_video must be greater than 0')

        if gops_per_video > 1:
            raise NotImplementedError('gops_per_video > 1 is not implemented yet')

        if build_on_init:
            for i in range(len(self.vision_dataset)):
                vision_dataset[i]['gop_full_filenames'] = [self._build_gop(self.vision_dataset[i], force_rebuild)]

    def __len__(self):
        return len(self.vision_dataset)
    
    def __getitem__(self, index):
        if index > len(self):
            raise RuntimeError(f'index {index} is greater than dataset length {self.length}')
        
        video_dict = self.vision_dataset[index]
        video_dict['gop_full_filenames'] = [self._build_gop(video_dict, self.force_rebuild)]

        return video_dict

    def _build_gop(self, video_dict: dict, force_rebuild: bool):
        gop_save_path = os.path.join(self.root, video_dict['device'], video_dict['media_type'], video_dict['property'])
        gop_filename = video_dict['name'] + f'_{self.gop_size}_{self.frame_width}x{self.frame_height}.gop'
        gop_full_filename = os.path.join(gop_save_path, gop_filename)

        if os.path.exists(gop_full_filename) and force_rebuild == False:
            return gop_full_filename

        # check if original video exists
        mp4_filename = os.path.join(video_dict['path'], video_dict['filename'])
        if not os.path.exists(mp4_filename):
            raise RuntimeError(f'File {mp4_filename} does not exist')
        
        # convert original video to h264
        h264_filename = self.h264_extractor.convert_to_h264(mp4_filename)
        yuv_filename, coded_data_filename = self.h264_extractor.extract_yuv_and_codes(h264_filename)

        # build gop
        video_handler = VideoHandler(mp4_filename, h264_filename, yuv_filename, coded_data_filename)

        gop = Gop(video_handler, self.gop_size, self.frame_width, self.frame_height)

        # remove temporary files
        os.remove(h264_filename)
        os.remove(yuv_filename)
        os.remove(coded_data_filename)

        if os.path.exists(h264_filename) or os.path.exists(yuv_filename) or os.path.exists(coded_data_filename):
            raise Exception(f'Error removing files {h264_filename}, {yuv_filename}, {coded_data_filename}')

        # save gop
        if not os.path.exists(gop_save_path):
            os.makedirs(gop_save_path)

        pickle.dump(gop, open(gop_full_filename, 'wb'))
        if not os.path.exists(gop_full_filename):
            raise FileNotFoundError(f'Error while saving gop {gop_full_filename}')
        
        return gop_full_filename