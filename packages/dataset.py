import os
import random
from torch.utils.data import Dataset

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

    def __init__(self, root, download_on_init = False, shuffle = False, devices = [], media_types = [], properties = [], extensions = []):
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

if __name__ == '__main__':
    from pprint import pprint
        
    dataset = VisionDataset(root='./datasets/tmp', download_on_init=False, media_types = ['images'], devices=['D01_Samsung_GalaxyS3Mini'])

    pprint(dataset[-1])