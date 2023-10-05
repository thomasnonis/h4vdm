# URL example
# https://lesc.dinfo.unifi.it/VISION/dataset/D01_Samsung_GalaxyS3Mini/videos/flatWA/D01_V_flatWA_panrot_0001.mp4

import os

def parse_url(url):
    structure = url.split('https://lesc.dinfo.unifi.it/VISION/dataset/')[1].split('/')
    device = structure[0]
    media_type = structure[1]
    property = structure[2]
    name = structure[3]

    return device, media_type, property, name

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.system(f'wget -P {path} {url}')

def get_videos_structure_from_url_file(filename):
    videos = []
    with open(filename) as f:
        urls = f.readlines()

        for url in urls:
            url = url.split('\n')[0]
            device, media_type, property, name = parse_url(url)

            if media_type == 'videos' and name.endswith('.mp4'):
                videos.append((url, device, media_type, property, name))
        
    return videos

