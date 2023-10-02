import os
import sys

project_path = os.getcwd()

# cd to the h264-extractor folder
lib_path = os.path.abspath(os.path.join(project_path, 'h264-extractor', 'openh264', 'info_shipout'))
if lib_path not in sys.path:
    sys.path.append(lib_path)

# from slice_pb2 import Slice
# from slice_pb2 import SliceType
