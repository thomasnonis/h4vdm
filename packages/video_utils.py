import os
import subprocess
import skimage.transform
import skimage.color
import numpy as np
import struct
import torch
import logging
import pickle

from slice_pb2 import Slice as SliceProto
from slice_pb2 import SliceType as SliceTypeProto

from packages.constants import MACROBLOCK_SIZE
from packages.common import create_custom_logger

# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================

def get_crop_dimensions(frame_width: int, frame_height: int, crop_width: int, crop_height: int, position = 'center'):
    """Returns the crop coordinates to crop a given frame size to a given crop size.

    Args:
        frame_width (int): The width of the original frame
        frame_height (int): The height of the original frame
        crop_width (int): The width of the desired crop
        crop_height (int): The height of the desired crop
        position (str, optional): Alignment of the cropped frame. Defaults to 'center'.

    Raises:
        NotImplementedError: If position is not 'center'.

    Returns:
        Tuple: The left, right, top and bottom coordinates of the crop
    """
    if position != 'center':
        raise NotImplementedError('Only center crop is supported for now')
    
    left_crop = (frame_width - crop_width) // 2
    right_crop = left_crop + crop_width
    top_crop = (frame_height - crop_height) // 2
    bottom_crop = top_crop + crop_height

    return (left_crop, right_crop, top_crop, bottom_crop)

def crop_frame(frame: np.ndarray, crop_width: int, crop_height: int):
    """Crops a given frame to a given size.

    Args:
        frame (np.ndarray): The frame to be cropped
        crop_width (int): The width of the desired crop
        crop_height (int): The height of the desired crop

    Raises:
        ValueError: If crop dimensions are not valid or if the frame has an unexpected number of dimensions

    Returns:
        np.ndarray: The cropped frame
    """
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # crop to center of frame
    (left_crop, right_crop, top_crop, bottom_crop) = get_crop_dimensions(frame_width, frame_height, crop_width, crop_height)

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError('Crop dimensions must be positive')
    
    if crop_width > frame_width or crop_height > frame_height:
        raise ValueError('Crop dimensions are bigger than frame dimensions')
    
    if len(frame.shape) == 3:
        return frame[top_crop:bottom_crop, left_crop:right_crop, :]
    elif len(frame.shape) == 2:
        return frame[top_crop:bottom_crop, left_crop:right_crop]
    else:
        raise ValueError('Frame has an unexpected number of dimensions')
        
def convert_image_to_tensor(image: np.ndarray):
    """Converts a given image to a tensor.

    Args:
        image (np.ndarray): The image to be converted

    Raises:
        ValueError: If the image has an unexpected number of dimensions

    Returns:
        Tensor: The converted image as a Tensor
    """
    tensor = torch.tensor((), dtype=torch.float)
    tensor = tensor.new_zeros((1, 3, image.shape[0], image.shape[1]))
    
    if len(image.shape) == 2:
        converted = torch.from_numpy(image)
        # 3-channel grayscale
        tensor[0][0] = converted
        tensor[0][1] = converted
        tensor[0][2] = converted
    elif len(image.shape) == 3:
        tensor[0] = torch.from_numpy(image).permute(2, 0, 1) # (h,w,c) -> (c,h,w)
    else:
        raise ValueError(f'Image has an unexpected number of das_tensorimensions: {len(converted.shape)}')
    
    return tensor

# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================

class H264Extractor():
    log = create_custom_logger('H264Extractor', logging.WARNING)

    def __init__(self, bin_filename, cache_dir):
        if not os.path.exists(bin_filename):
            raise FileNotFoundError(f'cannot locate the binary file "{bin_filename}", was it built?')
        self.bin_filename = bin_filename

        cache_dir = os.path.join(cache_dir, 'h264_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(os.path.join(cache_dir))
        self.cache_dir = cache_dir


    def convert_to_h264(self, video_filename):
        """converts a given mp4 video file to an h264 annex b file.
        The h264 file will be saved in the cache folder provided in the constructor

        Args:
            video_filename (str): full path to the mp4 video file

        Raises:
            FileNotFoundError: if the h264 is not generated correctly

        Returns:
            str: full path to the h264 file
        """
        video_name = os.path.basename(video_filename)

        h264_filename = os.path.join(self.cache_dir, video_name.split('.')[0] + '.h264')

        if os.path.exists(h264_filename):
            H264Extractor.log.info(f'H264 file for video {video_name} already exists at {h264_filename}, skipping conversion')
            return h264_filename
        
        H264Extractor.log.debug(f'Converting video {video_name} to h264')

        # extract h264 from the mp4 file using ffmpeg
        cp = subprocess.run(
                ['ffmpeg', '-y', '-i', video_filename, '-vcodec', 'copy', '-an', 
                '-bsf:v', 'h264_mp4toannexb', h264_filename, '-loglevel', 'panic'],
                check=True
            )
        
        if not os.path.exists(h264_filename):
            raise FileNotFoundError(f'cannot locate the h264 file "{h264_filename}", it probably hasn\'t been generated')

        H264Extractor.log.debug(f'H264 file for video {video_name} generated successfully at {h264_filename}')
        return h264_filename
    
    def extract_yuv_and_codes(self, h264_filename):
        """Extracts the YUV video sequence and the encoding parameters from a given h264 file.
        The files will be saved in the cache folder provided in the constructor


        Args:
            h264_filename (str): The path of the h264 file

        Raises:
            FileNotFoundError: If the yuv and/or coded data files haven't been generated

        Returns:
            Tuple: The path of the YUV file and the path of the encoding parameters file
        """
        # compute the filenames
        video_name = os.path.basename(h264_filename).split('.')[0]
        h264_name = os.path.basename(h264_filename)
        yuv_filename = os.path.join(self.cache_dir, video_name+ '.yuv') # YUV frames
        coded_data_filename = os.path.join(self.cache_dir, video_name + '.msg') # encoding parameters

        if os.path.exists(yuv_filename) and os.path.exists(coded_data_filename):
            H264Extractor.log.info(f'YUV and coded data files for h264 file {h264_name} already exist at {yuv_filename} and {coded_data_filename}, skipping extraction')
            return (yuv_filename, coded_data_filename)

        H264Extractor.log.debug(f'Extracting YUV and coded data from h264 file {h264_name}')


        # run the extractor to get the yuv and coded data files
        cp = subprocess.run(
                [self.bin_filename, h264_filename, '--yuv_out', yuv_filename, '--info_out', coded_data_filename, '--n_threads', '0'],
                # for now, only setting threads to 0 is allowed; using other values can result in unexpected behaviors
                check=True
            )

        # raise exception if the files haven't been generated
        if not os.path.exists(yuv_filename) or not os.path.exists(coded_data_filename):
            raise FileNotFoundError(f'cannot locate the yuv and/or coded data files, they probably haven\'t been generated')
        
        H264Extractor.log.debug(f'YUV and coded data files for h264 file {h264_name} generated successfully at {yuv_filename} and {coded_data_filename}')
        return (yuv_filename, coded_data_filename)
    
    def clean_cache(self):
        """Cleans the cache directory provided in the constructor
        """
        H264Extractor.log.debug(f'Cleaning cache directory {self.cache_dir}')
        if os.path.exists(self.cache_dir):
            for files in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, files))
            # os.rmdir(self.cache_dir)

# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================

class Video():
    h264_extractor = None
    log = create_custom_logger('Video', logging.CRITICAL)

    def __init__(self, filename: str, device: str, crop_width: int, crop_height: int, target_n_gops: int, target_gop_length: int, extract_gops_on_init: bool = False):
        """Constructor for the Video class

        Args:
            filename (str): The filename of the original video
            device (str): The name of the device that generated the video
            crop_width (int): The width of the desired crop
            crop_height (int): The height of the desired crop
            target_n_gops (int): The desired number of GOPs to be extracted
            target_gop_length (int): The desired length of each GOP
            extract_gops_on_init (bool, optional): Choose whether to extract the GOPs on init or to lazy load. Defaults to False.

        Raises:
            RuntimeError: If the H264 extractor has not been set
            FileNotFoundError: If the video file cannot be located
        """
        if Video.h264_extractor == None:
            raise RuntimeError('H264 extractor not set, call Video.set_h264_extractor() before instantiating a Video object')
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f'cannot locate the video file "{filename}"')
        
        # Original file properties
        self.filename = filename
        self.path = os.path.dirname(filename)
        self.name = os.path.basename(filename).split('.')[0]
        self.extension = os.path.basename(filename).split('.')[1]
        self.device = device

        self.h264_filename = None
        self.yuv_filename = None
        self.coded_data_filename = None

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.target_n_gops = target_n_gops
        self.target_gop_length = target_gop_length
        self.gops = []
        self.gops_paths = []

        if extract_gops_on_init:
            self._extract_gops()

    @staticmethod
    def set_h264_extractor(h264_extractor):
        """Sets the H264 extractor to be used by the Video class

        Args:
            h264_extractor (H264Extractor): Reference to the H264 extractor to be used

        Returns:
            H264Extractor: The H264 extractor that has been set
        """
        Video.h264_extractor = h264_extractor
        Video.log.debug(f'H264 extractor set to {Video.h264_extractor.bin_filename}')
        return Video.h264_extractor
    
    @staticmethod
    def save(video, path):
        """Saves a given video to a given path

        Args:
            video (Video): The video to be saved
            path (str): The path where the video will be saved

        Raises:
            FileNotFoundError: If the video file was not saved correctly

        Returns:
            str: The full path of the saved video
        """
        path = os.path.join(path, video.name)

        if not os.path.exists(path):
            os.makedirs(path)

        filename = os.path.join(path, video.name + '.video')
        Video.log.info(f'Saving video to {filename}')

        gops_paths = []
        for gop in video.gops:
            Gop.log.debug(f'Saving GOP {gop.get_filename()} to {path}/gops')
            gops_paths.append(Gop.save(gop, os.path.join(path, 'gops')))

        video.gops_paths = gops_paths

        video_data = (
            video.filename,
            video.device,
            video.crop_width,
            video.crop_height,
            video.target_n_gops,
            video.target_gop_length,
            video.gops_paths,
        )

        pickle.dump(video_data, open(filename, 'wb'))
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Error while saving gop {filename}')        
        return filename
    
    @staticmethod
    def load(filename):
        """Loads a video from a given filename

        Args:
            filename (str): The filename of the video to be loaded

        Raises:
            FileNotFoundError: If the video file cannot be located

        Returns:
            Video: The video object that has been loaded
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'cannot locate the video file "{filename}"')
        
        Video.log.info(f'Loading video from {filename}')

        video_data = pickle.load(open(filename, 'rb'))
        video_filename = video_data[0]
        video_device = video_data[1]
        video_crop_width = video_data[2]
        video_crop_height = video_data[3]
        video_target_n_gops = video_data[4]
        video_target_gop_length = video_data[5]
        video_gop_paths = video_data[6]

        video = Video(video_filename, video_device, video_crop_width, video_crop_height, video_target_n_gops, video_target_gop_length, extract_gops_on_init=False)
        video.gops_paths = video_gop_paths

        for gop_path in video_gop_paths:
            video.gops.append(Gop.load(gop_path, video))

        return video

    def _decode(self):     
        """Decodes the video to extract the YUV sequence and the encoding parameters
        """
        if self.coded_data_filename == None or not os.path.exists(self.coded_data_filename) or self.yuv_filename == None or not os.path.exists(self.yuv_filename):
            self.h264_filename = Video.h264_extractor.convert_to_h264(self.filename)

        (self.yuv_filename, self.coded_data_filename) = Video.h264_extractor.extract_yuv_and_codes(self.h264_filename)

    def _extract_gops(self):
        """Extracts the GOPs from the video

        Raises:
            ValueError: If the desired number of GOPs cannot be reached

        Returns:
            list: The list of GOPs that have been extracted
        """
        if len(self.gops) == self.target_n_gops:
            Video.log.debug(f'Video {self.name} already has {self.target_n_gops} GOPs, skipping extraction')
            return self.gops
        
        if len(self.gops_paths) == self.target_n_gops:
            Video.log.debug(f'Video {self.name} already has {self.target_n_gops} GOPs, loading saved GOPs')
            for gop_path in self.gops_paths:
                self.gops.append(Gop.load(gop_path, self))
            return self.gops
        
        self._decode()

        slice_iterator = Slice.get_slice_iterator(self.coded_data_filename)

        current_frame_number = 0

        while len(self.gops) < self.target_n_gops:
            try:
                Video.log.debug(f'Extracting GOP {len(self.gops) + 1}/{self.target_n_gops} from video {self.name}, starting from frame {current_frame_number}')
                gop = Gop(video_ref=self, current_frame_number=current_frame_number, slice_iterator=slice_iterator, extract_features_on_init=False)
                # update iterator with progress of last GOP
                slice_iterator = gop.get_slice_iterator()
                current_frame_number = gop.get_current_frame_number()
                self.gops.append(gop)
            except Exception:
                raise ValueError(f'Unable to reach desired number of GOPs for video {self.name}. Target was {self.target_n_gops}, actual is {len(self.gops)}')

        return self.gops
    
    def _get_rgb_frame(self, frame_number, width, height):
        """Extracts a frame from the YUV sequence and converts it to RGB.
        It assumes a YUV420 subsampling.

        Args:
            frame_number (int): Index of the frame to be extracted
            width (int): Width of the original frame, not the desired crop
            height (int): Height of the original frame, not the desired crop

        Returns:
            np.ndarray: The desired frame in RGB format
        """
        self._decode()

        # In YUV420 format, each pixel of the Y (luma) component is represented by 1 byte, while the U and V (chroma) components are subsampled,
        # so each of them is represented by 0.25 bytes. Hence, the total size is (width * height * 1.5).
        frame_size = int(width * height * 1.5)
        
        # Color space conversion constants
        U_MAX = 0.436
        V_MAX = 0.615

        with open(self.yuv_filename, 'rb') as yuv_sequence:
            # Read the frame at the specified frame number
            yuv_sequence.seek(frame_number * frame_size)
            
            y = np.frombuffer(yuv_sequence.read(width * height), dtype=np.uint8).reshape((height, width))
            u = np.frombuffer(yuv_sequence.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))
            v = np.frombuffer(yuv_sequence.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))

            # Rescale subsampled chroma components to the same size as the luma component
            y = skimage.img_as_float32(y)
            u = skimage.transform.rescale(u, 2.0, 1, anti_aliasing=False)
            v = skimage.transform.rescale(v, 2.0, 1, anti_aliasing=False)

            # Color space conversion
            u = (u * 2 * U_MAX) - U_MAX
            v = (v * 2 * V_MAX) - V_MAX

            # Convert to RGB
            yuv_frame = np.dstack([y, u, v])
            rgb_frame = skimage.color.yuv2rgb(yuv_frame)
        return rgb_frame
    
    def get_gops(self):
        """Returns the GOPs of the video

        Returns:
            list: The list of GOPs
        """
        if len(self.gops) == 0:
            self._extract_gops()

        return self.gops
    
    def get_gops_paths(self):
        """Returns the paths of the GOPs of the video

        Raises:
            RuntimeError: If no GOPs have been saved before

        Returns:
            list: The paths of the GOPs
        """
        if len(self.gops) == 0:
            raise RuntimeError('GOPs have never been saved before')

        return self.gops_paths
    
# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
class Gop():
    log = create_custom_logger('Gop', logging.WARNING)

    def __init__(self,
                current_frame_number: int,
                video_ref: Video = None,
                slice_iterator = None,
                extract_features_on_init: bool = False,
                extract_slices_on_init: bool = True,
                intra_frame = None,
                inter_frames = None,
                frame_types = None,
                mb_types = None,
                luma_qps = None,
                video_name: str = None,
                target_gop_length: int = None,
                crop_width: int = None,
                crop_height: int = None
                ):
        """Constructor for the GOP class

        Args:
            current_frame_number (int): The frame number of this GOP
            video_ref (Video, optional): The reference to the GOP's Video object. Defaults to None.
            slice_iterator (Iterator, optional): The Slice iterator. Defaults to None.
            extract_features_on_init (bool, optional): Choose whether to extract the features on init or to lazy load them. Defaults to False.
            extract_slices_on_init (bool, optional): Choose whether to extract slices on init or to lazy load them. Defaults to True.
            intra_frame (np.ndarray, optional): Intra frame if the GOP is to be constructed from existing features. Defaults to None.
            inter_frames (List[np.ndarray], optional): Inter frames if the GOP is to be constructed from existing features. Defaults to None.
            frame_types (List[int], optional): List of frame types if the GOP is to be constructed from existing features. Defaults to None.
            mb_types (List[np.ndarray], optional): List of macroblock type images if the GOP is to be constructed from existing features. Defaults to None.
            luma_qps (_type_, optional): List of luma quantization parameter images if the GOP is to be constructed from existing features. Defaults to None.
            video_name (str, optional): Name of the video. Defaults to None.
            target_gop_length (int, optional): The desired GOP length. Defaults to None.
            crop_width (int, optional): The desired crop width. Defaults to None.
            crop_height (int, optional): The desired crop height. Defaults to None.

        Raises:
            ValueError: If the object is not constructed with the correct parameters
        """
        

        if video_ref is None and video_name is not None and target_gop_length is not None and crop_width is not None and crop_height is not None:
            self.video_ref = None
            self.video_name = video_name
            self.target_gop_length = target_gop_length
            self.crop_width = crop_width
            self.crop_height = crop_height
        elif video_ref is not None and video_name is None and target_gop_length is None and crop_width is None and crop_height is None:
            self.video_ref = video_ref
            self.video_name = video_ref.name
            self.target_gop_length = video_ref.target_gop_length
            self.crop_width = video_ref.crop_width
            self.crop_height = video_ref.crop_height
        else:
            raise ValueError('Either video_ref is provided or video_name and target_gop_length are provided')

        self.current_frame_number = current_frame_number
        
        if intra_frame is None and inter_frames is None and frame_types is None and mb_types is None and luma_qps is None:
            self.is_frozen = False
            self.intra_frame = None
            self.inter_frames = []
            self.frame_types = []
            self.mb_types = []
            self.luma_qps = []
        elif intra_frame is not None and len(inter_frames) > 0 and len(frame_types) > 0 and len(mb_types) > 0 and len(luma_qps) > 0:
            self.is_frozen = True
            self.intra_frame = intra_frame
            self.inter_frames = inter_frames
            self.frame_types = frame_types
            self.mb_types = mb_types
            self.luma_qps = luma_qps
        else:
            raise ValueError('Either all features are provided or none of them')
              
        self.slices = []

        if not self.is_frozen:
            if slice_iterator == None:
                self.slice_iterator = Slice.get_slice_iterator(self.video_ref.coded_data_filename)
            else:
                self.slice_iterator = slice_iterator

            if extract_slices_on_init:
                self._extract_slices(video_ref.target_gop_length)

            if extract_features_on_init:
                self._extract_features(self.slices)

    @staticmethod
    def save(gop, path: str):
        """Saves a given GOP to a given path

        Args:
            gop (Gop): The GOP to be saved
            path (str): The path where the GOP will be saved

        Raises:
            FileNotFoundError: If the GOP file was not saved correctly

        Returns:
            str: The full path of the saved GOP
        """
        if not os.path.exists(path):
            os.makedirs(path)

        filename = gop.get_filename(path)
        Gop.log.info(f'Saving GOP to {filename}')

        if len(gop) == 0:
            gop._extract_slices(gop.target_gop_length)

        gop_data = (
            gop.current_frame_number,
            gop.video_name,
            gop.target_gop_length,
            gop.get_intra_frame(),
            gop.get_inter_frames(),
            gop.get_frame_types(),
            gop.get_macroblock_images(),
            gop.get_luma_qp_images(),
            gop.crop_width,
            gop.crop_height
            )

        pickle.dump(gop_data, open(filename, 'wb'))

        if not os.path.exists(filename):
            raise FileNotFoundError(f'Error while saving gop {filename}')
        
        return filename

    @staticmethod
    def load(path: str, video_ref: Video = None):
        """Loads a GOP from a given path

        Args:
            path (str): The path of the GOP file to be loaded
            video_ref (Video, optional): The reference to the corresponding Video object of the GOP. Defaults to None.

        Raises:
            FileNotFoundError: _description_

        Returns:
            _type_: _description_
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'cannot locate the video file "{path}"')
        
        Gop.log.info(f'Loading GOP from {path}')
        gop_data = pickle.load(open(path, 'rb'))

        if video_ref is None:
            gop = Gop(
                current_frame_number=gop_data[0],
                video_name=gop_data[1],
                target_gop_length=gop_data[2],
                intra_frame=gop_data[3],
                inter_frames=gop_data[4],
                frame_types=gop_data[5],
                mb_types=gop_data[6],
                luma_qps=gop_data[7],
                # TODO: will need to regenerate everything. Temporary fix
                # crop_width=gop_data[8],
                crop_width=224,
                # crop_height=gop_data[9]
                crop_height=224
            )
        else:
            gop = Gop(
                    video_ref = video_ref,
                    current_frame_number = gop_data[0],
                    intra_frame=gop_data[3],
                    inter_frames=gop_data[4],
                    frame_types=gop_data[5],
                    mb_types=gop_data[6],
                    luma_qps=gop_data[7]
                    )

        return gop

    def __eq__(self, other):
        return self.__str__() == other.__str__()
    
    def __len__(self):
        if self.is_frozen:
            return len(self.frame_types)
        else:
            return len(self.slices)

    def __getitem__():
        return NotImplementedError('__getitem__() Not implemented yet')

    def __str__(self):
        return f'{self.video_name}_{self.current_frame_number - self.target_gop_length + 1}'

    def _extract_slices(self, target_length: int) -> list:
        """Extracts the desired number of slices from the video

        Args:
            target_length (int): The desired number of slices to be extracted

        Raises:
            RuntimeError: If the GOP has been loaded from a file and has not be directly generated
            ValueError: If the desired number of slices cannot be reached

        Returns:
            list: The list of slices that have been extracted
        """
        # each video has x gops and each gop has y slices
        if self.is_frozen:
            raise RuntimeError('GOP is frozen, cannot extract slices')
        
        slices = []
        is_first_iteration = True
        while len(slices) < target_length:
            try:
                slice_raw = next(self.slice_iterator)
                if self.current_frame_number == 0 and is_first_iteration:
                    is_first_iteration = False
                else:
                    self.current_frame_number += 1
            except StopIteration:
                Gop.log.critical(f'next() raised StopIteration, Video is over at frame {self.current_frame_number}')
                break

            if len(slices) == 0 and slice_raw.type != SliceTypeProto.I:
                Gop.log.debug(f'Found slice of type {SliceTypeProto.Name(slice_raw.type)} at frame {self.current_frame_number}, skipping')
                continue

            if slice_raw.type == SliceTypeProto.I:
                Gop.log.debug(f'Found slice of type {SliceTypeProto.Name(slice_raw.type)} at frame {self.current_frame_number}, initializing GOP')
                # reset gop if intra frame is found
                slices = []
            
            Gop.log.debug(f'Appending slice of type {SliceTypeProto.Name(slice_raw.type)} at frame {self.current_frame_number}')
            slices.append(Slice(slice_raw, self.current_frame_number, self))

        if len(slices) != target_length:
            raise ValueError(f'Unable to reach desired GOP length of {target_length}, actual gop length is {len(slices)}')
        
        Gop.log.debug(f'GOP succesfully extracted from frames {self.current_frame_number - target_length + 1} - {self.current_frame_number}')
        self.slices = slices
        return slices

    def _extract_features(self):
        """Extracts the features from the slices

        Raises:
            RuntimeError: If the GOP has been loaded from a file and has not be directly generated
            ValueError: If the GOP has not been extracted yet

        Returns:
            None: None
        """
        if self.is_frozen:
            raise RuntimeError('GOP is frozen, cannot extract features')
        
        if len(self) == 0 or self.slices is None:
            raise ValueError('GOP not extracted yet')
        
        Gop.log.debug(f'Extracting features for GOP of video {self.video_name} starting from frame {self.current_frame_number}')

        for slice in self.slices:
            # append frame type
            self.frame_types.append(slice.get_type())

            # append intra frame
            if slice.is_intra():
                self.intra_frame = slice.get_rgb_frame()
                # include difference between I frame and itself (zeros)
                self.inter_frames.append(self.intra_frame - self.intra_frame)
            # append inter frame
            else:
                self.inter_frames.append(slice.get_rgb_frame() - self.intra_frame) # abs()?
            
            # create macroblock image, where each pixel is the type of the macroblock it belongs to
            self.mb_types.append(slice.get_macroblock_image())
            
            # create luma quantization parameter image, where each pixel is the luma qp of the macroblock it belongs to
            self.luma_qps.append(slice.get_luma_qp_image())

        return None
    
    def get_slice_iterator(self):
        """Returns the slice iterator of the GOP

        Raises:
            RuntimeError: If the GOP has been loaded from a file and has not be directly generated

        Returns:
            Iterator: The slice iterator
        """
        if self.is_frozen:
            raise RuntimeError('GOP is frozen, cannot get slice iterator')
        return self.slice_iterator
    
    def get_current_frame_number(self):
        """Gets the current frame number of the slice iterator

        Returns:
            int: The frame number
        """
        return self.current_frame_number
    
    def get_first_frame_number(self):
        """Gets the frame number of the first frame of the GOP

        Returns:
            int: The frame number of the first frame of the GOP
        """
        return self.current_frame_number - len(self.slices) + 1
    
    def get_filename(self, path: str = None):
        """Returns the filename of the GOP.
        If the path is given, it returns the full path of the GOP

        Args:
            path (str, optional): The path in which to build the filenmae. Defaults to None.

        Returns:
            str: The filename
        """
        if path is None:
            return self.video_name + f'_{self.current_frame_number - self.target_gop_length + 1}.gop'

        return os.path.join(path, self.video_name) + f'_{self.current_frame_number - self.target_gop_length + 1}.gop'
    
    def get_rgb_frame(self, frame_number: int):
        """Returns the RGB frame of the GOP at a given frame number

        Args:
            frame_number (int): The frame number

        Raises:
            RuntimeError: If the GOP has been loaded from a file and has not be directly generated
            ValueError: If the frame number is not in the GOP range

        Returns:
            np.ndarray: The RGB frame
        """
        if self.is_frozen:
            raise RuntimeError('GOP is frozen, cannot get rgb frame')
        
        if frame_number < self.get_first_frame_number() or frame_number > self.get_current_frame_number():
            raise ValueError(f'Frame number {frame_number} is not in the GOP range [{self.get_first_frame_number()}, {self.get_current_frame_number()}]')
        
        return self.video_ref._get_rgb_frame(frame_number, self.slices[0].original_width, self.slices[0].original_height)
    
    def get_intra_frame(self):
        if self.intra_frame is None:
            self._extract_features()
        return self.intra_frame
    
    def get_intra_frame_as_tensor(self):
        frame = self.get_intra_frame()
        return convert_image_to_tensor(frame)
    
    def get_inter_frames(self):
        if len(self.inter_frames) == 0:
            self._extract_features()
        return self.inter_frames
    
    def get_inter_frames_as_tensor(self):
        frames = self.get_inter_frames()
        tensors = []
        for frame in frames:
            tensors.append(convert_image_to_tensor(frame))

        return torch.cat(tensors, dim=0)
    
    def get_frame_types(self):
        if len(self.frame_types) == 0:
            self._extract_features()
        return self.frame_types
    
    def get_frame_types_as_tensor(self):
        frame_types = self.get_frame_types()
        return torch.tensor(frame_types)
    
    def get_macroblock_images(self):
        if len(self.mb_types) == 0:
            self._extract_features()
        return self.mb_types
    
    def get_macroblock_images_as_tensor(self):
        frames = self.get_macroblock_images()
        tensors = []
        for frame in frames:
            tensors.append(convert_image_to_tensor(frame))

        return torch.cat(tensors, dim=0)
    
    def get_luma_qp_images(self):
        if len(self.luma_qps) == 0:
            self._extract_features()
        return self.luma_qps
    
    def get_luma_qp_images_as_tensor(self):
        frames = self.get_luma_qp_images()
        tensors = []
        for frame in frames:
            tensors.append(convert_image_to_tensor(frame))

        return torch.cat(tensors, dim=0)
        
    def is_same_device(self, other_gop) -> bool:
        """Returns whether the GOP is from the same device as another GOP

        Args:
            other_gop (Gop): The other GOP

        Returns:
            bool: Whether the GOP is from the same device as the other GOP
        """
        return self.video_properties['device'] == other_gop.video_properties['device']
    
# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
    
class Slice:

    log = create_custom_logger('Slice', logging.WARNING)

    def __init__(self, slice, frame_number: int, gop_ref: Gop):
        """Constructor for the Slice class

        Args:
            slice (Slice): Protobuffer slice object
            frame_number (int): The frame number of the slice
            gop_ref (Gop): Reference to the corresponding GOP object
        """
        if slice is None:
            self.type = None
            self.macroblocks = None
            self.original_width = None
            self.original_height = None
        else:
            self.type = slice.type
            self.macroblocks = slice.mbs
            self.original_width = slice.width
            self.original_height = slice.height
        
        self.crop_width = gop_ref.crop_width
        self.crop_height = gop_ref.crop_height
        
        self.frame_number = frame_number
        self.video_ref = gop_ref.video_ref

        self.rgb_frame = None
        self.luma_qp_image = None
        self.macroblock_image = None

    @staticmethod
    def _get_ep_file_iterator(coded_data_filename: str):
        """Gets an iterator for the encoding parameters file of the video

        Args:
            coded_data_filename (str): The filename of the encoding parameters file

        Yields:
            Iterator: The iterator for the encoding parameters file
        """
        with open(coded_data_filename, 'rb') as file:
            file_size = os.stat(coded_data_filename).st_size
            while file.tell() < file_size:
                length_bytes = file.read(4)
                # Interpret data as little-endian unsigned int to convert from C layer to Python
                length = struct.unpack('<I', length_bytes)[0]
                yield file.read(length)

    @staticmethod
    def get_slice_iterator(coded_data_filename: str):
        """Gets an iterator for the slices of the video

        Args:
            coded_data_filename (str): The filename of the encoding parameters file

        Yields:
            Slice: The slice object
        """
        iterator = Slice._get_ep_file_iterator(coded_data_filename)
        for bytes in iterator:
            slice = SliceProto()
            slice.ParseFromString(bytes)
            yield slice
    
    @staticmethod
    def save(slice, path: str):
        """Saves a given slice to a given path

        Args:
            slice (Slice): The slice to be saved
            path (str): The path where the slice will be saved

        Raises:
            FileNotFoundError: If the slice file was not saved correctly

        Returns:
            str: The full path of the saved slice
        """
        if not os.path.exists(path):
            os.makedirs(path)

        slice._extract_features()

        filename = os.path.join(path, slice.video_ref.name) + f'_{slice.frame_number}.slice'
        Slice.log.info(f'Saving slice to {filename}')

        slice_data = (
            slice.type,
            slice.original_width,
            slice.original_height,
            slice.crop_width,
            slice.crop_height,
            slice.frame_number,
            slice.rgb_frame,
            slice.luma_qp_image,
            slice.macroblock_image
            )

        pickle.dump(slice_data, open(filename, 'wb'))
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Error while saving gop {filename}')  
        return filename
    
    @staticmethod
    def load(filename, gop_ref: Gop):
        """Loads a slice from a given filename

        Args:
            filename (str): The filename of the slice to be loaded
            gop_ref (Gop): The reference to the corresponding GOP object

        Raises:
            FileNotFoundError: If the slice file cannot be located

        Returns:
            Slice: The slice object that has been loaded
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'cannot locate the video file "{filename}"')
        
        Slice.log.info(f'Loading slice from {filename}')
        slice_data = pickle.load(open(filename, 'rb'))
        slice = Slice(None, slice_data[5], gop_ref)
        slice.type = slice_data[0]
        slice.original_width = slice_data[1]
        slice.original_height = slice_data[2]
        slice.crop_width = slice_data[3]
        slice.crop_height = slice_data[4]
        slice.rgb_frame = slice_data[6]
        slice.luma_qp_image = slice_data[7]
        slice.macroblock_image = slice_data[8]
        return slice


    def get_type(self):
        """Gets the type of the slice

        Returns:
            int: The type of the slice
        """
        return self.type

    def is_intra(self) -> bool:
        """Returns whether the slice is an intra slice

        Returns:
            bool: True if the slice is an intra slice, False otherwise
        """
        return self.get_type() == SliceTypeProto.I

    def get_rgb_frame(self):
        if self.rgb_frame is None:
            self.rgb_frame = self.video_ref._get_rgb_frame(self.frame_number, self.original_width, self.original_height)
            self.rgb_frame = crop_frame(self.rgb_frame, self.crop_width, self.crop_height)
        
        return self.rgb_frame
    
    def _extract_features(self):
        """Extracts the features from the slice
        """
        luma_qp_image = np.zeros((self.original_height, self.original_width))
        macroblock_image = np.zeros((self.original_height, self.original_width))
        for mb in self.macroblocks:
            luma_qp_image[mb.y * MACROBLOCK_SIZE:mb.y * MACROBLOCK_SIZE + MACROBLOCK_SIZE, mb.x * MACROBLOCK_SIZE:mb.x * MACROBLOCK_SIZE + MACROBLOCK_SIZE] = mb.luma_qp
            macroblock_image[mb.y * MACROBLOCK_SIZE:mb.y * MACROBLOCK_SIZE + MACROBLOCK_SIZE, mb.x * MACROBLOCK_SIZE:mb.x * MACROBLOCK_SIZE + MACROBLOCK_SIZE] = mb.type

        self.luma_qp_image = crop_frame(luma_qp_image, self.crop_width, self.crop_height)
        self.macroblock_image = crop_frame(macroblock_image, self.crop_width, self.crop_height)
        return
    
    def get_luma_qp_image(self):
        # create luma quantization parameter image, where each pixel is the luma qp of the macroblock it belongs to
        if self.luma_qp_image is None:
            self._extract_features()
        
        return self.luma_qp_image
    
    def get_macroblock_image(self):
        # create macroblock image, where each pixel is the type of the macroblock it belongs to
        if self.macroblock_image is None:
            self._extract_features()

        return self.macroblock_image

    def get_rgb_frame_as_tensor(self):
        frame = self.get_rgb_frame()

        frame_tensor = torch.tensor((), dtype=torch.float, requires_grad=True)
        frame_tensor = frame_tensor.new_zeros((1, frame.shape[2], frame.shape[0], frame.shape[1]))
        frame_tensor[0] = torch.from_numpy(self.intra_frame).permute(2, 0, 1) # (h,w,c) -> (c,h,w)
        return frame_tensor
    
    # def get_frame_types_as_tensor(self):
    #     if self.frame_types is None:
    #         raise ValueError('GOP not extracted yet')
        
    #     frame_types_tensor = torch.tensor(self.frame_types)
    #     return frame_types_tensor

    def get_macroblock_image_as_tensor(self):
        macroblock_image = self.get_macroblock_image()
        
        mb_image_tensor = torch.tensor((), dtype=torch.float, requires_grad=True)
        mb_image_tensor = mb_image_tensor.new_zeros((1, 3, macroblock_image.shape[0], macroblock_image.shape[1]))
        converted = torch.from_numpy(macroblock_image)

        # image is grayscale, but we need 3 channels for the network. Copy the grayscale image to all 3 channels
        mb_image_tensor[0][0] = converted
        mb_image_tensor[0][1] = converted
        mb_image_tensor[0][2] = converted

        return mb_image_tensor
    
    def get_luma_qps_as_tensor(self):
        luma_qps_image = self.get_luma_qp_image()
        
        luma_qps_tensor = torch.tensor((), dtype=torch.float, requires_grad=True)
        luma_qps_tensor = luma_qps_tensor.new_zeros((1, 3, luma_qps_image.shape[0], luma_qps_image.shape[1]))
        converted = torch.from_numpy(luma_qps_image)

        # image is grayscale, but we need 3 channels for the network. Copy the grayscale image to all 3 channels
        luma_qps_tensor[0][0] = converted
        luma_qps_tensor[0][1] = converted
        luma_qps_tensor[0][2] = converted

        return luma_qps_tensor