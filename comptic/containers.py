"""
Copyright 2017 Waller Lab
The University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from tifffile import imsave
import numpy as np
import json, datetime, glob, tifffile, os, copy
from llops.display import objToString, Color
import llops as yp
from .metadata import Metadata
from comptic.camera import demosaic


VERSION = 0.3

# Flag to
STRICT_METADATA_FORM = False


def printTiffTags(tif):
    """
    Function to print tags in a TIFF image
    """
    for page in tif:
        for tag in page.tags.values():
            t = tag.name, tag.value
            print(t)

def writeMultiPageTiff(file_name, data, compression=0, bit_depth=16):
    """ Helper function to write a multi-page tiff file """
    # Reference: https://github.com/scivision/pyimagevideo/blob/master/Image_write_multipage.py

    if bit_depth == 16:
        tifffile.imsave(file_name, np.uint16(data), compress=compression)
    elif bit_depth == 8:
        tifffile.imsave(file_name, np.uint8(np.round(data / np.max(data) * 256.)), compress=compression)
    else:
        raise ValueError('Invalid bit depth')


def readMultiPageTiff(file_name, tags_only=False, frame_subset=None):
    """
    Helper function to read a multi-page tiff file
    """
    with tifffile.TiffFile(str(file_name)) as tif:
        if tags_only:
            printTiffTags(tif)
        else:
            stack = tif.asarray(key=frame_subset)
            if np.ndim(stack) is not 3:
                return [stack]
            else:
                return [frame for frame in tif.asarray(key=frame_subset)]


def isDataset(directory):
    """Checks if a given directory contains a single dataset."""
    file_list = os.listdir(directory)

    tif_found = any(['.tif' in file for file in file_list])
    json_found = any(['.json' in file for file in file_list])

    return tif_found and json_found


class Dataset():

    """
    This is a dataset class as used by acquisition functions
    """

    def __init__(self, dataset_path=None,
                 dtype=None, backend=None,
                 divide_background=False,
                 subtract_mean_dark_current=False,
                 use_median_filter=False,
                 median_filter_threshold=1000,
                 force_type=None,
                 use_calibration=True):

        # Initialize metadata
        self.metadata = Metadata()

        # Store dataset path
        self.directory = dataset_path

        # Store dtype and backend
        self.dtype = dtype if dtype is not None else yp.config.default_dtype
        self.backend = backend if backend is not None else yp.config.default_backend

        # Initialize data
        self._frame_list = None
        self._frame_state_list = None
        self.frame_mask = None
        self.channel_mask = None

        # Background, containing artifacts due to system such as illumination
        # and dust on optical elements
        self.background = None
        # Decide whether we need to perform background removal
        self.divide_background = divide_background

        # Dark current, which captures the signal when there is no illumination
        self.dark_current = None
        self.subtract_mean_dark_current = subtract_mean_dark_current

        # Define version
        self.version = VERSION
        
        # Ground truth
        self.ground_truth = None

        # Median filter parameters
        self.median_filter_threshold = median_filter_threshold
        self.use_median_filter = use_median_filter

        # Initialize frame state container
        self.frame_state_container = None

        # Load dataset information if path is provided
        if self.directory is not None:

            # Load dataset metadata
            self.loadMetadata()

            # Force type if requested
            if force_type is not None:
                self.metadata.type = force_type

            # Load background and dark current
            self.loadBackground()

            # Load first frame
            self.loadFrames(frame_mask_override=[0])

            # Load calibration parameters if present in dataset directory
            if use_calibration:
                self.loadCalibration()

            # motion deblur specific functions
            if self.metadata.type in ['motion_deblur', 'stop_and_stare']:
                self.motiondeblur = MotionDeblurFunctions(self)

    def __getitem__(self, item):
        if self._frame_list[self.frame_mask[item]] is None:
            self.loadFrames(frame_mask_override=[self.frame_mask[item]])

        if item >= len(self.frame_mask):
            raise IndexError("Index out of range")

        # Subsample frame list
        if self.channel_mask is None:
            frame = self._frame_list[item]
        elif len(self.channel_mask) == 1:
            frame = self._frame_list[item][:, :, self.channel_mask[0]]
        else:
            frame = self._frame_list[item][:, :, self.channel_mask]

        return (frame, self.frame_state_list[item])

    def __len__(self):
        if self.frame_mask is not None:
            return(len(self.frame_mask))
        else:
            return len(self._frame_list)

    @property
    def frame_shape(self):
        """Returns dimensions of each frame in dataset."""
        if any([frame is not None for frame in self._frame_list]):
            frame_shape = list(yp.shape([frame for frame in self._frame_list if frame is not None][0]))

            if self.channel_mask and len(self.channel_mask) == 1:
                frame_shape = frame_shape[:2]
        else:
            # Load first frame
            self.loadFrames(frame_mask_override=[0])

            # Get shape
            frame_shape = list(yp.shape(self._frame_list[0]))

        # Check if channel_mask is used - if so, set the color (third) dimension to be the correct length
        if self.channel_mask is not None and len(self.channel_mask) > 1:
            # Set correct length
            frame_shape.append(len(self.channel_mask))

        return tuple(frame_shape)

    @property
    def frame_mask(self):
        """Returns a list of current frames which are to be used."""
        if self._frame_mask is not None:
            return self._frame_mask
        else:
            return self.frame_mask_full

    @frame_mask.setter
    def frame_mask(self, new_frame_mask):
        """Sets the current frame last. Input should be a list."""
        self._frame_mask = new_frame_mask

    @property
    def frame_mask_full(self):
        """Returns list of all possible frame numbers in the dataset."""
        return list(range(len(self._frame_list)))

    @property
    def shape(self):
        """Returns the shape of the dataset. First dimension is the number of elements in .frame_list, all other elements are the shape of each frame."""
        return tuple([len(self.frame_mask)] + list(self.frame_shape))

    @property
    def frame_count(self):
        """Returns the number of frames in thhe dataset."""
        return len(self)

    @property
    def frame_list(self):
        """Returns a list of frames in the dataset. Also loads each frame as required."""
        # Select subset of frame list if defined

        # Ensure all frames are populated
        if any([self._frame_list[index] is None for index in self.frame_mask]):
            self.loadFrames(frame_mask_override=[index for index in self.frame_mask if self._frame_list[index] is None])

        # Subsample frame list
        if self.channel_mask is None:
            return [self._frame_list[i] for i in self.frame_mask]
        elif len(self.channel_mask) == 1:
            if yp.ndim(self._frame_list[0]) == 3:
                return [self._frame_list[i][:, :, self.channel_mask[0]] for i in self.frame_mask]
            else:
                return [self._frame_list[i] for i in self.frame_mask]
        else:
            return [self._frame_list[i][:, :, self.channel_mask] for i in self.frame_mask]

    @frame_list.setter
    def frame_list(self, new_frame_list):
        """Allows assigning frames in the dataset."""

        # Generate frame list if it doesn't already exist
        if not self._frame_list:
            self._frame_list = [None] * len(new_frame_list)
            self._frame_mask_full = list(range(len(new_frame_list)))

        # Assign new frames
        if self._frame_mask is not None:
            for index, frame_index in enumerate(self.frame_mask):
                self._frame_list[frame_index] = yp.dcopy(new_frame_list[index])
        else:
            for i in range(len(new_frame_list)):
                self._frame_list[i] = yp.dcopy(new_frame_list[i])

    @property
    def frame_state_list(self):
        """Returns a list of frame states, which is a free-form variable which is typically over-ridden."""
        if self.frame_mask is not None:
            return [self._frame_state_list[i] for i in self.frame_mask]
        else:
            return self._frame_state_list

    @frame_state_list.setter
    def frame_state_list(self, new_frame_state_list):
        """Returns a list of frame states, which is a free-form variable which is typically over-ridden."""
        if self._frame_state_list:
            if self.frame_mask is not None:
                for i in self.frame_mask:
                    self._frame_state_list[i] = copy.deepcopy(new_frame_state_list[i])
            else:
                for i in range(len(new_frame_state_list)):
                    self._frame_state_list[i] = copy.deepcopy(new_frame_state_list[i])
        else:
            self._frame_state_list = copy.deepcopy(new_frame_state_list)

    @property
    def frame_state_list_full(self):
        """Returns the full frame state list."""
        return self._frame_state_list

    @property
    def frame_list_full(self):
        """Returns the full frame list."""
        return self._frame_list

    def clearMask(self):
        """Clear frame mask (set to default)."""
        self.frame_mask = list(range(len(self._frame_list)))

    def save(self, dataset_path=None, header=None, tif_compression=0,
                   metadata_only=False, precision=6, bit_depth=16):
        """
        This is the workhorse save function for all data types. It will be modified as needed
        """
        json_dict = {}

        # convert structure to metadata
        json_dict['metadata'] = self.metadata.__dict__.copy()
        json_dict['metadata']['camera'] = self.metadata.camera.__dict__.copy()
        json_dict['metadata']['camera']['roi'] = self.metadata.camera.roi.__dict__.copy()
        json_dict['metadata']['sample'] = self.metadata.sample.__dict__.copy()
        json_dict['metadata']['sample']['roi'] = self.metadata.sample.roi.__dict__.copy()
        json_dict['metadata']['pupil'] = self.metadata.pupil.__dict__.copy()
        json_dict['metadata']['pupil']['state_list'] = self.metadata.pupil.state_list.__dict__.copy()
        json_dict['metadata']['focus'] = self.metadata.focus.__dict__.copy()
        json_dict['metadata']['focus']['state_list'] = self.metadata.focus.state_list.__dict__.copy()
        json_dict['metadata']['position'] = self.metadata.position.__dict__.copy()
        json_dict['metadata']['position']['state_list'] = self.metadata.position.state_list.__dict__.copy()
        json_dict['metadata']['illumination'] = self.metadata.illumination.__dict__.copy()
        json_dict['metadata']['illumination']['state_list'] = self.metadata.illumination.state_list.__dict__.copy()
        json_dict['metadata']['illumination']['spectrum'] = self.metadata.illumination.spectrum.__dict__.copy()
        json_dict['metadata']['objective'] = self.metadata.objective.__dict__.copy()
        json_dict['metadata']['system'] = self.metadata.system.__dict__.copy()
        json_dict['metadata']['background'] = self.metadata.background.__dict__.copy()
        json_dict['metadata']['background']['roi'] = self.metadata.background.roi.__dict__.copy()
        json_dict['metadata']['groundtruth'] = self.metadata.groundtruth.__dict__.copy()

        # Clean up dictionary values
        yp.sanatizeDictionary(json_dict)

        # Initialize frame state list
        json_dict['frame_state_list'] = {}

        # Default file header is the date and time
        json_dict['metadata']['file_header'] = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Append dataset type if it exists
        if json_dict['metadata']['type'] is not None:
            json_dict['metadata']['file_header'] = json_dict['metadata']['type'] + \
                '_' + json_dict['metadata']['file_header']

        # Append user header if they provide one
        if header is not None:
            json_dict['metadata']['file_header'] = header + '_' + json_dict['metadata']['file_header']

        # Generate save path
        if dataset_path is None:
            out_dir = os.getcwd().replace('\\', '/') + '/' + json_dict['metadata']['file_header'] + '/'
        else:
            out_dir = dataset_path.replace('\\', '/') + json_dict['metadata']['file_header'] + '/'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        json_dict['frame_state_list'] = applyPrecisionToFrameStateList(self.frame_state_list)

        # Write frame list
        if not metadata_only:
            if self.frame_list is not None:
                data_file_name = os.path.join(out_dir, json_dict['metadata']['file_header'] + '.tif')
                writeMultiPageTiff(data_file_name, self.frame_list, compression=tif_compression, bit_depth=bit_depth)
                print("Wrote tiff file to file: %s" % data_file_name)
            else:
                raise ValueError('ERROR - frame_list does not exist in dataset!')

            # Write background
            if self.background is not None:
                self.metadata.background.filename = 'background.tif'
                filename = os.path.join(out_dir, self.metadata.background.filename)
                writeMultiPageTiff(filename, [self.background], compression=tif_compression)
                print("Wrote background tiff file to file: %s" % filename)

            # Write dark current
            if self.dark_current is not None:
                self.metadata.background.dark_current_filename = 'dark_current.tif'
                filename = os.path.join(out_dir, self.metadata.background.dark_current_filename)
                writeMultiPageTiff(filename, [self.dark_current], compression=tif_compression)
                print("Wrote dark current tiff file to file: %s" % filename)

        # Determinbe file header
        if json_dict['metadata']['file_header'] is None:
            json_dict['metadata']['file_header'] = 'dataset'
        metadata_file_name = out_dir + json_dict['metadata']['file_header'] + '.json'

        # Sanatize json file
        yp.sanatizeDictionary(json_dict)

        # Assign directory in dataset object
        self.directory = out_dir

        # Write metadata to JSON
        with open(metadata_file_name, 'w+') as out_file:
            json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
            json.dump(json_dict, out_file, indent=4)

        return json_dict

    def saveReconstruction(self, object, parameters, label='',
                           omit_time_in_filename=False):

        # Generate reconstruction label
        time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        reconstruction_filename = 'reconstruction'

        # Add label if one is provided
        if label is not None:
            reconstruction_filename += ('_' + label)

        # Add time if user indicates
        if not omit_time_in_filename:
            reconstruction_filename += ('_' + time_str)

        # Ensure reconstructions directory exists
        reconstruction_subdirectory = os.path.join(self.directory, 'reconstructions')
        if not os.path.exists(reconstruction_subdirectory):
            os.makedirs(reconstruction_subdirectory)

        # Generate output path
        output_path = os.path.join(reconstruction_subdirectory, reconstruction_filename)

        # Cast reconstruction as an 8-bit tiff_file
        max_value, min_value = yp.max(object), yp.min(object)
        reconstruction = 255.0 * (yp.dcopy(object) - min_value) / (max_value if max_value != 0.0 else 1.0)
        reconstruction = yp.cast(reconstruction, dtype='uint8', backend='numpy')

        # Write reconstruction
        writeMultiPageTiff(output_path + '.tif',
                           reconstruction, bit_depth=8, compression=0)

        # Add max_value and min_value to reconstruction parameters
        parameters['file_scale'] = {'min_value': min_value,
                                    'max_value': max_value}

        # Store label
        parameters['label'] = label

        # Sanatize parameters (convert to lists from numpy arrays)
        yp.sanatizeDictionary(parameters)

        # Write reconstruction parameters
        with open(output_path + '.json', 'w+') as fp:
            json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
            json.dump(parameters, fp, indent=4)

        # Let the user know we saved the reconstruction
        print('Saved reconstruction to file header: %s' % output_path)

    def loadCalibration(self):
        """Load calibration information from disk."""
        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # See if there are any tif files in directory
        if os.path.isfile(self.directory + '/calibration.json'):
            with open(self.directory + '/calibration.json') as data_file:
                self.metadata.calibration = json.load(data_file)

    def saveCalibration(self):
        """Save calibration information to disk."""
        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # Assign dataset name inside calibration json
        self.metadata.calibration['dataset_header'] = self.metadata.file_header

        # Sanatize dictionary
        yp.sanatizeDictionary(self.metadata.calibration)

        # Write reconstruction parameters
        with open(self.directory + '/calibration.json', 'w+') as fp:
            json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
            json.dump(self.metadata.calibration, fp, indent=4)

    def loadBackground(self):
        """Loads background and dark current images if found in the dataset directory."""

        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # Use real background if available
        tif_list = []
        for file in glob.glob(self.directory + "/*.tif*"):
            tif_list.append(file)

        # Load background and dark current
        for tif_filename in tif_list:
            if 'background' in tif_filename or 'bg' in tif_filename:
                self.background = yp.cast(readMultiPageTiff(tif_filename)[0],
                                          dtype=self.dtype,
                                          backend=self.backend)
            elif 'dark_current' in tif_filename:
                self.dark_current = yp.cast(readMultiPageTiff(tif_filename)[0],
                                            dtype=self.dtype,
                                            backend=self.backend)

    def clearFramesFromMemory(self):
        """Erases all frames from memory."""
        for frame_index in range(len(self._frame_list)):
            self._frame_list[frame_index] = None

        from time import sleep
        # Call garbage collection routine
        for i in range(10):
            yp.garbageCollect()
            sleep(0.1)

    def getSingleFrame(self, frame_index, do_demosaic=True):
        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # See if there are any tif files in directory
        tif_list = []
        for file in glob.glob(self.directory + "/*.tif*"):
            tif_list.append(file)

        # Load frame from disk
        for tif_filename in tif_list:
            if 'background' not in tif_filename and 'dark_current' not in tif_filename:
                frame = yp.cast(yp.squeeze(readMultiPageTiff(tif_filename, frame_subset=[frame_index])[0]),
                                dtype=self.dtype, backend=self.backend)

        # Remove background (if provided)
        if self.divide_background and self.background is not None:
            frame = removeBackground(frame,
                                     self.background,
                                     max_thresh=self.metadata.background.max_thresh)

        # Remove background (if provided)
        if self.subtract_mean_dark_current and self.dark_current is not None:
            frame = frame - yp.mean(self.dark_current)

        # Demosaic (if color camera)
        if self.metadata.camera.is_color and do_demosaic:
            frame = demosaic(frame, bayer_coupling_matrix=self.metadata.camera.bayer_coupling_matrix)

        # Subsample frame list
        if self.channel_mask is None:
            return frame
        elif len(self.channel_mask) == 1:
            return frame[:, :, self.channel_mask[0]]
        else:
            return frame[:, :, self.channel_mask]

    def loadFrames(self, frame_mask_override=None):
        """
        This function loads a set of frames as indicated by the frame_mask value.
        All other frames will be set to None until this function is told to load
        them.
        """

        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # See if there are any tif files in directory
        tif_list = []
        for file in glob.glob(self.directory + "/*.tif*"):
            tif_list.append(file)

        # Load data from multi-page tiff
        frame_mask = self.frame_mask if frame_mask_override is None else frame_mask_override

        print('Loading %d frames...' % len(frame_mask))

        for tif_filename in tif_list:
            if 'background' not in tif_filename and 'bg' not in tif_filename and 'dark_current' not in tif_filename:
                updated_frame_list = readMultiPageTiff(tif_filename, frame_subset=frame_mask)

        # Update new frames
        for frame_index, new_frame in zip(frame_mask, updated_frame_list):
            self._frame_list[frame_index] = yp.cast(yp.squeeze(new_frame), self.dtype, self.backend)

        # Perform background subtraction and demosaicing if configured
        for frame_index in frame_mask:

            # Subtract dark current
            if self.subtract_mean_dark_current and self.dark_current is not None:
                self._frame_list[frame_index] -= self.dark_current
                self._frame_list[frame_index][self._frame_list[frame_index] < 0] = 0

            # Divide background
            if self.divide_background and self.background is not None:
                self._frame_list[frame_index] = removeBackground(self._frame_list[frame_index],
                                                                 self.frame_background,
                                                                 max_thresh=self.metadata.background.max_thresh)

            # Demosaic
            if self.metadata.camera.is_color:
                self._frame_list[frame_index] = demosaic(self._frame_list[frame_index],
                                                         bayer_coupling_matrix=self.metadata.camera.bayer_coupling_matrix)

            # Median filter
            if yp.mean(self._frame_list[frame_index]) < self.median_filter_threshold and self.use_median_filter:
                self._frame_list[frame_index] = yp.filter.median(self._frame_list[frame_index])

    def loadMetadata(self):
        """This function loads the metadata from a libwallerlab dataset."""
        class Bunch(object):
            """
            Helper class to convert dict to class structure
            """

            def __init__(self, adict):
                self.__dict__.update(adict)

            def __str__(self):
                return(objToString(self, text_color=Color.BLUE, use_newline=False))

        # Generate empty dataset object to populate
        metadata = Metadata()
        tif_list = []
        json_list = []

        # Get list of files in directory
        tiff_file_list = glob.glob(os.path.join(self.directory, '*'))

        # See if there are any tif files in directory
        for file in tiff_file_list:
            if 'background' not in file and 'backup' not in file and 'dark_current' not in file:
                if '.tif' in file:
                    tif_list.append(file)

        # See if there are any tif files in directory
        for file in tiff_file_list:
            if 'calibration' not in file and 'backup' not in file:
                if '.json' in file:
                    json_list.append(file)

        assert len(tif_list) == 1, "Could not find tif file in directory %s (Found %d files)" % (self.directory, len(tif_list))
        assert len(json_list) == 1, "Could not find json file!"

        # Load Json file
        with open(json_list[0]) as data_file:
            json_file = json.load(data_file)

        def replaceRoiObjects(_dict):
            for key in _dict:
                if 'roi' in key:
                    _dict[key] = {'start': (0,0), 'shape': (0,0), 'units': 'pixels'}
                elif type(_dict[key]) == dict:
                    replaceRoiObjects(_dict[key])

        # Load metadata object
        if json_file['metadata'] is not None:
            replaceRoiObjects(json_file['metadata'])
            loadDictRecursive(metadata, json_file['metadata'])

        # Convert Roi
        convertRoiRecursive(metadata)

        # Get frame state_list
        frame_state_list = json_file['frame_state_list']

        # Set metadata in dataset
        self.metadata = metadata

        # Set frame_state_list in dataset
        self._frame_state_list = frame_state_list

        # Set frame_list to list of None values (for now)
        self._frame_list = [None] * len(frame_state_list)

    def showIllum(self, cmap='gray', figsize=(5, 5), figaspect="wide", show_fourier=False):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import Slider
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.subplots_adjust(bottom=0.2)

        na_plot = NaPlot(ax, self.frame_state_list,
                         np.asarray(self.metadata.illumination.state_list.design),
                         objective_na=self.metadata.objective.na,
                         foreground_color='b')

        def update(val):
            frame_index = int(val)

            na_plot.update(val)

            # Draw
            fig.canvas.draw_idle()

        # Add slider
        ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])
        slider = Slider(ax_slider, 'Image Index', 0, len(self.frame_state_list), valinit=0, valfmt='%i')
        slider.on_changed(update)

        plt.show()
        plt.tight_layout()
        return slider

    def show(self, hardware_element=None, cmap='gray', figsize=(10,4), figaspect="wide",
             show_fourier=False, hide_sliders=False, show_iteration_plot=False,
             images_only=False, hw_type=None):

        from matplotlib.widgets import Slider
        from matplotlib import pyplot as plt

        device_list_0 = [self.metadata.illumination,
                         self.metadata.position,
                         self.metadata.focus,
                         self.metadata.pupil]

        device_list = [self.metadata.camera]  # Always include camera
        for device in device_list_0:
            # Make sure device is connected
            if device.device_name is not None:
                # Make sure device is in frame_state_list
                for hw_element in self.frame_state_list[0]:
                    if hw_element.lower() in str(device.__class__.__name__).lower():
                        if hardware_element is None or str(device.__class__.__name__).lower() == hardware_element.lower():
                            device_list.append(device)

        # Store numbr of devices (excluding camera)
        if not images_only:
            device_count = len(device_list) + int(show_iteration_plot)
        else:
            device_count = 1  # image plot

        fig, axis_list = plt.subplots(1, device_count, figsize=figsize)
        plt.subplots_adjust(bottom=0.3)
        if type(axis_list) not in (np.ndarray, list, tuple):
            axis_list = [axis_list]

        frame_index = 0
        self.current_frame_index = 0
        self.current_illumination_index = 0

        axis_index = 0
        self.plot_list = []

        if show_iteration_plot and not images_only:
            self.plot_list.append(ImageIterationPlot(axis_list[axis_index], self.frame_list))
            axis_index += 1

        self.plot_list.append(ImagePlot(axis_list[axis_index],
                                        self.frame_list,
                                        cmap=cmap,
                                        pixel_size_um=self.metadata.system.eff_pixel_size_um))
        axis_index += 1
        # for device in device_list:
        #     if device.__class__.__name__ is "Camera":
        #         self.plot_list.append(ImagePlot(axis_list[axis_index],
        #                                         self.frame_list,
        #                                         cmap=cmap,
        #                                         pixel_size_um=self.metadata.system.eff_pixel_size_um,
        #                                         roi=self.metadata.camera.roi))
        #         axis_index += 1
        #
        #     elif device.__class__.__name__  is "Illumination" and not images_only:
        #
        #         self.plot_list.append(NaPlot(axis_list[axis_index], self.frame_state_list,
        #                                                   np.asarray(self.metadata.illumination.state_list.design),
        #                                                   objective_na=self.metadata.objective.na))
        #         axis_index += 1
        #
        #     elif device.__class__.__name__  is "Position" and not images_only:
        #         self.plot_list.append(PositionPlot(axis_list[axis_index], self.frame_state_list, pixel_size_um=self.metadata.system.eff_pixel_size_um))
        #         axis_index += 1
        #
        #     elif device.__class__.__name__  is "Focus" and not images_only:
        #         raise NotImplementedError
        #
        #     elif device.__class__.__name__  is "Pupil" and not images_only:
        #
        #         raise NotImplementedError

        def update(val):
            frame_index = int(val)

            # Update
            for p in self.plot_list:
                p.update(frame_index)

            # Draw
            fig.canvas.draw_idle()

            # Update frame index
            self.current_frame_index = frame_index
            self.current_illumination_index = 0

        def updateInterFrame(val):
            illumination_index = int(val)

            # Update
            for p in self.plot_list:
                if type(p) in (NaPlot, PositionPlot):
                    p.update(self.current_frame_index, subframe_time_index=illumination_index)
            # Draw
            fig.canvas.draw_idle()

            # Update frame index
            self.current_illumination_index = illumination_index

        # Add slider
        ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, self.frame_count, valinit=frame_index, valfmt='%d')
        slider.on_changed(update)

        if hide_sliders:
            slider.ax.set_axis_off()

        def frame_update_function_handle(frame_index):
            return update(frame_index)

        def interframe_update_function_handle(interframe_index):
            return updateInterFrame(interframe_index)

        plt.show()

        return(fig, slider, frame_update_function_handle, interframe_update_function_handle)

    # Background subtracton
    def subtractBackground(self, plot_bg=False, force_subtraction=False, new_roi=False, max_thresh=None):
        from matplotlib import pyplot as plt
        if not self.metadata.background.is_subtracted or force_subtraction:

            if max_thresh is None:
                max_thresh = self.metadata.background.max_thresh

            # Convert roi to new shape if necessary
            if type(self.metadata.background.roi) is list:
                self.metadata.background.roi = yp.Roi(y_start=self.metadata.background.roi[0], y_end=self.metadata.background.roi[1],
                                                   x_start=self.metadata.background.roi[2], x_end=self.metadata.background.roi[3])

            # Convert to single-precision floating point if required
            if yp.getDatatype(self.frame_list[0]) is not 'float32':
                for i in range(len(self.frame_list)):
                    self.frame_list[i] = yp.astype(self.frame_list[i], 'float32')

            # Perform background subtraction
            if self.background is None:
                if self.metadata.background.roi is None or new_roi:
                    self.metadata.background.roi = getROI(
                        self.frame_list[0, :, :], 'Select region for background subtraction')

                # Deal with roi being whole image (default case)
                if self.metadata.background.roi.x_end is -1:
                    self.metadata.background.roi.x_end = self.frame_list[0].shape[1]
                if self.metadata.background.roi.y_end is -1:
                    self.metadata.background.roi.y_end = self.frame_list[0].shape[0]

                #  Generate list of background values
                means = sum([yp.mean(self.frame_list[i] for i in range(len(self.frame_list)))]) / len(self.frame_list)

                # Plot Background as a function of image index if user desires
                if plot_bg:
                    plt.figure()
                    plt.plot(bg_list)
                    plt.xlabel("Image index")
                    plt.ylabel("Background (counts)")
                    plt.title("Background vs image index")

                # Subtract background values only if they don't exceed sample.background_max_thresh, otherwise subtract sample.background_max_thresh
                print('Using threshold: %.2f' % max_thresh)
                for i in range(len(self.frame_list)):
                    if bg_list[img_idx] < max_thresh:
                        self.frame_list[i] -= np.double(bg_list[img_idx])
                    else:
                        self.frame_list[i] -= np.double(max_thresh)

            else:
                for frame in self.frame_list:
                    frame /= np.squeeze(self.frame_background)
            # Intensity is always positive, so enforce it
            for frame in self.frame_list:
                frame[frame < 0.] = 0.

            # Set background subtracted flag
            self.metadata.background.is_subtracted = True

            print("Finished dividing background.")
        else:
            print("Metadata indicates background has already been subtracted.")

    def applyNaTransformation(self, rotation_deg=0., scale=1., shift_na=(0., 0.), flip_xy=False, flip_x=False, flip_y=False):
        """
        This function applied a transformation to the source
        """
        # Shift matrix
        Sh = np.array([[1, 0, shift_na[0]], [0, 1, shift_na[1]], [0, 0, 1]])

        # Rotation Matrix
        R = np.array([[np.cos(rotation_deg * np.pi / 180), -np.sin(rotation_deg * np.pi / 180), 0],
                      [np.sin(rotation_deg * np.pi / 180), np.cos(rotation_deg * np.pi / 180), 0],
                      [0, 0, 1]])

        # Scaling Matrix
        Sc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1. / scale]])

        # Total Matrix
        M = np.dot(np.dot(Sh, R), Sc)

        list_of_source_lists = []
        if self.metadata.illumination.state_list.design is not None:
            list_of_source_lists.append(self.metadata.illumination.state_list.design)
        if self.metadata.illumination.state_list.calibrated is not None:
            list_of_source_lists.append(self.metadata.illumination.state_list.calibrated)

        for source_list_index in range(len(list_of_source_lists)):
            if list_of_source_lists[source_list_index] is not None:
                na = list_of_source_lists[source_list_index].copy()
                na = np.append(na, np.ones([np.size(na, 0), 1]), 1)
                na_2 = np.dot(M, na.T).T
                na_2[:, 0] /= na_2[:, 2]
                na_2[:, 1] /= na_2[:, 2]
                na_2 = na_2[:, 0:2]

                # Apply flip in x/y
                if flip_xy:
                    tmp = na_2.copy()
                    na_2[:, 0] = tmp[:, 1].copy()
                    na_2[:, 1] = tmp[:, 0].copy()

                if flip_x:
                    na_2[:, 0] *= -1

                if flip_y:
                    na_2[:, 1] *= -1

                list_of_source_lists[source_list_index] = na_2.copy()

        if self.metadata.illumination.state_list.design is not None:
            self.metadata.illumination.state_list.design = list_of_source_lists[0].copy()
        if self.metadata.illumination.state_list.calibrated is not None:
            self.metadata.illumination.state_list.calibrated = list_of_source_lists[1].copy()


def loadDictRecursive(inner_attr, inner_dict):
    for key_name in inner_dict:
        if hasattr(inner_attr, key_name):
            if type(inner_dict[key_name]) is dict:
                # recurses down a layer of dict and metadata class' attributes
                if getattr(inner_attr, key_name) is not None:
                    loadDictRecursive(getattr(inner_attr, key_name), inner_dict[key_name])
                else:
                    setattr(inner_attr, key_name, inner_dict[key_name])
            else:
                # adds to current level of object
                try:
                    setattr(inner_attr, key_name, inner_dict[key_name])
                except AttributeError:
                    print('Failed to set attribute %s' % key_name)
        else:
            if STRICT_METADATA_FORM:
                print("Skipping unknown key %s" % key_name)
            else:
                setattr(inner_attr, key_name, inner_dict[key_name])


def convertRoiRecursive(metadata):
    attr_list = dir(metadata)
    for key_name in attr_list:
        if '__' not in key_name:
            if key_name == 'roi':
                if type(getattr(metadata, key_name)) is dict:
                    roi_dict = getattr(metadata, key_name)
                    if roi_dict['y_end'] is not None:
                        new_roi = yp.Roi(start=(roi_dict['y_start'], roi_dict['x_start']),
                                         end=(roi_dict['y_end'], roi_dict['x_end']))
                        setattr(metadata, key_name, new_roi)
                    else:
                        setattr(metadata, key_name, None)
            elif hasattr(getattr(metadata, key_name), '__dict__'):
                convertRoiRecursive(getattr(metadata, key_name))

def loadLedPositonsFromJson(file_name, z_offset=0):
    """Function which loads LED positions from a json file

    Args:
        fileName: Location of file to load
        zOffset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (x, y, z) in mm
    """
    json_data = open(file_name).read()
    data = json.loads(json_data)

    source_list_cart = np.zeros((len(data['led_list']), 3))
    x = [d['x'] for d in data['led_list']]
    y = [d['y'] for d in data['led_list']]
    z = [d['z'] for d in data['led_list']]
    board_idx = [d['board_index'] for d in data['led_list']]

    source_list_cart[:, 0] = x
    source_list_cart[:, 1] = y
    source_list_cart[:, 2] = z

    illum = Illumination()

    illum.state_list.design = cartToNa(source_list_cart, z_offset=0)
    illum.source_list_rigid_groups = board_idx

    return illum

def cartToNa(point_list_cart, z_offset=0):
    """Function which converts a list of cartesian points to numerical aperture (NA)

    Args:
        point_list_cart: List of (x,y,z) positions relative to the sample (origin)
        z_offset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    point_list_cart = np.asarray(point_list_cart)
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    result = np.zeros((np.size(point_list_cart, 0), 2))
    result[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    result[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))

    return(result)


def saveImageStack(filename, img, clim=None, bit_depth=8, cmap=None):
    from matplotlib.pyplot import get_cmap
    assert bit_depth == 8 or bit_depth == 16, "saveImageStack only handles 8-bit and 16-bit images!"
    if cmap is not None:
        bit_depth = 8
        cmap = get_cmap(cmap)
    if clim is None:
        img_tiff = img - img.min()
        img_tiff /= img_tiff.max()
    elif len(clim) == 2:
        img_tiff = img - clim[0]
        img_tiff[img_tiff < 0] = 0
        img_tiff /= (clim[1] - clim[0])
        img_tiff[img_tiff > 1] = 1
    else:
        print('clim should be a tuple of length 2, i.e. (lowest_value,highest_value)!')
        raise
    img_tiff *= 2**bit_depth - 1
    if cmap is None:
        if bit_depth == 8:
            img_tiff = img_tiff.astype('uint8')
        elif bit_depth == 16:
            img_tiff = img_tiff.astype('uint16')
        if len(img.shape) == 2:
            imsave(filename, img_tiff)
        elif len(img.shape) == 3:
            imsave(filename, img_tiff[:, np.newaxis, :, :])
    else:
        assert bit_depth == 8, 'cmap only support 8-bit images!'
        img_tiff = cmap(img_tiff.astype('uint8'))
        img_tiff *= 2**bit_depth - 1
        img_tiff = img_tiff.astype('uint8')
        img_tiff = img_tiff.transpose((0, 3, 1, 2)) if len(img.shape) == 3 else img_tiff.transpose((2, 0, 1))
        img_tiff = img_tiff[:, :3, :, :] if len(img.shape) == 3 else img_tiff[:3, :, :]
        imsave(filename, img_tiff)


class MultiTiff:
    def __init__(self, file_path, size_c=-1, size_z=-1, size_t=-1, dimensionOrder=None,
                 isOmeTiff=False, returnDimOrder='CZT', crop_x=-1, crop_y=-1):
        self.file_path = file_path
        self.returnDimOrder = returnDimOrder
        self.crop_x = crop_x
        self.crop_y = crop_y

        # Create tiff object
        t = tifffile.TiffFile(file_path)

        # Get number of pages in tiff
        self.nPages = len(t)

        # Get tags as one big string
        infoStr = t.info()

        if isOmeTiff:
            # Get dimension orer from tiff metadata
            if dimensionOrder is None:
                dimensionOrderStart = infoStr.find("DimensionOrder")
                dimensionOrderEnd = infoStr.find("ID=", dimensionOrderStart, dimensionOrderStart + 200)
                self.tiffDimOrder = infoStr[(dimensionOrderStart + 18):dimensionOrderEnd - 2]
            else:
                self.tiffDimOrder = dimensionOrder

            if size_c <= 0:
                sizeCStart = infoStr.find(' SizeC')
                sizeCEnd = infoStr.find(' SizeT', sizeCStart, sizeCStart + 200)
                assert sizeCStart > 0, "No SizeC string found in tiff file"
                self.size_c = int(infoStr[sizeCStart + 8:sizeCEnd - 1])
            else:
                self.size_c = size_c

            if size_t <= 0:
                sizeTStart = infoStr.find(' SizeT')
                sizeTEnd = infoStr.find(' SizeX', sizeTStart, sizeTStart + 200)
                assert sizeTStart > 0, "No SizeT string found in tiff file"
                self.size_t = int(infoStr[sizeTStart + 8:sizeTEnd - 1])
            else:
                self.size_t = size_t

            if size_z <= 0:
                sizeZStart = infoStr.find(' SizeZ')
                sizeZEnd = infoStr.find(' Type=', sizeTStart, sizeTStart + 200)
                assert sizeZStart > 0, "No SizeZ string found in tiff file"
                self.size_z = int(infoStr[sizeZStart + 8:sizeZEnd - 1])
            else:
                self.size_z = size_z

            # Get image size
            sizeXStart = infoStr.find(' SizeX')
            sizeXEnd = infoStr.find('SizeY', sizeXStart, sizeXStart + 200)
            self.imgSizeX = int(infoStr[sizeXStart + 8:sizeXEnd - 2])

            sizeYStart = infoStr.find(' SizeY')
            sizeYEnd = infoStr.find('SizeZ', sizeXStart, sizeXStart + 200)
            self.imgSizeY = int(infoStr[sizeYStart + 8:sizeYEnd - 2])
        else:
            # Get dimension orer from tiff metadata
            if dimensionOrder is None:
                raise ValueError("Must provide dimension order for non-ome tiff.")
            else:
                self.tiffDimOrder = dimensionOrder

            if size_c < 0:
                self.size_c = self.nPages
            else:
                self.size_c = size_c

            if size_t <= 0:
                if (self.size_c == self.nPages):
                    self.size_t = 1
                else:
                    raise ValueError('Must provide time point count for non-OME tiff.')
            else:
                self.size_t = size_t

            if size_z <= 0:
                if ((self.size_c == self.nPages) | (self.size_t == self.nPages)):
                    self.size_z = 1
                else:
                    raise ValueError('Must provide color z-position count for non-OME tiff.')
            else:
                self.size_z = size_z

            # Get image size
            sizeXStart = infoStr.find('256 image_width (1H)')
            sizeXEnd = infoStr.find('257', sizeXStart, sizeXStart + 200)
            self.imgSizeX = int(infoStr[sizeXStart + 21:sizeXEnd - 3])

            sizeYStart = infoStr.find('257 image_length (1H)')
            sizeYEnd = infoStr.find('258', sizeXStart, sizeXStart + 200)
            self.imgSizeY = int(infoStr[sizeYStart + 22:sizeYEnd - 3])

            if type(self.crop_x) == tuple:
                self.imgSizeX = self.crop_x[1] - self.crop_x[0]

            if type(self.crop_y) == tuple:
                self.imgSizeY = self.crop_y[1] - self.crop_y[0]

        # Determine index for get_item based on axis sizes, otherwise default to first channel
        if (size_t * size_z * size_c == max([size_t, size_z, size_c])):
            if size_t > 1:
                get_item_axis = self.returnDimOrder.find('C')
            elif size_z > 1:
                get_item_axis = self.returnDimOrder.find('Z')
            elif size_c > 1:
                get_item_axis = self.returnDimOrder.find('C')
        else:
            get_item_axis = 0

        # Store as a tuple for easy access
        self.imgSize = (self.imgSizeY, self.imgSizeX)

        assert self.size_c * self.size_t * self.size_z == self.nPages, "Tiff tag values to not match number of frames in image!"
        print('Instantiated tiff object with %d channel(s), %d time point(s), and %d axial step(s)' %
              (self.size_c, self.size_t, self.size_z))

    def read(self, time=-1, channel=-1, z=-1, debugFlag=False, squeezeResult=True):

        # Convert range to list
        if type(channel) is range:
            channel = list(channel)
        if type(z) is range:
            z = list(z)
        if type(time) is range:
            time = list(time)

        # Convert single values to lists
        if type(channel) is not list:
            channel = [channel]
        if type(time) is not list:
            time = [time]
        if type(z) is not list:
            z = [z]

        # Deal with case where inputs are -1 (slice all values)
        if channel[0] < 0:
            channel = range(self.size_c)
        if time[0] < 0:
            time = range(self.size_t)
        if z[0] < 0:
            z = range(self.size_z)

        # Generate a mask for each channel
        cMask = np.zeros(self.size_c)
        cMask[channel] = 1
        zMask = np.zeros(self.size_z)
        zMask[z] = 1
        tMask = np.zeros(self.size_t)
        tMask[time] = 1

        # Get ordering indicies of each channel
        dimRangeDict = {}
        dimRangeDict['d' + str(self.tiffDimOrder.find('C'))] = cMask.astype(int)
        dimRangeDict['d' + str(self.tiffDimOrder.find('T'))] = zMask.astype(int)
        dimRangeDict['d' + str(self.tiffDimOrder.find('Z'))] = tMask.astype(int)

        # Create a mask for pages to load
        frameIdx = np.kron(np.kron(dimRangeDict['d2'], dimRangeDict['d1']), dimRangeDict['d0'])
        pagesToLoad = np.arange(self.nPages)
        pagesToLoad = list(pagesToLoad[frameIdx == 1])

        if debugFlag:
            print('pages to load:')

        # Read pagesToLoad page list from tiff file
        with open(self.file_path, 'rb') as f:
            tif = tifffile.TiffFile(f, pages=pagesToLoad, is_ome=False)
            tiffPages = tif.asarray()

        # Deal with cropping in x and y
        if type(self.crop_x) == tuple:
            tiffPages = tiffPages[:, :, self.crop_x[0]:self.crop_x[1]]
        if type(self.crop_y) == tuple:
            tiffPages = tiffPages[:, self.crop_y[0]:self.crop_y[1], :]

        if tiffPages.ndim == 2:
            tiffPages = tiffPages[np.newaxis, :, :]

        # Place values in c1orrect positions
        cDim = self.tiffDimOrder.find('C')
        zDim = self.tiffDimOrder.find('Z')
        tDim = self.tiffDimOrder.find('T')

        resSize = np.zeros(3, dtype=np.int)
        resChanMix = np.zeros(3, dtype=np.int)

        resSize[self.returnDimOrder.find('C')] = len(channel)
        resSize[self.returnDimOrder.find('Z')] = len(z)
        resSize[self.returnDimOrder.find('T')] = len(time)

        result = np.empty((resSize[0], resSize[1], resSize[2],
                           self.imgSizeY, self.imgSizeX),
                          dtype=np.uint16)
        # Place
        stride_dict = {}
        for cIdx in range(len(channel)):
            for zIdx in range(len(z)):
                for tIdx in range(len(time)):

                    # Determine mapping of component indicies to tiff page index
                    stride_dict['d' + str(self.tiffDimOrder.find('C'))] = \
                        {'idx': cIdx, "ct": len(channel)}
                    stride_dict['d' + str(self.tiffDimOrder.find('Z'))] = \
                        {'idx': zIdx, "ct": len(z)}
                    stride_dict['d' + str(self.tiffDimOrder.find('T'))] = \
                        {'idx': tIdx, "ct": len(time)}

                    posIdx = stride_dict['d0']['idx'] + \
                        stride_dict['d0']['ct'] * stride_dict['d1']['idx'] + \
                        stride_dict['d0']['ct'] * stride_dict['d1']['ct'] * stride_dict['d2']['idx']

                    if debugFlag:
                        print("cidx: %d, zIdx %d, tIdx %d :: posIdx %d"
                              % (cIdx, zIdx, tIdx, posIdx))

                    # Grab Correct Frame
                    resChanMix[self.returnDimOrder.find('C')] = cIdx
                    resChanMix[self.returnDimOrder.find('Z')] = zIdx
                    resChanMix[self.returnDimOrder.find('T')] = tIdx

                    result[resChanMix[0], resChanMix[1], resChanMix[2], :, :] = \
                        tiffPages[posIdx, :, :]

        # Remove extra dimensions from result (can be disabled by setting to false)
        if (squeezeResult):
            result = np.squeeze(result)

        return(result)

def applyPrecisionToFrameStateList(frame_state_list, precision=5):
    import copy
    frame_state_list_filtered = copy.deepcopy(frame_state_list)
    for frame in frame_state_list_filtered:
        for device_name in frame:
            if isinstance(frame[device_name], dict):
                if device_name != 'time_sequence_s':
                    # Process common metadata
                    for key in frame[device_name]['common']:
                        if isinstance(frame[device_name]['common'][key], float):
                            frame[device_name]['common'][key] = round(frame[device_name]['common'][key], precision)

                    # Process individual states
                    for state in frame[device_name]['states']:
                        for state_t in state:
                            for key in state_t:
                                if isinstance(state_t[key], float):
                                    state_t[key] = round(state_t[key], precision)
                                elif isinstance(state_t[key], dict):
                                    for key2 in state_t[key]:
                                        if isinstance(state_t[key][key2], float):
                                            state_t[key][key2] = round(state_t[key][key2], precision)
    return frame_state_list_filtered


def removeBackground(frame, background=None, max_thresh=None, roi=None):
    # Perform background subtraction
    if background is not None:
        # Normalize by background
        frame /= background
    else:
        # If max_thresh is not defined, set to infinity
        if max_thresh is None:
            max_thresh = np.inf

        # Divide by the mean
        if yp.mean(frame) < max_thresh:
            frame /= yp.mean(frame)
        else:
            frame /= max_thresh

    # Intensity is always positive, so enforce it
    frame[frame < 0.] = 0.

    return frame


class ImageIterationPlot():
    def __init__(self, ax, frame_list, max_value=65535):
        index=np.arange(1)
        self.frame_list = frame_list
        self.mean_plot, = ax.plot(index, np.mean(np.mean(frame_list[index, :, :],axis=1),axis=1), c='y', label='Mean')
        self.max_plot, = ax.plot(index, np.max(np.max(frame_list[index, :, :],axis=1),axis=1), c='b', label='Max')
        # self.mean_plot.set_xlim([0, frame_list.shape[0]])
        ax.set_xlim([1, yp.shape(frame_list)[0]])
        ax.set_ylim([0, max_value])
        ax.set_title('Frame Mean and Max')

        ax.legend()

        ax.set_xlabel('Frame Number')

    def update(self, frame_index):
        index = np.arange(frame_index + 1)
        self.mean_plot.set_xdata(index+1)
        self.mean_plot.set_ydata(np.mean(np.mean(self.frame_list[index, :, :], axis=1),axis=1))
        self.max_plot.set_xdata(index+1)
        self.max_plot.set_ydata(np.max(np.max(self.frame_list[index,:,:],axis=1),axis=1))



class NaPlot():
    def __init__(self, ax, frame_state_list, source_list_na, objective_na=None,
                 show_background=True, background_color='k', foreground_color='y',
                 use_slider=True, marker_size=15, normalize_color=True, flip_y_axis=True,
                 color_names = ('r', 'g', 'b')):

        from matplotlib import pyplot as plt
        self.frame_state_list = frame_state_list
        self.source_list_na = source_list_na

        frame_led_list = []
        frame_led_color_list = []

        # Loop over all frames to generate led_color_list
        for frame_state in frame_state_list:
            for led_pattern in frame_state['illumination']['states']:
                for led in led_pattern:
                    frame_led_list.append(led["index"])
                    color = [0, 0, 0]
                    for color_index, color_name in enumerate(color_names):
                        if color_name in led['value']:
                            color[color_index] = led["value"][color_name]

                    if max(color) > 1 and normalize_color:
                        color[0] /= max(color)
                        color[1] /= max(color)
                        color[2] /= max(color)

                frame_led_color_list.append(color)

        if show_background:
            # Background plot, containing all LEDs
            self.background_plot = ax.scatter(source_list_na[:, 0], source_list_na[:, 1], s=marker_size, alpha=0.5, c=background_color)
        else:
            self.background_plot = None

        # Foreground plot, containing LEDs from current frame
        if use_slider:
            self.foreground_plot = ax.scatter(source_list_na[frame_led_list, 0],
                                              source_list_na[frame_led_list, 1], c=frame_led_color_list, s=marker_size)

        ax.set_title("LED Illumination")
        ax.set_xlabel("$NA_x$")
        ax.set_ylabel("$NA_y$")

        ax_lim = 1.1 * np.max(np.abs(source_list_na))
        if ax_lim == 0.0:
            ax_lim = 0.1

        ax.set_xlim(xmin=-ax_lim, xmax=ax_lim)
        ax.set_ylim(ymin=-ax_lim, ymax=ax_lim)
        # ax.set_aspect(1.)

        if objective_na is not None:
            circle1 = plt.Circle((0, 0), objective_na, edgecolor='r', fill=False, lw=1)
            ax.add_artist(circle1)

    def update(self, frame_index, subframe_time_index=0):
        frame_index = int(frame_index)
        illumination_index = int(subframe_time_index)

        pattern_led_list = []
        pattern_led_color_list = []

        # Get illumination pattern for this frame and illumination index
        led_pattern = self.frame_state_list[frame_index]['illumination'][illumination_index]

        for led in led_pattern:
            pattern_led_list.append(led["index"])
            color_dict = {'r' : 0,'g' : 0, 'b' : 0}

            # Grab RGB color values
            for led_color in led['value']:
                if led_color in color_dict:
                    color_dict[led_color] = led['value'][led_color]

            # Deal with the case that illumnination is not normalized (0 to 1)
            if max(list(color_dict.values())) > 1:
                color_dict['r'] /= max(list(color_dict.values()))
                color_dict['g'] /= max(list(color_dict.values()))
                color_dict['b'] /= max(list(color_dict.values()))

            pattern_led_color_list.append(list(color_dict.values()))

        # Update highlighted NA position
        self.foreground_plot.set_offsets(self.source_list_na[pattern_led_list, :])
        self.foreground_plot.set_color(pattern_led_color_list)


class PositionPlot():
    def __init__(self, ax, frame_state_list, show_legend=False, pixel_size_um=1, frame_number=0, subframe_time_index=0,
                 forground_color='y', forground_highlight_color='w', background_color='m', units='mm'):
        self.ax = ax
        self.frame_state_list = frame_state_list
        # Draw initial plot
        max_x = -1e10
        max_y = -1e10
        min_x = 1e10
        min_y = 1e10

        # First index in segment veriable represents each "segment" of the kernel, second is the sub-indicies (or pixels)
        self.position_list = []
        for state_index, state in enumerate(self.frame_state_list):
            position_list_frame = []
            for time_point_position_list in state['position']['states']:
                for position in time_point_position_list:
                    # Convert all units to um
                    if units == 'pixels':
                        position_list_frame.append([position['value']['y'] * pixel_size_um, position['value']['x'] * pixel_size_um])
                    elif units == 'mm':
                        position_list_frame.append([position['value']['y'] * 1000., position['value']['x'] * 1000.])
                    elif units == 'um':
                        position_list_frame.append([position['value']['y'], position['value']['x']])
                    else:
                        raise ValueError('Units %s are unsupported!' % units)
            self.position_list.append(position_list_frame)

        for positon_frame in self.position_list:
            positon_frame_np = np.asarray(positon_frame)
            ax.plot(positon_frame_np[:, 1], positon_frame_np[:, 0], color=background_color,
                    linestyle='-', linewidth=2)

            max_x = max(max_x, np.max(positon_frame_np[:, 1]))
            min_x = min(min_x, np.min(positon_frame_np[:, 1]))
            max_y = max(max_y, np.max(positon_frame_np[:, 0]))
            min_y = min(min_y, np.min(positon_frame_np[:, 0]))

        mean_x = (max_x + min_x) / 2
        mean_y = (max_y + min_y) / 2

        # Create foreground plot
        positon_frame_np = np.asarray(self.position_list[frame_number])
        self.foreground_plot, = ax.plot(positon_frame_np[:, 1], positon_frame_np[:, 0], color=forground_color, linestyle='-', linewidth=2)

        # Create foreground highlight plot
        # self.foreground_highlight_plot, = ax.plot(positon_frame_np[subframe_time_index:subframe_time_index+1, 1], positon_frame_np[subframe_time_index:subframe_time_index+1, 0], color='r', linestyle='-', linewidth=2)

        self.foreground_highlight_plot = ax.scatter(positon_frame_np[subframe_time_index, 1], positon_frame_np[subframe_time_index, 0], color='w')

        plot_size = 1.1 * max(max_x - min_x, max_y - min_y) / 2
        ax.set_xlim(mean_x + np.array([-plot_size, plot_size]))
        ax.set_ylim(mean_y + np.array([-plot_size, plot_size]))

        ax.set_xlabel('Position X (um)')
        ax.set_ylabel('Position Y (um)')
        ax.invert_yaxis()

    def update(self, frame_number, subframe_time_index=0):
        frame_number = int(frame_number)
        subframe_time_index = int(subframe_time_index)
        positon_frame_np = np.asarray(self.position_list[frame_number])
        self.foreground_plot.set_xdata(positon_frame_np[:, 1])
        self.foreground_plot.set_ydata(positon_frame_np[:, 0])

        self.foreground_highlight_plot.set_offsets([positon_frame_np[subframe_time_index, 1], positon_frame_np[subframe_time_index, 0]])
        print(positon_frame_np[subframe_time_index, :])

        self.ax.set_title('XY Position')


class ImagePlot():
    def __init__(self, ax, frame_list, cmap='gray', contrast='global',
                 pixel_size_um=None, colorbar=True, **kwargs):

        from matplotlib import pyplot as plt
        from matplotlib_scalebar.scalebar import ScaleBar

        # Store frame list
        self.frame_list = frame_list

        # Store axis
        self.ax = ax

        # Calculate image bounds
        vmin_global = min([np.min(frame) for frame in frame_list])
        vmax_global = max([np.max(frame) for frame in frame_list])

        # Determine vmax and vmin
        if contrast == 'global':
            vmin, vmax = vmin_global, vmax_global
        else:
            vmin, vmax = np.min(self.frame_list[0]), np.max(self.frame_list[0])

        # Plot first image
        self.image_plot = ax.imshow(self.frame_list[0], cmap=cmap,
                                    vmin=vmin, vmax=vmax, **kwargs)

        if pixel_size_um is not None:
            scalebar = ScaleBar(pixel_size_um, units="um") # 1 pixel = 0.2 1/cm
            ax.add_artist(scalebar)

        self.ax.set_aspect(1.)
        self.ax.set_title("Frame %d / %d" % (0, len(frame_list)))

        # Turn off axis
        plt.axis('off')

        # Draw colorbar, if desired
        if colorbar:
            self.image_colorbar = plt.colorbar(self.image_plot, ax=ax)

    def update(self, val):
        frame_index = int(val)
        self.image_plot.set_data(self.frame_list[frame_index])
        self.ax.set_title("Frame %d / %d" % (val, len(self.frame_list)))
