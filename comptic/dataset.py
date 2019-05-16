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
from scipy import io as sio
from llops.display import objToString, Color
from libwallerlab.utilities.display import ImageIterationPlot, NaPlot
from libwallerlab.utilities.camera import demosaic
import llops as yp

from libwallerlab.utilities.metadata import Metadata, Reconstructions

VERSION = 0.2

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
        self.divide_background = divide_background

        # Dark current, which captures the signal when there is no illumination
        self.dark_current = None
        self.subtract_mean_dark_current = subtract_mean_dark_current

        # Define version
        self.version = VERSION

        # Initialize frame subsampling lists
        self._position_segment_indicies = []

        # See if any reconstrucitons exist
        self.reconstructions = Reconstructions(self)

        # Ground truth
        self.ground_truth = None

        # Decide whether we need to perform background removal
        self.divide_background = divide_background

        # Median filter parameters
        self.median_filter_threshold = median_filter_threshold
        self.use_median_filter = use_median_filter

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
        if self._frame_mask is not None:
            return self._frame_mask
        else:
            return self.frame_mask_full

    @frame_mask.setter
    def frame_mask(self, new_frame_mask):
        self._frame_mask = new_frame_mask

    @property
    def frame_mask_full(self):
        return list(range(len(self._frame_state_list)))

    @property
    def shape(self):
        return tuple([len(self.frame_mask)] + list(self.frame_shape))

    @property
    def type(self):
        return self.metadata.type

    @type.setter
    def type(self, new_type):
        self.metadata.type = new_type

    @property
    def frame_list(self):
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
        if self._frame_list:
            if self.frame_mask is not None:
                for index, frame_index in enumerate(self.frame_mask):
                    self._frame_list[frame_index] = yp.dcopy(new_frame_list[index])
            else:
                for i in range(len(new_frame_list)):
                    self._frame_list[i] = yp.dcopy(new_frame_list[i])
        else:
            self._frame_list = []
            if new_frame_list is not None:
                for i in range(len(new_frame_list)):
                    self._frame_list.append(yp.dcopy(new_frame_list[i]))
            else:
                self._frame_list = None

    @property
    def frame_state_list(self):
        if self.frame_mask is not None:
            return [self._frame_state_list[i] for i in self.frame_mask]
        else:
            return self._frame_state_list

    @frame_state_list.setter
    def frame_state_list(self, new_frame_state_list):
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
        q = Color.YELLOW
        self.frame_mask = list(range(len(self._frame_list)))


    # def __str__(self):
    #     """
    #     A robust, nice printer for dataset parameters
    #     """
    #     str_to_print = "\n                +------------------+\n\
    #            /                  /|\n\
    #           / %s/ |\n\
    #          /                  /  |\n\
    #         +------------------+   |\n\
    #         |      %s|   |\n\
    #         |                  |   |\n\
    #         |                  |   +\n\
    #         | %s     |  /\n\
    #         |                  | /\n\
    #         |    Data Shape    |/\n\
    #         +------------------+\n " % ('{message: <{fill}}'.format(message=Color.YELLOW + "# Images = " + str(self.frame_list.shape[0]) + Color.END, fill='15'),
    #                                     '{message: <{fill}}'.format(
    #                                         message=Color.YELLOW + "N = " + str(self.frame_list.shape[2]) + Color.END, fill='15'),
    #                                     '{message: <{fill}}'.format(message=Color.YELLOW + "M = " + str(self.frame_list.shape[1]) + Color.END, fill='15'))
    #
    #     str_to_print = str_to_print + '\n' + Color.BOLD + \
    #         Color.RED + "Metadata:\n" + Color.END + str(self.metadata)
    #     return(str_to_print)

    def save(self, dataset_path=None, header=None, tif_compression=0, metadata_only=False, precision=6, bit_depth=16):
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

    def showIllum(self, colormap='gray', figsize=(5, 5), figaspect="wide", show_fourier=False):
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

    def show(self, hardware_element=None, colormap='gray', figsize=(10,4), figaspect="wide",
             show_fourier=False, hide_sliders=False, show_iteration_plot=False,
             images_only=False, hw_type=None):

        from matplotlib.widgets import Slider
        from matplotlib import pyplot as plt
        from .display import ImagePlot, NaPlot, PositionPlot

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

        for device in device_list:
            if device.__class__.__name__ is "Camera":
                self.plot_list.append(ImagePlot(axis_list[axis_index],
                                                             self.frame_list,
                                                             colormap=colormap,
                                                             pixel_size_um=self.metadata.system.eff_pixel_size_um,
                                                             roi=self.metadata.camera.roi))
                axis_index += 1

            elif device.__class__.__name__  is "Illumination" and not images_only:

                self.plot_list.append(NaPlot(axis_list[axis_index], self.frame_state_list,
                                                          np.asarray(self.metadata.illumination.state_list.design),
                                                          objective_na=self.metadata.objective.na))
                axis_index += 1

            elif device.__class__.__name__  is "Position" and not images_only:
                self.plot_list.append(PositionPlot(axis_list[axis_index], self.frame_state_list, pixel_size_um=self.metadata.system.eff_pixel_size_um))
                axis_index += 1

            elif device.__class__.__name__  is "Focus" and not images_only:
                raise NotImplementedError

            elif device.__class__.__name__  is "Pupil" and not images_only:

                raise NotImplementedError

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
        slider = Slider(ax_slider, 'Frame Index', 0, len(self.frame_state_list), valinit=frame_index, valfmt='%d')
        slider.on_changed(update)

        if not images_only:
            ax_slider_illum = plt.axes([0.25, 0.08, 0.65, 0.03])
            slider_illum = Slider(ax_slider_illum, 'Interframe Index', 0, len(self.frame_state_list[frame_index]['illumination']['states']), valinit=0, valfmt='%d')
            slider_illum.on_changed(updateInterFrame)
        else:
            slider_illum = None

        if hide_sliders:
            slider.ax.set_axis_off()
            slider_illum.ax.set_axis_off()

        def frame_update_function_handle(frame_index): return update(frame_index)

        def interframe_update_function_handle(interframe_index): return updateInterFrame(interframe_index)

        plt.show()

        return(fig, slider, slider_illum, frame_update_function_handle, interframe_update_function_handle)

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


def saveImageStack(filename, img, clim=None, bit_depth=8, colormap=None):
    from matplotlib.pyplot import get_cmap
    assert bit_depth == 8 or bit_depth == 16, "saveImageStack only handles 8-bit and 16-bit images!"
    if colormap is not None:
        bit_depth = 8
        cmap = get_cmap(colormap)
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
    if colormap is None:
        if bit_depth == 8:
            img_tiff = img_tiff.astype('uint8')
        elif bit_depth == 16:
            img_tiff = img_tiff.astype('uint16')
        if len(img.shape) == 2:
            imsave(filename, img_tiff)
        elif len(img.shape) == 3:
            imsave(filename, img_tiff[:, np.newaxis, :, :])
    else:
        assert bit_depth == 8, 'colormap only support 8-bit images!'
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


class MotionDeblurFunctions():

    def __init__(self, dataset):
        """This class implements motion-deblur specific changes and fixes."""
        self.dataset = dataset

        # Whether to skip first frame in a segment
        self.skip_first_frame_segment = False

        # Apply fixes for old datasets
        if hasattr(self.dataset, 'version'):
            if self.dataset.version < 1.0:

                self.fixOldMdDatasets()

                if self.dataset.metadata.type == 'stop and stare':
                    self.dataset.frame_state_list = list(reversed(self.dataset.frame_state_list))

    def fixOldMdDatasets(self):
        # Expand frame_state_list
        self.expandFrameStateList(self.dataset.frame_state_list, position_units='mm')

        # Fix frame state list
        self.fixFrameStateList(self.dataset.frame_state_list)

        # Flip position coordinates
        self.flipPositionCoordinates(x=True)

    @property
    def frame_segment_list(self):
        """Calculates and returns position segment indicies of each frame."""
        position_segment_indicies = []
        for frame_state in self.dataset.frame_state_list:
            position_segment_indicies.append(frame_state['position']['common']['linear_segment_index'])

        return position_segment_indicies

    @property
    def frame_segment_direction_list(self):
        """Calculates and returns position segment indicies of each frame."""

        # Determine the direction of individual segments
        segment_direction_list = []
        segment_list = sorted(yp.unique(self.frame_segment_list))

        # Store previous position segment indicies
        frame_mask_old = self.dataset.frame_mask

        # Loop over segments
        for segment_index in segment_list:

            # Set positon segment index
            self.dataset.motiondeblur.position_segment_indicies = [segment_index]

            # Get start position of first frame in segment
            x_start = self.dataset.frame_state_list[0]['position']['states'][0][0]['value']['x']
            y_start = self.dataset.frame_state_list[0]['position']['states'][0][0]['value']['y']

            # Get start position of last frame in segment
            x_end = self.dataset.frame_state_list[-1]['position']['states'][-1][0]['value']['x']
            y_end = self.dataset.frame_state_list[-1]['position']['states'][-1][0]['value']['y']

            vector = np.asarray(((y_end - y_start), (x_end - x_start)))
            vector /= np.linalg.norm(vector)

            # Append segment direction vector to list
            segment_direction_list.append(vector.tolist())

        # Reset position segment indicies
        self.dataset.frame_mask = frame_mask_old

        # Expand to frame basis
        frame_segment_direction_list = []
        for frame_index in range(self.dataset.shape[0]):
            # Get segment index
            segment_index = self.frame_segment_list[frame_index] - min(self.frame_segment_list)

            # Get segment direction
            segment_direction = segment_direction_list[segment_index]

            # Append to list
            frame_segment_direction_list.append(segment_direction)

        return frame_segment_direction_list

    def expandFrameStateList(self, frame_state_list, position_units='mm'):
        """ This function expands redundant information in the frame_state_list of a dataset (specific to motion deblur datasets for now)"""

        # Store first frame as a datum
        frame_state_0 = frame_state_list[0]

        # Loop over frame states
        for frame_state in frame_state_list:

            # Fill in illumination and position if this frame state is compressed
            if type(frame_state['illumination']) is str:
                frame_state['illumination'] = copy.deepcopy(frame_state_0['illumination'])

                # Deterime direction of scan
                dx0 = frame_state['position']['states'][-1][0]['value']['x'] - frame_state['position']['states'][0][0]['value']['x']
                dy0 = frame_state['position']['states'][-1][0]['value']['y'] - frame_state['position']['states'][0][0]['value']['y']
                direction = np.asarray((dy0, dx0))
                direction /= np.linalg.norm(direction)

                # Get frame spacing
                spacing = frame_state['position']['common']['velocity'] * frame_state['position']['common']['led_update_rate_us'] / 1e6
                dy = direction[0] * spacing
                dx = direction[1] * spacing

                # Assign new positions in state_list
                states_new = []
                for state_index in range(len(frame_state_0['position']['states'])):
                    state = copy.deepcopy(frame_state['position']['states'][0])
                    state[0]['time_index'] = state_index
                    state[0]['value']['x'] += dx * state_index
                    state[0]['value']['y'] += dy * state_index
                    state[0]['value']['units'] = position_units
                    states_new.append(state)

                frame_state['position']['states'] = states_new

    def fixFrameStateList(self, frame_state_list, major_axis='y'):
        """Catch-all function for various hacks and dataset incompatabilities."""
        axis_coordinate_list = []
        # Loop over frame states
        for frame_state in frame_state_list:
            # Check if this state is a shallow copy of the first frame
            if id(frame_state['illumination']) is not id(self.dataset._frame_state_list[0]['illumination']):

                # Check if position is in the correct format (some stop and stare datasets will break this)
                if type(frame_state['position']) is list:

                    # Fix up positions
                    frame_state['position'] = {'states': frame_state['position']}
                    frame_state['position']['common'] = {'linear_segment_index': 0}

                    # Fix up illumination
                    frame_state['illumination'] = {'states': frame_state['illumination']}

                # Add linear segment indicies if these do not already exist
                frame_axis_coordinate = frame_state['position']['states'][0][0]['value'][major_axis]
                if frame_axis_coordinate not in axis_coordinate_list:
                    axis_coordinate_list.append(frame_axis_coordinate)
                    position_segment_index = len(axis_coordinate_list) - 1
                else:
                    position_segment_index = axis_coordinate_list.index(frame_axis_coordinate)

                # position_segment_indicies.append(position_segment_index)
                frame_state['position']['common']['linear_segment_index'] = position_segment_index
            else:
                print('Ignoring state.')

    def flipPositionCoordinates(self, x=False, y=False):
        for frame_state in self.dataset._frame_state_list:
            # Check if this state is a shallow copy of the first frame
            if id(frame_state) is not id(self.dataset._frame_state_list[0]):
                for state in frame_state['position']['states']:
                    for substate in state:
                        if x:
                            substate['value']['x'] *= -1
                        if y:
                            substate['value']['y'] *= -1
            else:
                print('Ignoring state.')

    def flipIlluminationSequence(self):
        for frame_state in self.dataset._frame_state_list:
            frame_state['illumination']['states'] = list(reversed(frame_state['illumination']['states']))

    def blur_vectors(self, dtype=None, backend=None, debug=False,
                     use_phase_ramp=False, corrections={}):
        """
        This function generates the object size, image size, and blur kernels from
        a libwallerlab dataset object.

            Args:
                dataset: An io.Dataset object
                dtype [np.float32]: Which datatype to use for kernel generation (All numpy datatypes supported)
            Returns:
                object_size: The object size this dataset can recover
                image_size: The computed image size of the dataset
                blur_kernel_list: A dictionary of blur kernels lists, one key per color channel.

        """
        # Assign dataset
        dataset = self.dataset

        # Get corrections from metadata
        if len(corrections) is 0 and 'blur_vector' in self.dataset.metadata.calibration:
            corrections = dataset.metadata.calibration['blur_vector']

        # Get datatype and backends
        dtype = dtype if dtype is not None else yp.config.default_dtype
        backend = backend if backend is not None else yp.config.default_backend

        # Calculate effective pixel size if necessaey
        if dataset.metadata.system.eff_pixel_size_um is None:
            dataset.metadata.system.eff_pixel_size_um = dataset.metadata.camera.pixel_size_um / \
                (dataset.metadata.objective.mag * dataset.metadata.system.mag)

        # Recover and store position and illumination list
        blur_vector_roi_list = []
        position_list, illumination_list = [], []
        frame_segment_map = []

        for frame_index in range(len(dataset.frame_mask)):
            frame_state = copy.deepcopy(dataset.frame_state_list[frame_index])

            # Store which segment this measurement uses
            frame_segment_map.append(frame_state['position']['common']['linear_segment_index'])

            # Extract list of illumination values for each time point
            if 'illumination' in frame_state:
                illumination_list_frame = []
                for time_point in frame_state['illumination']['states']:
                    illumination_list_time_point = []
                    for illumination in time_point:
                        illumination_list_time_point.append(
                            {'index': illumination['index'], 'value': illumination['value']})
                    illumination_list_frame.append(illumination_list_time_point)

            else:
                raise ValueError('Frame %d does not contain illumination information' % frame_index)

            # Extract list of positions for each time point
            if 'position' in frame_state:
                position_list_frame = []
                for time_point in frame_state['position']['states']:
                    position_list_time_point = []
                    for position in time_point:
                        if 'units' in position['value']:
                            if position['value']['units'] == 'mm':
                                ps_um = dataset.metadata.system.eff_pixel_size_um
                                position_list_time_point.append(
                                    [1000 * position['value']['y'] / ps_um, 1000 * position['value']['x'] / ps_um])
                            elif position['value']['units'] == 'um':
                                position_list_time_point.append(
                                    [position['value']['y'] / ps_um, position['value']['x'] / ps_um])
                            elif position['value']['units'] == 'pixels':
                                position_list_time_point.append([position['value']['y'], position['value']['x']])
                            else:
                                raise ValueError('Invalid units %s for position in frame %d' %
                                                 (position['value']['units'], frame_index))
                        else:
                            # print('WARNING: Could not find posiiton units in metadata, assuming mm')
                            ps_um = dataset.metadata.system.eff_pixel_size_um
                            position_list_time_point.append(
                                [1000 * position['value']['y'] / ps_um, 1000 * position['value']['x'] / ps_um])

                    position_list_frame.append(position_list_time_point[0])  # Assuming single time point for now.

                # Define positions and position indicies used
                positions_used, position_indicies_used = [], []
                for index, pos in enumerate(position_list_frame):
                    for color in illumination_list_frame[index][0]['value']:
                        if any([illumination_list_frame[index][0]['value'][color] > 0 for color in illumination_list_frame[index][0]['value']]):
                            position_indicies_used.append(index)
                            positions_used.append(pos)

                # Generate ROI for this blur vector
                from libwallerlab.projects.motiondeblur.blurkernel import getPositionListBoundingBox
                blur_vector_roi = getPositionListBoundingBox(positions_used)

                # Append to list
                blur_vector_roi_list.append(blur_vector_roi)

                # Crop illumination list to values within the support used
                illumination_list.append([illumination_list_frame[index] for index in range(min(position_indicies_used), max(position_indicies_used) + 1)])

                # Store corresponding positions
                position_list.append(positions_used)

        # Apply kernel scaling or compression if necessary
        if 'scale' in corrections:

            # We need to use phase-ramp based kernel generation if we modify the positions
            use_phase_ramp = True

            # Modify position list
            for index in range(len(position_list)):
                _positions = np.asarray(position_list[index])
                for scale_correction in corrections['scale']:
                    factor, axis = corrections['scale']['factor'], corrections['scale']['axis']
                    _positions[:, axis] = ((_positions[:, axis] - yp.min(_positions[:, axis])) * factor + yp.min(_positions[:, axis]))
                position_list[index] = _positions.tolist()

        # Synthesize blur vectors
        blur_vector_list = []
        for frame_index in range(dataset.shape[0]):
            #  Generate blur vectors
            if use_phase_ramp:
                import ndoperators as ops
                kernel_shape = [yp.fft.next_fast_len(max(sh, 1)) for sh in blur_vector_roi_list[frame_index].shape]
                offset = yp.cast([sh // 2 + st for (sh, st) in zip(kernel_shape, blur_vector_roi_list[frame_index].start)], 'complex32', dataset.backend)

                # Create phase ramp and calculate offset
                R = ops.PhaseRamp(kernel_shape, dtype='complex32', backend=dataset.backend)

                # Generate blur vector
                blur_vector = yp.zeros(R.M, dtype='complex32', backend=dataset.backend)
                for pos, illum in zip(position_list[frame_index], illumination_list[frame_index]):
                    pos = yp.cast(pos, dtype=dataset.dtype, backend=dataset.backend)
                    blur_vector += (R * (yp.cast(pos - offset, 'complex32')))

                # Take inverse Fourier Transform
                blur_vector = (yp.abs(yp.cast(yp.iFt(blur_vector)), 0.0))

            else:
                blur_vector = yp.asarray([illum[0]['value']['w'] for illum in illumination_list[frame_index]],
                                         dtype=dtype, backend=backend)

            # Normalize illuminaiton vectors
            blur_vector /= yp.scalar(yp.sum(blur_vector))

            # Append to list
            blur_vector_list.append(blur_vector)

        # Return
        return blur_vector_list, blur_vector_roi_list

    @property
    def roi_list(self):

        # Get blur vectors and ROIs
        blur_vector_list, blur_vector_roi_list = self.blur_vectors()

        # Generate measurement ROIs
        roi_list = []
        for index, (blur_vector, blur_roi) in enumerate(zip(blur_vector_list, blur_vector_roi_list)):
            # Determine ROI start from blur vector ROI
            convolution_support_start = [kernel_center - sh // 2 for (kernel_center, sh) in zip(blur_roi.center, self.dataset.frame_shape)]

            # Generate ROI
            roi_list.append(yp.Roi(start=convolution_support_start, shape=self.dataset.frame_shape))

        return roi_list

    @property
    def position_segment_indicies(self):
        """Returns all segment indicies which are currently used (one per segment, NOT one per frame)."""
        if self._position_segment_indicies:
            # Return saved position_segment_indicies
            return self._position_segment_indicies
        else:
            # Determine which position segment indicies are in this dataset
            position_segment_indicies = []
            for frame_state in self.dataset.frame_state_list:
                if frame_state['position']['common']['linear_segment_index'] not in position_segment_indicies:
                    position_segment_indicies.append(frame_state['position']['common']['linear_segment_index'])

            # Return these indicies
            return position_segment_indicies

    @position_segment_indicies.setter
    def position_segment_indicies(self, new_position_segment_indicies):

        # Ensure the input is of resonable type
        assert type(new_position_segment_indicies) in (tuple, list)

        # Check that all elements are within the number of frames
        assert all([index in self.position_segment_indicies_full for index in new_position_segment_indicies])

        # Set new position_segment_index
        self._position_segment_indicies = new_position_segment_indicies

        # Update frame_mask to reflect this list
        frame_subset = []
        first_frame_is_skipped = not self.skip_first_frame_segment
        for position_segment_index in new_position_segment_indicies:
            for index, frame_state in enumerate(self.dataset._frame_state_list):
                if frame_state['position']['common']['linear_segment_index'] == position_segment_index:
                    if not first_frame_is_skipped:
                        first_frame_is_skipped = True
                    else:
                        frame_subset.append(index)
        self.dataset.frame_mask = frame_subset

    @property
    def frame_position_segment_indicies(self):
        """Returns a list of segment indicies for each frame."""
        _frame_position_segment_indicies = []
        for frame_state in self.dataset._frame_state_list:
            _frame_position_segment_indicies.append(frame_state['position']['common']['linear_segment_index'])
        return _frame_position_segment_indicies

    @property
    def position_segment_indicies_full(self):
        """Returns all segment indicies which are in the dataset (one per segment, NOT one per frame)."""
        _position_segment_indicies_full = []
        for frame_state in self.dataset._frame_state_list:
            if frame_state['position']['common']['linear_segment_index'] not in _position_segment_indicies_full:
                _position_segment_indicies_full.append(frame_state['position']['common']['linear_segment_index'])
        return _position_segment_indicies_full

    def normalize(self, force=False, debug=False):
        if 'normalization' not in self.dataset.metadata.calibration or force:
            # Calculation normalization vectors
            from libwallerlab.projects.motiondeblur.recon import normalize_measurements
            (frame_normalization_list_y, frame_normalization_list_x) = normalize_measurements(self.dataset, debug=debug)

            # Convert to numpy for saving
            _frame_normalization_list_y = [yp.changeBackend(norm, 'numpy').tolist() for norm in frame_normalization_list_y]
            _frame_normalization_list_x = [yp.changeBackend(norm, 'numpy').tolist() for norm in frame_normalization_list_x]

            # Save in metadata
            self.dataset.metadata.calibration['normalization'] = {'frame_normalization_x': _frame_normalization_list_x,
                                                                  'frame_normalization_y': _frame_normalization_list_y}
            # Save calibration file
            self.dataset.saveCalibration()

    def register(self, force=False, segment_offset=(0, 0), frame_offset=0,
                 blur_axis=1, frame_registration_mode='xc', debug=False,
                 segment_registration_mode='xc', write_file=True):

        if 'registration' not in self.dataset.metadata.calibration or force:

            # Assign all segments
            self.dataset.motiondeblur.position_segment_indicies = self.dataset.motiondeblur.position_segment_indicies_full

            # Pre-compute indicies for speed
            frame_segment_list = self.dataset.motiondeblur.frame_segment_list
            frame_segment_direction_list = self.dataset.motiondeblur.frame_segment_direction_list

            # Apply pre-computed offset
            frame_offset_list = []
            for frame_index in range(len(self.dataset.frame_mask)):

                # Get segment index and direction
                segment_direction_is_left_right = frame_segment_direction_list[frame_index][1] > 0

                # Get index of frame in this segment
                frame_segment_index = len([segment for segment in frame_segment_list[:frame_index] if segment == frame_segment_list[frame_index]])

                # Get index of current segment
                segment_index = frame_segment_list[frame_index]

                # Apply frame dependent offset
                _offset_frame = [0, 0]
                _offset_frame[blur_axis] = frame_segment_index * frame_offset
                if not segment_direction_is_left_right:
                    _offset_frame[blur_axis] *= -1

                # Apply segment dependent offset
                _offset_segment = list(segment_offset)
                if segment_direction_is_left_right:
                    for ax in range(len(_offset_segment)):
                        if ax is blur_axis:
                            _offset_segment[ax] *= -1
                        else:
                            _offset_segment[ax] *= segment_index

                # Combine offsets
                offset = [_offset_frame[i] + _offset_segment[i] for i in range(2)]

                # Append to list
                frame_offset_list.append(offset)

            # Apply registration
            if frame_registration_mode is not None:

                # Register frames within segments
                for segment_index in self.dataset.motiondeblur.position_segment_indicies_full:
                    self.dataset.motiondeblur.position_segment_indicies = [segment_index]

                    # Get frame ROI list
                    roi_list = self.dataset.motiondeblur.roi_list

                    # Get offsets for this segment
                    frame_offset_list_segment = [frame_offset_list[index] for index in self.dataset.frame_mask]

                    # Apply frame offsets from previous steps
                    for roi, offset in zip(roi_list, frame_offset_list_segment):
                        roi += offset

                    # Perform registration
                    from libwallerlab.projects.motiondeblur.recon import register_roi_list
                    frame_offset_list_segment = register_roi_list(self.dataset.frame_list,
                                                                  roi_list,
                                                                  debug=debug,
                                                                  tolerance=(1000, 1000),
                                                                  method=frame_registration_mode,
                                                                  force_2d=False,
                                                                  axis=1)

                    # Apply correction to frame list
                    for index, frame_index in enumerate(self.dataset.frame_mask):
                        for i in range(len(frame_offset_list[frame_index])):
                            frame_offset_list[frame_index][i] += frame_offset_list_segment[index][i]

            if segment_registration_mode is not None:
                from ndoperators import VecStack, Segmentation
                from libwallerlab.projects.motiondeblur.recon import alignRoiListToOrigin, register_roi_list
                stitched_segment_list, stitched_segment_roi_list = [], []
                # Stitch segments
                for segment_index in self.dataset.motiondeblur.position_segment_indicies_full:
                    self.dataset.motiondeblur.position_segment_indicies = [segment_index]

                    # Get frame ROI list
                    roi_list = self.dataset.motiondeblur.roi_list

                    # Get offsets for this segment
                    frame_offset_list_segment = [frame_offset_list[index] for index in self.dataset.frame_mask]

                    # Apply frame offsets from previous steps
                    for roi, offset in zip(roi_list, frame_offset_list_segment):
                        roi += offset

                    # Determine segment ROI
                    stitched_segment_roi_list.append(sum(roi_list))

                    # Align ROI list to origin
                    alignRoiListToOrigin(roi_list)

                    # Create segmentation operator
                    G = Segmentation(roi_list)

                    # Get measurement list
                    y = yp.astype(VecStack(self.dataset.frame_list), G.dtype)

                    # Append to list
                    stitched_segment_list.append(G.inv * y)

                # Register stitched segments
                frame_offset_list_segment = register_roi_list(stitched_segment_list,
                                                              stitched_segment_roi_list,
                                                              debug=debug,
                                                              tolerance=(200, 200),
                                                              method=segment_registration_mode)

                # Apply registration to all frames
                self.dataset.motiondeblur.position_segment_indicies = self.dataset.motiondeblur.position_segment_indicies_full

                # Apply offset to frames
                for frame_index in range(self.dataset.shape[0]):

                    # Get segment index
                    segment_index = self.dataset.motiondeblur.frame_segment_list[frame_index]

                    # Apply offset
                    for i in range(len(frame_offset_list[frame_index])):
                        frame_offset_list[frame_index][i] += frame_offset_list_segment[segment_index][i]

            # Set updated values in metadata
            self.dataset.metadata.calibration['registration'] = {'frame_offsets': frame_offset_list,
                                                                 'segment_offset': segment_offset,  # For debugging
                                                                 'frame_offset': frame_offset}      # For debugging

            # Save calibration file
            if write_file:
                self.dataset.saveCalibration()
