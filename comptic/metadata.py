import os
import glob
from llops.display import objToString, Color
from llops import Roi

class Metadata():
    def __init__(self, from_dict=None):
        self.focus = Focus()
        self.pupil = Pupil()
        self.position = Position()
        self.illumination = Illumination()
        self.camera = Camera()
        self.objective = Objective()
        self.sample = Sample()
        self.system = System()
        self.background = Background()
        self.groundtruth = GroundTruth()
        self.reconstructions = []
        self.type = None
        self.comments = None
        self.file_header = "None"
        self.directory = "None"
        self.acquisition_time_s = None
        self.calibration = {}

        if from_dict is not None:
            from .containers import loadDictRecursive
            loadDictRecursive(self, from_dict)

    def __str__(self):
        return(objToString(self, text_color=Color.CYAN))

    def checkSampling(self, coherent=False):
        k_max_camera = 1 / (2 * self.camera.pixel_size_um / (self.objective.mag * self.system.mag))
        for item in self.illumination.spectrum.center:
            k_max_objective = (int(coherent) + 1) * self.objective.na / self.illumination.spectrum.center[item]
            if k_max_camera < k_max_objective:
                print('Channel ' + item + ': Max camera sampling is %.2f um^-1, max objective sampling is %.2f um^-1, sampling is ' % (
                    k_max_camera, k_max_objective) + Color.RED + 'Insufficient' + Color.END)
            else:
                print('Channel ' + item + ': Max camera sampling is %.2f um^-1, max objective sampling is %.2f um^-1, sampling is ' % (
                    k_max_camera, k_max_objective) + Color.YELLOW + 'Sufficient' + Color.END)

# class Calibration():
#     def __init__(self, dataset):
#         self.registration = RegistrationCalibration(dataset)
#         self.normalization = NormalizationCalibration(dataset)
#
# class RegistrationCalibration():
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.frame_offsets = None
#         self.reset()
#
#     def reset(self):
#         self.frame_offsets = [[0, 0]] * len(self.dataset._frame_state_list)
#
#
# class NormalizationCalibration():
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.frame_offsets = None
#         self.reset()
#
#     def reset(self):
#         self.frame_offsets = [[0, 0]] * len(self.dataset._frame_state_list)

class HardwareElement():
    """
    Abstract hardware element class
    """

    def __init__(self):
        self.device_name = None
        self.device_type = None

        # State list
        self.state_list = StateList()

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Camera():
    """
    Class containing camera parameters
    """

    def __init__(self):
        self.roi = Roi()                   # Region of interest to use for the camera
        self.transpose = False              # Are the image x/y coordinates flipped relative to the sample
        self.flip_x = False                 # Is the image flipped in x by the system relative to the sample
        self.flip_y = False                 # Is the image flipped in y by the system relative to the sample
        self.pixel_size_um = None           # Camera pixel size (no magnification or color factors, physical size)
        self.is_color = None                # Is this a color camera
        self.bayer_coupling_matrix = None   # Color coupling matrix
        self.device_name = None             # Camera device name
        self.bit_depth = 16                 # Camera bit depth used
        self.is_demosaiced = False          # Whether camera has been demosaiced

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Sample():
    """
    Class containing sample parameters
    """

    def __init__(self):
        self.roi = Roi()    # ROI of full sample, used for wide-FOV reconstrucitons
        self.name = None    # Sample name
        self.size_mm = None

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Objective():
    """
    Class containing objective lens parameters
    """

    def __init__(self):
        self.mag = None
        self.na = None

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Spectrum():
    """
    Class containing parameters for illumination spectrum
    """

    def __init__(self):
        self.center = None
        self.full = None
        self.units = "um"

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Illumination(HardwareElement):
    """
    Class containing illumination parameters
    """

    def __init__(self):
        HardwareElement.__init__(self)
        self.device_name = None
        self.device_type = None
        self.z_distance_mm = None
        self.bit_depth = 16
        self.is_color = False

        # LED Spectrum
        self.spectrum = Spectrum()


class Pupil(HardwareElement):
    """
    Class containing pupil parameters
    """

    def __init__(self):
        HardwareElement.__init__(self)
        # More to come


class StateList():
    """
    Generic state list class
    """

    def __init__(self):
        self.design = None
        self.calibrated = None
        self.grouping = None
        self.units = None

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Position(HardwareElement):
    """
    Class containing sample global position parameters
    """

    def __init__(self):
        HardwareElement.__init__(self)
        self.velocity_mm_s = None
        self.acceleration_mm_s2 = None
        self.units = 'mm'


class Focus(HardwareElement):
    """
    Class containing focus parameters
    """

    def __init__(self):
        HardwareElement.__init__(self)
        self.velocity_mm_s = None
        self.acceleration_mm_s2 = None


class System():
    """
    Class containing variables that don't go anywhere else, or are derived from several quantitites in other classes
    """

    def __init__(self):
        self.mag = 1.
        self.eff_pixel_size_um = None
        self.image_flip_x = False
        self.image_flip_y = False

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Background():
    """
    Class containing variables related to background subtraction
    """

    def __init__(self):
        self.max_thresh = 1000
        self.roi = Roi()
        self.is_subtracted = False

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class GroundTruth():
    """Class for ground truth"""

    def __init__(self):
        self.object = None
        self.parameters = None

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Reconstruction():
    """Class for reconstructed objects"""

    def __init__(self):

        # Search for datasets
        self.object = None
        self.parameters = None

    def __str__(self):
        return(objToString(self, text_color=Color.BLUE, use_newline=False))


class Reconstructions():
    """
    This class is a wrapper for
    """

    def __init__(self, parent_dataset):

        # Store parent dataset
        self.parent_dataset = parent_dataset

        # Find reconstructions
        self._findReconstructions()

    def _findReconstructions(self):
        """Search for datasets in the reconstruction directory."""

        if self.parent_dataset.directory is not None:
            # Generate reconstruction directory
            reconstruction_directory = os.path.join(self.parent_dataset.directory, 'reconstructions')

            # Generate tuples of json and .tif files for parameters and recon data respectivly.
            self.reconstruction_file_pairs = []
            if os.path.exists(reconstruction_directory):
                # Find json files
                json_file_list = glob.glob(os.path.join(reconstruction_directory, '*.json'))

                # See if any of these paths have tiff files
                for json_file in json_file_list:
                    for tiff_extension in ('.tif', '.tiff'):
                        if os.path.exists(os.path.splitext(json_file)[0] + tiff_extension):
                            self.reconstruction_file_pairs.append((json_file, os.path.splitext(json_file)[0] + tiff_extension))

    def __len__(self):
        return len(self.reconstruction_file_pairs)

    def __getitem__(self, i):
        """Return a reconstruction object which contains both the object and reconstruction parameters used."""

        # Generate recon parameters
        recon = Reconstruction()

        # Load recon parameters
        with open(self.reconstruction_file_pairs[i][0]) as data_file:
            recon.parameters = json.load(data_file)

        # Load recon object
        recon.object = readMultiPageTiff(self.reconstruction_file_pairs[i][1])
        recon.object = yp.astype(recon.object[0], 'float32')

        # Normalize to correct values
        min_value = recon.parameters['file_scale']['min_value']
        max_value = recon.parameters['file_scale']['max_value']
        recon.object -= yp.min(recon.object)
        recon.object /= yp.max(recon.object)
        recon.object *= (max_value - min_value)
        recon.object += min_value

        # Store label
        recon.label = recon.parameters['label']

        return recon
