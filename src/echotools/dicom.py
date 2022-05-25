import numpy as np
import pydicom
import cv2
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from skimage.transform import resize
from typing_extensions import TypedDict
from pydicom.uid import ExplicitVRLittleEndian

from pydicom.pixel_data_handlers.util import convert_color_space
from image import to_grayscale, HSVFilters, to_pil
from ecg import get_extractor
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
from scipy.io import loadmat, matlab
from scipy.io.matlab.mio5_params import mat_struct
from scipy import ndimage


def find_nearest(source, target):
    return ((np.expand_dims(target, -1) - source) ** 2).argmin(1)


class DicomMetadata(TypedDict):
    path: str
    patient_name: str
    mrn: str
    phn: str
    dob: str
    sex: str
    study_date: str
    manufacturer: str
    model: str
    physical_delta_x: float
    physical_delta_y: float
    transducer_frequency: int


class Dicom:

    def __init__(self, file_name: str, threshold=1 / 800, **kwargs):
        """

        :param file_name:
        :param threshold:
        """
        self.file_name = file_name
        self.is_mat = False
        if file_name.endswith('.dcm'):
            self.dicom = pydicom.read_file(file_name)
            if 'PixelData' in self.dicom:
                self.video = self.dicom.pixel_array
                self.contains_pixel_data = True
            else:
                self.video = None
                self.contains_pixel_data = False
        elif file_name.endswith('.mat'):
            self.dicom = _loadmat(file_name)
            self.is_mat = True
            self.video = self.dicom['Patient']['DicomImage'].transpose((3, 0, 1, 2))
        else:
            raise ValueError(f'Invalid file name {file_name}, not .dcm or .mat')

        self._mask = None
        self._bounding_box = None
        self.threshold = threshold
        self._ecg = None
        self.is_video = self.dicom_is_video()

    def check_contains_pixel_data(self):
        """
        :return: True if DICOM file contains pixel data
        """
        return self.contains_pixel_data

    def dicom_is_video(self):
        """
        :return: True if DICOM file is a video
        """
        if "NumberOfFrames" in self.dicom:
            return self.dicom.NumberOfFrames > 1
        else:
            return False

    def get_doppler_threshold(self):
        manufacturer = self.try_get_metadata(self.dicom, "Manufacturer")
        model = self.try_get_metadata(self.dicom, "ManufacturerModelName")
        if manufacturer=='SIEMENS':
            if model=='S2000':
                return 0.007
            else:
                return 0.003
        elif manufacturer=="TOSHIBA_MEC_US":
            if model=="TUS-A500":
                return 0.004
            else:
                return 0.003
        return 0.003

    def convert_colourspace(self):
        """
        Converts colourspace of the image to RGB
        """
        if not self.is_mat:
            colour_space = self.try_get_metadata(
                self.dicom, 'PhotometricInterpretation')
            if "YBR" in colour_space:
                arr = convert_color_space(self.dicom.pixel_array,
                                          self.dicom.PhotometricInterpretation,
                                          'RGB')
                self.video = arr
                self.dicom.PhotometricInterpretation = 'RGB'

    def try_get_metadata(self, dataset, attribute):
        """
        Extracts desired attribute from given dataset

        :param dataset: dataset to extract metadata from
        :param attribute: attribute to extract
        :return: value of desired attribute if present
        """
        if attribute in dataset:
            return dataset[attribute].value
        else:
            return ''


    def metadata(self) -> DicomMetadata:
        """
        :return: Standard metadata for this dicom file.
        """
        if self.is_mat:
            info = self.dicom['Patient']['DicomInfo']
            metadata: DicomMetadata = dict(
                path=self.file_name,
                frames=info['NumberOfFrames'],
                patient_id=info['PatientID'],
                heart_rate=info['HeartRate'],
                frame_time=info['FrameTime'],
                date=info['StudyDate'],
                machine=info['ManufacturerModelName']
            )
        else:
            # noinspection PyUnresolvedReferences
            if 'SequenceOfUltrasoundRegions' in self.dicom:
                regions = self.dicom.SequenceOfUltrasoundRegions.pop()
            else:
                regions = ""
            metadata: DicomMetadata = dict(
                path=self.file_name,
                patient_name=self.try_get_metadata(self.dicom, 'PatientName'),
                mrn=self.try_get_metadata(self.dicom, 'PatientID'),
                phn=self.try_get_metadata(self.dicom, 'OtherPatientIDs'),
                dob=self.try_get_metadata(self.dicom, 'PatientBirthDate'),
                sex=self.try_get_metadata(self.dicom, 'PatientSex'),
                study_date=self.try_get_metadata(self.dicom, 'StudyDate'),
                manufacturer=self.try_get_metadata(self.dicom, 'Manufacturer'),
                model=self.try_get_metadata(self.dicom, 'ManufacturerModelName'),
                physical_delta_x=self.try_get_metadata(regions, 'PhysicalDeltaX'),
                physical_delta_y=self.try_get_metadata(regions, 'PhysicalDeltaY'),
                transducer_frequency=self.try_get_metadata(regions, 'TransducerFrequency')
                # study_date=self.try_get_metadata(regions, 'TransducerFrequency')
            )
            if regions:
                self.dicom.SequenceOfUltrasoundRegions.append(regions)

        return metadata

    def anonymize(self):
        """
        Removes private and other desired fields from DICOM metadata
        """
        type_3_fields = ['OtherPatientIDs',
                         'SendingApplicationEntityTitle',
                         'SpecificCharacterSet',
                         'PatientID',
                         'PatientBirthDate',
                         'PatientSex',
                         'ImageType',
                         'ContentDate',
                         'ContentTime',
                         'StudyTime',
                         'AccessionNumber',
                         'Modality',
                         'InstitutionName',
                         'ReferringPhysicianName',
                         'StationName',
                         'StudyDescription',
                         'InstitutionalDepartmentName',
                         'NameOfPhysiciansReadingStudy',
                         'OperatorsName',
                         'PatientName',
                         'StageName',
                         'StageNumber',
                         'NumberofStages',
                         'NumberofViewsinStage',
                         'SoftwareVersions',
                         'PatientBirthDate',
                         'PatientSex',
                         'OtherPatientIDs',
                         'PatientSize',
                         'PatientWeight',
                         'PatientComments',
                         'DeviceSerialNumber',
                         'SoftwareVersion',
                         'TriggerTime',
                         'TransducerType',
                         'SeriesInstanceUID'
                         'StudyID',
                         'InstanceNumber',
                         'PatientOrientation',
                         'ImagesinAquisition',
                         'LossyImageCompression',
                         'RequestingPhysician']

        self.dicom.remove_private_tags()
        for field in type_3_fields:
            if field in self.dicom:
                delattr(self.dicom, field)

    def pydicom(self):
        return self.dicom

    def save(self, filename):
        self.dicom.save_as(filename)

    @property
    def ecg(self):
        """
        Get the raw ECG signal
        :return: (x_ecg, y_ecg), (x_samples, y_samples)
        Returns the raw extracted signal as well as a sample taken for each frame in the cine
        """
        if self._ecg is None:
            self._ecg = get_extractor(self.metadata()['machine'])(self.video).extract()
        return self._ecg

    def peak_frames(self, prepend=10, quantile=.98, distance=50, return_all=False):
        """
        Returns all peaks in the ECG sorted by height
        Tuning options
        :param prepend: int >= 0 [default = 10], h
        ow many elements to prepend your signal with (y <- [y.mean()] * prepend + y)
        useful for signals that start with a peak cut off
        :param quantile: float in [0, 1] [default = .98], what quantile the peaks must be above
        :param distance: int > 0 [default = 50], minimum number of pixels to separate each peak
        :param return_all: bool, option for returning all ecg data instead of just the frames
        :return: frames of original video nearest to detected peaks
        """
        (x, y), points = self.ecg
        if prepend:
            y = np.concatenate([[y.mean()] * prepend, y])
        peaks, properties = find_peaks(y, distance=distance, height=np.quantile(y, quantile))
        peaks -= prepend
        peaks = peaks[properties['peak_heights'].argsort()]
        peaks = peaks[peaks >= 0]
        if return_all:
            return find_nearest(points[0], x[peaks]), peaks, x, y[prepend:], points
        return find_nearest(points[0], x[peaks])

    def plot_ecg(self, **kwargs):
        frames, peaks, x, y, points = self.peak_frames(**kwargs, return_all=True)

        plt.figure(figsize=(20, 5))
        plt.plot(x, savgol_filter(y, 11, 3), linewidth=3)
        for p in points[0]:
            plt.axvline(p, c='red', linewidth=.5)
        plt.scatter(x[peaks], y[peaks], c='green', s=500)
        plt.title(f'Peak{"s" if len(frames) > 1 else ""}: {", ".join(map(str, frames.tolist()))}')
        plt.show()

    def find_mask(self, threshold, index=-2):
        if self.is_video:
            v = to_grayscale(self.video[(0 if len(self.video) % 2 == 0 else 1):] / 255)
            v1, v2 = v.reshape(len(v) // 2, 2, *v.shape[1:]).transpose(1, 0, 2, 3)
            diff_mean = (np.clip(v1 - v2, 0, 1)).mean(0)
        else:
            v = to_grayscale(self.video / 255)
            diff_mean = (np.clip(v, 0, 1))
        # find connected components after threshold
        cc = label(diff_mean > threshold)
        v, c = np.unique(cc, return_counts=True)
        # find second largest connect component (largest is the background)
        second_largest_component = v[c.tolist().index(sorted(c)[index])]

        # take convex hull to remove small gaps
        # noinspection PyTypeChecker
        return convex_hull_image(np.where(cc == second_largest_component, 1, 0))

    def remove_patient_info(self):
        """
        Masks out private sections of the DICOM file. This is in addition
        to masked beam in case not all patient information is hidden.
        """
        manufacturer = self.try_get_metadata(self.dicom, "Manufacturer")
        model = self.try_get_metadata(self.dicom, "ManufacturerModelName")
        error_message = "{}, {} manufacturer and model unknown".format(manufacturer, model)

        if self.is_video:
            mask = np.ones((self.video.shape[1], self.video.shape[2]))
        else:
            mask = np.ones((self.video.shape[0], self.video.shape[1]))

        if manufacturer=='Philips Medical Systems':
            if model=='iE33' or model=='iU22' or model=='CX50':
                mask[0:int(mask.shape[0]*1/10), :] = 0
            elif model=='SONOS':
                mask[0:int(mask.shape[0]*0.4), 0:int(mask.shape[1]*1.2/5)] = 0
            elif model=='EPIQ 7C' or model=='EPIQ 7G' or model=='EPIQ 5G' or model=='Affiniti 70G':
                mask[0:int(mask.shape[0]*0.06), :] = 0
            elif model=='IntelliSpace PACS Radiology':
                mask[:, :] = 0
            else:
                print(error_message)
                mask[:, :] = 0
        elif manufacturer=='GE Healthcare' or manufacturer=='GE Vingmed Ultrasound' or manufacturer=='GEMS Ultrasound':
            if model=='Vivid i':
                mask[0:int(mask.shape[0]*0.061), 0:int(mask.shape[1]*2/5)] = 0
            elif model=='Vivid E9':
                mask[0:int(mask.shape[0]*1/9.5), :] = 0
            elif model=='Vivid E95':
                mask[0:int(mask.shape[0]*1/9.5), :] = 0
            elif model=='EchoPAC PC' or model=='Vivid7':
                mask[0:int(mask.shape[0]*2/5), 0:int(mask.shape[1]*1.2/5)] = 0
            elif model=='Vivid S6' or model=='LOGIQE10':
                pass
            elif model=='LOGIQE9':
                mask[0:int(mask.shape[0]*0.1), :] = 0
            elif model=='EchoPAC PC SW-Only':
                mask[0:int(mask.shape[0]*2/5), 0:int(mask.shape[1]*1.4/5)] = 0
            else:
                print(error_message)
                mask[:, :] = 0
        elif manufacturer=='ACUSON':
            if model=='SEQUOIA':
                mask[0:int(mask.shape[0]*0.061), 0:int(mask.shape[1]*3/5)] = 0
            else:
                print(error_message)
                mask[:, :] = 0
        elif manufacturer=='TOSHIBA_MEC_US':
            if model=='TUS-A500':
                if 'SECONDARY' in self.try_get_metadata(self.dicom, 'ImageType'):
                    mask[:, :] = 0
                else:
                    mask[0:int(mask.shape[0]*0.08), 0:int(mask.shape[1]*85/100)] = 0
            elif model=='TUS-AI800':
                mask[0:int(mask.shape[0]*0.08), 0:int(mask.shape[1]*85/100)] = 0
            else:
                print(error_message)
                mask[:, :] = 0
        elif manufacturer=='SIEMENS':
            if model=='S2000':
                mask[0:int(mask.shape[0]*0.075),0:int(mask.shape[1]*2/5)] = 0
            elif model=='Definition AS+' or model=='Sensation Cardiac 64' or model=='Sensation 64':
                mask[:, :] = 0
            else:
                print(error_message)
                mask[:, :] = 0
        elif manufacturer=='Siemens Medical Systems - Ultrasound Division':
            if model=='Antares':
                mask[0:int(mask.shape[0]*0.05), :] = 0
            else:
                print(error_message)
                mask[:, :] = 0
        elif manufacturer=='CANON_MEC':
            if model=='CUS-AA450':
                mask[0:int(mask.shape[0]*0.08), 0:int(mask.shape[1]*85/100)] = 0
            else:
                print(error_message)
                mask[:, :] = 0
        elif manufacturer=='PACSGEAR' or manufacturer=='Hyland' \
                or manufacturer=='Lexmark' or manufacturer=='INTELERAD MEDICAL SYSTEMS':
            mask[:, :] = 0
        else:
            print(error_message)
            mask[:, :] = 0
        if len(self.video.shape)==3 and self.video.shape[-1]==3:
            self.video = np.expand_dims(mask, -1) * self.video
        elif len(self.video.shape)==3 and self.video.shape[-1]!=3:
            self.video = np.expand_dims(mask, 0) * self.video
        elif len(self.video.shape)==4:
            mask = np.expand_dims(mask, 0)
            self.video = np.expand_dims(mask, -1) * self.video
        else:
            self.video = mask * self.video
        self.dicom.PixelData = self.video.tobytes()

    @property
    def mask(self, **kwargs):
        """
        Find a binary mask for pixels inside the ultrasound beam
        :return: Numpy {0, 1} array of the same dimension as a video frame
        """
        if self._mask is None:

            # noinspection PyTypeChecker
            self._mask = self.find_mask(self.threshold)
            total = np.product(self._mask.shape)
            covering = self._mask.sum() / total
            com = ndimage.measurements.center_of_mass(self._mask)

            if covering < .15:
                self._mask = self.find_mask(self.threshold / 2)
                covering = self._mask.sum() / total
            if covering > .9 or com[0] < self._mask.shape[0]/4:
                self._mask = self.find_mask(self.threshold, -1)
                covering = self._mask.sum() / total

            if covering > .9 or covering < .15 or com[0] < self._mask.shape[0]/4:
                raise RuntimeError(f'Error cleaning {self.file_name} ' + str(covering))
        return self._mask

    @property
    def bounding_box(self):
        """
        Find the minimum square bounding box around the generated mask
        :return: bounding box coordinates on original image, x1, y1, x2, y2
        """
        if self._bounding_box is None:
            regions = regionprops(self.mask.astype('uint8'))
            bbox = regions[0].bbox
            y1, x1, y2, x2 = bbox
            self._bounding_box = x1, y1, x2, y2
        return self._bounding_box

    def clean_frame(self, frame=0, crop=True):
        if crop:
            x1, y1, x2, y2 = self.bounding_box
            return to_pil(HSVFilters.color_filter(self.video[frame] * np.expand_dims(self.mask, -1))[y1:y2, x1:x2])
        return to_pil(HSVFilters.color_filter(self.video[frame] * np.expand_dims(self.mask, -1)))

    def comparison(self, frame=0):
        cf = self.clean_frame(frame, crop=False)
        of = self.video[frame]
        return to_pil(np.concatenate((of, cf), 1))

    def pil_mask(self):
        return to_pil(self.mask)

    def check_for_doppler(self, gray_mask, colour_mask):
        doppler_thresold = self.get_doppler_threshold()
        if colour_mask.shape[-1] == 3:
            if (np.abs(np.stack([gray_mask] * 3, -1) / 255 - colour_mask / 255)).mean() > doppler_thresold:
                raise ValueError('Color Doppler Detected. Cleaning of this video is not supported: ' + str((np.abs(np.stack([gray_mask] * 3, -1) / 255 - colour_mask / 255)).mean()))

    def masked_video(self, crop=True, resized=(224, 224), grayscale=True, exclude_doppler=False, **kwargs) -> np.ndarray:
        """
        Return the cine with UI elements removed [F x H x W x 3]
        :param crop: bool, If true crops black space around the beam
        :param resized: Optional[Tuple[int, int]], If not none interpolates each frame to dimension resized
        :param grayscale: bool, if True converts the output to grayscale
        :raises ValueError for videos with color doppler
        """
        if(self.is_video):
            g = to_grayscale(self.video.mean(0) / 255)
        else:
            g = to_grayscale(self.video / 255)

        com = ndimage.measurements.center_of_mass(self.mask)

        if(com[0] > 3 * self.mask.shape[0]/4):
            raise ValueError('COM too low')

        mask = self.mask

        if len(self.video.shape)==3 and self.video.shape[-1]==3:
            masked_video = np.expand_dims(mask, -1) * self.video
        elif len(self.video.shape)==3 and self.video.shape[-1]!=3:
            masked_video = np.expand_dims(mask, 0) * self.video
        elif len(self.video.shape)==4:
            mask = np.expand_dims(mask, 0)
            masked_video = np.expand_dims(mask, -1) * self.video
        else:
            masked_video = mask * self.video

        if exclude_doppler:
            self.check_for_doppler(to_grayscale(masked_video), masked_video)

        if crop:
            x1, y1, x2, y2 = self.bounding_box
            masked_video = masked_video[:, y1:y2, x1:x2]
            dx, dy = x2 - x1, y2 - y1
            pad = np.abs(dx - dy) // 2
            pad_pattern = [[0, 0]] * 4
            if dx > dy:
                pad_pattern[1] = [pad, pad]
            elif dy > dx:
                pad_pattern[2] = [pad, pad]
            masked_video = np.pad(masked_video, ((0, 0), (30, 30), (0, 0), (0, 0)), 'constant', constant_values=0)
        if resized:
            masked_video = resize(masked_video, (masked_video.shape[0], *resized))
        if grayscale:
            return to_grayscale(masked_video)
        return masked_video

    def mask_pixel_array(self, crop=True, resized=(224, 224), grayscale=True, exclude_doppler=False, **kwargs) -> np.ndarray:
        """
        Changes pixel array to the cine with UI elements removed [F x H x W x 3]

        :param crop: bool, If true crops black space around the beam
        :param resized: Optional[Tuple[int, int]], If not none interpolates each frame to dimension resized
        :param grayscale: bool, if True converts the output to grayscale
        :raises ValueError for videos with color doppler
        """
        mask = self.masked_video(crop=crop, resized=resized, grayscale=grayscale, exclude_doppler=exclude_doppler)
        self.dicom.PixelData = mask.tobytes()
        self.video = mask

    def create_dopper_differential(self):
        grey_video = to_grayscale(self.video)
        l, h, w = grey_video.shape
        differential = np.zeros((l-1, h, w))
        for i in range(l-1):
            differential[i, :, :] = cv2.absdiff(grey_video[i], grey_video[i+1])
        differential = np.stack((differential, differential, differential), axis=-1)
        
        self.dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        self.dicom.PixelData = differential.tobytes()
        self.video = differential


def _to_even(x):
    if x % 2 == 0:
        return x
    return x - 1


def _loadmat(filename):
    """
    this function should be called instead of direct loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    """
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(_dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in _dict:
        if isinstance(_dict[key], mat_struct):
            _dict[key] = _todict(_dict[key])
    return _dict


def _todict(matobj):
    """
    A recursive function which constructs from mat objects nested dictionaries
    """
    _dict = {}
    # noinspection PyProtectedMember
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mat_struct):
            # noinspection PyTypeChecker
            _dict[strg] = _todict(elem)
        else:
            _dict[strg] = elem
    return _dict
