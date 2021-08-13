# Open Kidney Dataset
## Fine-grained Annotated Ultrasound for Medical Image Analysis

### Introduction

Ultrasound imaging is a portable, real-time, non-ionizing, and non-invasive imaging modality. It is the first line for numerous organs, including the kidney. With recent advances in technology, the world of artificial intelligence (AI)-enhanced ultrasound is imminently upon us. However, compared to other modalities like CT or MRI, there is a lack of open ultrasound data available for researchers to use.

We present the Open Kidney Dataset. It includes over ### two-dimensional B-mode abdominal ultrasound images and two sets of fine-grained polygon annotations from four classes that are available for non-commericial use. 

![Image](src)

### Motivation
Artificial intelligence for medical imaging has seen unprecedented growth in the last ten years. As a result of the creation of imaging data being made available to researchers, cornerstone algorithms like U-net have been created. However, in the field of ultrasound, there is a lack of data available. This is in part due to difficulty in acessing medical imaging data as well as anonymization and privacy considerations. However, even in competititions within biomedical imaging such as the The MICCAI Segmentation Decathalon, ultrasound is underrpresented. The lack of data accentuates the growing reproducibility crisis within the ultrasound machine learning field. To the best of our knowledge there is no widely available kidney ultrasound dataset that exists.

To further expand and improve academic efforts for machine learning in ultrasound, we present the Open Kidney Dataset.

This dataset may provide standardization to ultrasound segmentation benchmarking, as well as in the long-term reduce ultrasound interpretation efforts, furthering simplifying ultrasound use


### Data Description
<placeholder>
  REB approval. Time frame. Adults only. Vendors included are . No imaging acquisition metadata included or patient-level data. Data pre-processing methods: Anonymization will be performed including the cropping of metadata included on the image. No images with biometric labels will be included in the dataset.
  
  Each annotated image additionally comes with an ordinal quality label, as well as labels for the view type and kidney type (native or transplant).

  
  An example data record includes....
  
Annotations. Two sonographers with >30 years of experience. Definition of quality, view, and the classes were made initially. Iterated after a set of 20 images were provided and used for practice annotations
Updated definitions thereafter
Define:
Quality [unacceptable, poor, fair, good]
View [transverse, longitudinal, other]
Classes [capsule, cortex, medulla, central echogenic complex]
Manual annotation using VGG Image Annotator (VIA) 
Annotations are hand-drawn polygons using the VGG Annotation Tool performed by two[RS3]  expert sonographers. Expert sonographers assessed image quality first, and only annotated images with good or excellent image quality. All annotations will be reviewed by R Singla. 
Verification of labels and annotations as well as consensus for discrepancies
Are there specific annotation instructions? E.g. closed contours only, annotate things larger than X cm
are ultrasound images taken from different machines? How varied are the settings? Do certain images have the kidney cut off? Are they all of standardized quality? Are there shadows from ribs cutting off the kidney? Is it a single closed contour?

### License

### Access
1. <placeholder for Research Use Agreement?>
2. <placeholder for collecting information via MailChimp or a Qualtrics embedded form> 
  
### Code and Trained Models
1. Preprocessing code for the ultrasound images is included. Anonymization and conversion from DICOM to PNG.[Link](https://www.google.com/)

2. An nnU-net model, data splits and modifications are included ...[Link](https://www.google.com/)
  
### Supplemental Material
In case anything else is needed. Include the annotation guidelines and rules, as well as sources of error and errata. 
  
  
### Citation
<placeholder for paper/arXiv submission>
   
### Support or Contact
For additional information, or to report errors in the data, please contact us at rsingla [at] ece [dot] ubc [dot] ca 
