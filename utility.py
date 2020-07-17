#utility.py
from numpy import *
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from cv2 import *
import numpy as np
import SimpleITK as sitk


"""self.file_list就是dicom文件序列"""


def show_image(lable, matrix_, seg, seg_flag):
    minval = matrix_.min()
    tmpImage = (matrix_ - minval).astype('float')
    if tmpImage.max() != 0:
        rescaledImage = ((tmpImage / tmpImage.max()) * 255.0).astype(np.uint8)#归一化
    else:
        rescaledImage = tmpImage.astype(np.uint8)
    #rescaledImage = rescaledImage.resize(())
    h, w = rescaledImage.shape

    if seg_flag:
        threshold = 0.4
        value = [174, 26, 26]#朱砂色

        print(seg.shape, rescaledImage.shape)
#把图像和label转换成Mat
        rescaledMat = cvtColor(rescaledImage, COLOR_GRAY2RGB)
        labelMat = merge([mat(((seg>threshold)*value[0]).astype(np.uint8)),
                          mat(((seg>threshold)*value[1]).astype(np.uint8)),
                          mat(((seg>threshold)*value[2]).astype(np.uint8))])
#将label合并到图像上
        print(rescaledMat.shape, labelMat.shape)
        FinalImg = cv2.add(rescaledMat, labelMat)

        pixmap = QtGui.QImage(FinalImg, w, h, w*3, QtGui.QImage.Format_RGB888)

        lb = lable
        lb.setPixmap(QPixmap.fromImage(pixmap))
        lb.resize(w, h)
        lb.show()

    if not seg_flag:
        rescaledMat = cvtColor(rescaledImage, COLOR_GRAY2RGB)
        pixmap = QtGui.QImage(rescaledMat, w, h, w*3, QtGui.QImage.Format_RGB888)
        # pixmap.scaled(137, 512)
        lb = lable
        lb.setPixmap(QPixmap.fromImage(pixmap))
        lb.resize(w, h)
        lb.show()

def Preprocess(matrix3D):
    #插值
    matrix3D = resample(matrix3D, sitk.sitkLinear)

    #padding
    size = matrix3D.GetSize()
    pad = 680 - size[1]
    if pad % 2 == 1:
        lowbound = ((int)(pad / 2), (int)(pad / 2), 0)
        upbound = ((int)(pad / 2) + 1, (int)(pad / 2) + 1, 0)
    else:
        lowbound = ((int)(pad / 2), (int)(pad / 2), 0)
        upbound = ((int)(pad / 2), (int)(pad / 2), 0)
    print(pad)
    stats_i = sitk.ConstantPadImageFilter()
    stats_i.SetConstant(-200)
    stats_i.SetPadLowerBound(lowbound)
    stats_i.SetPadUpperBound(upbound)
    matrix3D = stats_i.Execute(matrix3D)

    # 灰度截断
    matrix3D = sitk.GetArrayFromImage(matrix3D)
    matrix3D[matrix3D > 200] = 200
    matrix3D[matrix3D < -200] = -200
    return matrix3D

def resample(image, interpolators, new_spacing=np.array([1, 1, 1], dtype=np.float64)):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample

    # VectorUInt32 size
    inputSpacing = np.array(image.GetSpacing())  # dtype = float64
    inputSize = np.array(image.GetSize())  # dtype = uint32
    real_size = inputSpacing * inputSize  # dtype = float32 , 'millimeter'
    #print(inputSize, real_size)
    outputSize = real_size / new_spacing  # dtype = float32
    outputSize = np.ceil(outputSize)  # dtype = float32
    outputSize = tuple(map(int, tuple(outputSize)))  # dtype = int

    # Transform transform
    dimension = image.GetDimension()
    transform = sitk.Transform(dimension, sitk.sitkIdentity)

    # itk::simple::InterpolatorEnum interpolator
    interpolator = interpolators

    # VectorDouble outputOrigin
    inputOrigin = np.array(image.GetOrigin())
    outputOrigin = inputOrigin - 0.5 * inputSpacing + 0.5 * new_spacing
    outputOrigin = tuple(outputOrigin)

    # VectorDouble outputSpacing
    outputSpacing = new_spacing

    # VectorDouble outputDirection
    outputDirection = image.GetDirection()

    # double defaultPixelValue
    img_arr = sitk.GetArrayFromImage(image)
    defaultPixelValue = img_arr.min()
    defaultPixelValue = float(defaultPixelValue)

    # itk::simple::PixelIDValueEnum outputPixelType
    outputPixelType = image.GetPixelID()

    '''
    Resample(Image image1, VectorUInt32 size, Transform transform, itk::simple::InterpolatorEnum interpolator, 
    VectorDouble outputOrigin, VectorDouble outputSpacing, VectorDouble outputDirection, 
    double defaultPixelValue=0.0, itk::simple::PixelIDValueEnum outputPixelType) -> Image
    '''
    return sitk.Resample(image, outputSize, transform, interpolator,
                         outputOrigin, outputSpacing, outputDirection,
                         defaultPixelValue, outputPixelType)

def ReadSegment(root):
    segment_files = [f for f in os.listdir(root) if 'segment_' in f]
    segment_files.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
    print("lenght of segment file: ", len(segment_files))
    total_seg = []
    for file in segment_files:
        seg = np.load(file)
        destroyWindow(file)
        total_seg.append(seg)
    total_seg = np.array(total_seg).squeeze()
    print("total shape: ", total_seg.shape)
    return total_seg

a = ReadSegment(os.getcwd())
print(a.shape)