import SimpleITK as sitk
import os
import numpy as np
import shutil
import cv2

DIR = 'E:/Data_set'
DATA_PATH = os.path.join(DIR, 'LITS')
THRESHOLD_DIR = os.path.join(DIR, 'LITS_T')
OUTPUT_DIR = os.path.join(DIR, 'LITS_R')
PADDING_DIR = os.path.join(DIR, 'LITS_P')
AUGMENT_DIR = os.path.join(DIR, 'LITS_A')
SAMPLE_DIR = os.path.join(DIR, 'SAMPLE')
SAMPLE_N_DIR = os.path.join(DIR, 'SAMPLE_N')


def pickimg(root, structures):
    image_list = [os.path.join(root, f, 'img.nrrd') for f in sorted(os.listdir(root))]
    label_list = []


    for f in sorted(os.listdir(root)):
        label_list.append([os.path.join(root, f, 'structures', structure + '.nrrd')for structure in structures])

    if len(image_list) != len(label_list):
        print("images and labels do not match")

    print(len(image_list), len(label_list))
    data = []

    for (i, ls) in zip(image_list, label_list):
        img = sitk.ReadImage(i)
        save_flag = True

        for l in ls:
            #如果一类label不存在则整个病例丢弃
            if not os.path.exists(l):
                save_flag = False
                break

            label = sitk.ReadImage(l)
            if img.GetSpacing()[2] > 5.0 or label.GetSpacing()[2] > 5.0:  # 如果层间距<=5.0mm则记录文件位置
                save_flag=False

        if save_flag:
            data.append((i, ls))

    print(len(data))
    return data


def go_resample(data):
    for (i, ls) in data:
        image = sitk.ReadImage(i)

        #print("before resample:", image.GetSize(), image.GetSpacing(), image.GetDirection(), image.GetOrigin())
        image = resample(image, sitk.sitkLinear)

        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # 除了resample之外还重新设置了Direction和Oringin
        image.SetOrigin((0, 0, 0))

        label_arr = np.zeros(sitk.GetArrayFromImage(image).shape)
        labels = []

#得到label的数组，方便后续计算label_arr
        for l in ls:
            label = sitk.ReadImage(l)
            label = resample(label, sitk.sitkNearestNeighbor)
            labels.append(sitk.GetArrayFromImage(label))

        for count in range(len(labels)):
            label_arr[np.array(labels[count]) == 1] = count+1

        print(np.unique(label_arr))
        label_dcm = sitk.GetImageFromArray(label_arr)
        label_dcm.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        label_dcm.SetOrigin((0, 0, 0))

        if image.GetSize() != label_dcm.GetSize():
            print('imagesize:{0}, labelsize:{1}'.format(image.GetSize(), label_dcm.GetSize()))
        if image.GetOrigin() != label_dcm.GetOrigin():
            print('imageorigin:{0}, labelorigin:{1}'.format(image.GetSize(), label_dcm.GetSize()))
        if image.GetSpacing() != label_dcm.GetSpacing():
            print('imagespacing:{0}, labelspacing:{1}'.format(image.GetSize(), label_dcm.GetSize()))
        else:
            print('successfully {0}'.format(i.split('\\')[-2]))
        # print("after resample: ", image.GetSize(), image.GetSpacing())
        # print('Direction :{0} Origin: {1}'.format(image.GetDirection(),image.GetOrigin()),'\n')
        path = i.split('\\')[-2]
        path_l = ls[0].split('\\')[-3]

        sitk.WriteImage(image, os.path.join(OUTPUT_DIR, '{0}_image.mhd'.format(path)))
        sitk.WriteImage(label_dcm, os.path.join(OUTPUT_DIR, '{0}_label.mhd'.format(path_l)))


def resample(image, interpolators, new_spacing=np.array([1, 1, 2.5], dtype=np.float64)):
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


def threshold_image():
    image_list = [os.path.join(OUTPUT_DIR, f) for f in sorted(os.listdir(OUTPUT_DIR)) if 'image' in f and 'mhd' in f]
    label_list = [os.path.join(OUTPUT_DIR, f) for f in sorted(os.listdir(OUTPUT_DIR)) if 'label' in f and 'mhd' in f]

    for img_path in image_list:
        img = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img)
        '''
        w = np.where(img_arr<-200)
        for i in range(w[0].shape[0]):
            img_arr[w[0][i],w[1][i],w[2][i]] = -200

        h = np.where(img_arr>200)
        for j in range(h[0].shape[0]):
            img_arr[h[0][j],h[1][j],h[2][j]] = 200
        '''
        img_arr[img_arr < -200] = -200
        img_arr[img_arr > 200] = 200

        image = sitk.GetImageFromArray(img_arr)
        image.SetDirection(img.GetDirection())
        image.SetSpacing(img.GetSpacing())
        image.SetOrigin(img.GetOrigin())
        # print(image.GetSize(), image.GetSpacing())
        sitk.WriteImage(image, os.path.join(THRESHOLD_DIR, img_path.split('\\')[-1]))

    for l in label_list:
        label = sitk.ReadImage(l)
        print(np.unique(sitk.GetArrayFromImage(label)))

        sitk.WriteImage(label, os.path.join(THRESHOLD_DIR, l.split('\\')[-1]))

def padding_image():
    label_list = [os.path.join(THRESHOLD_DIR, f) for f in sorted(os.listdir(THRESHOLD_DIR)) if
                  'label' in f and 'mhd' in f]

    for l in label_list:
        label = sitk.ReadImage(l)
        stats = sitk.ConstantPadImageFilter()
        size = label.GetSize()
        pad = 680 - size[0]
        if pad % 2 == 1:
            lowbound = ((int)(pad / 2), (int)(pad / 2), 0)
            upbound = ((int)(pad / 2) + 1, (int)(pad / 2) + 1, 0)
        else:
            lowbound = ((int)(pad / 2), (int)(pad / 2), 0)
            upbound = ((int)(pad / 2), (int)(pad / 2), 0)
        print(lowbound, upbound)
        stats.SetConstant(0)
        stats.SetPadLowerBound(lowbound)
        stats.SetPadUpperBound(upbound)
        label_p = stats.Execute(label)
        sitk.WriteImage(label_p, os.path.join(PADDING_DIR, l.split('\\')[-1]))

        path = l.split('label')[0]
        image = sitk.ReadImage(os.path.join(THRESHOLD_DIR, '{0}image.mhd'.format(path)))
        stats_i = sitk.ConstantPadImageFilter()

        stats_i.SetConstant(-200)
        stats_i.SetPadLowerBound(lowbound)
        stats_i.SetPadUpperBound(upbound)
        image_p = stats_i.Execute(image)
        sitk.WriteImage(image_p, os.path.join(PADDING_DIR,
                                              os.path.join(PADDING_DIR, '{0}image.mhd'.format(path).split('\\')[-1])))

        print('after padding: image_size:{0}, label_size:{1}'.format(image_p.GetSize(), label_p.GetSize()))
        if image_p.GetSize() != label_p.GetSize():
            print('incorrelate size!')


def sample_all():
    image_list = [os.path.join(PADDING_DIR, f) for f in sorted(os.listdir(PADDING_DIR)) if
                  'image' in f and 'mhd' in f]
    label_list = [os.path.join(PADDING_DIR, f) for f in sorted(os.listdir(PADDING_DIR)) if
                  'label' in f and 'mhd' in f]

    # print(image_list)
    # print(label_list)

    for (i, l) in zip(image_list, label_list):
        image = sitk.ReadImage(i)
        label = sitk.ReadImage(l)
        label_arr = sitk.GetArrayFromImage(label)

        # print((i,l))
        # print(label.GetSize())
        # print(image.GetSize())
        for j in range(1, label.GetSize()[2] - 1):
            # print('{0}:{1}'.format(i.split('\\')[-1], j))
            id = i.split('\\')[-1]
            id = id.split('.')[0]

            id_l = l.split('\\')[-1]
            id_l = id_l.split('.')[0]

            #print(np.unique(label_arr))

            if np.sum(label_arr[j, :, :]) != 0:
                if not os.path.exists(os.path.join(SAMPLE_DIR, '{0}-{1}-t.mhd'.format(id, j))):
                    sitk.WriteImage(image[:, :, j - 1:j + 2], os.path.join(SAMPLE_DIR, '{0}-{1}-t.mhd'.format(id,"%03d" %j)))

                if not os.path.exists(os.path.join(SAMPLE_DIR, '{0}-{1}-t.mhd'.format(id_l, j))):
                    sitk.WriteImage(label[:, :, j], os.path.join(SAMPLE_DIR, '{0}-{1}-t.mhd'.format(id_l, "%03d" %j)))
            # else:
            #     sitk.WriteImage(image[:, :, j - 1:j + 2], os.path.join(SAMPLE_N_DIR, '{0}-{1}-f.mhd'.format(path, j)))
            #     sitk.WriteImage(label[:, :, j], os.path.join(SAMPLE_N_DIR, '{0}-{1}-f.mhd'.format(path_l, j)))


structures = ['BrainStem', 'Chiasm', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R', 'Submandibular_L', 'Submandibular_R']
#label有9类，加上背景label的值应该是0-9

#TODO： Chiasm已经预处理完毕

# # 筛选层间距<5.0mm的mhd文件
# data = pickimg('D:/Data_set/PDCDCA1.4.1/PDDCA-1.4.1', structures)
#
# # 线性插值得到指定spacing，做空间转换后保存到OUTDIR
# go_resample(data)
#
# # 对图像做阈值截断，保存到THRESHOLD_DIR，lable复制过去
# threshold_image()
#
# # img和lable都pad到680*680
# padding_image()

# 三个做一组保存：lable里有东西后缀为-t，没东西后缀为-f
sample_all()
