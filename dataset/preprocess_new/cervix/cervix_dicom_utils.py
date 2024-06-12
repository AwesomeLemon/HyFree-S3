'''
Preprocessing code from Vangelis
'''
import glob
import os
import os.path
import warnings
from functools import partial
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom

from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.exceptions import ContourOutOfBoundsException

def corrected_image_mask(outputs, img_mode, slices_to_remove):
    """
    With most patients, the doctors do not annotate the final slices. These slices need to be removed because if used
    to train a model they will mislead the model to not annotate regions it should annotate.

    Args:
        outputs: dictionary containing the images, their corresponding axial voxel spacing, and the mask
        img_mode: see convert_patient_images_from_dcm_to_nii
        slices_to_remove: number of slices to remove from the axial view

    Returns: outputs with corrected image(s) and mask

    """
    if slices_to_remove != 0 and img_mode != 'bffe':
        if img_mode in ['t2', 't2_bffe']:
            final_slices = slices_to_remove
        elif img_mode == 'bffe_t2':
            # take voxel spacing into account when removing slices in bffe_t2 mode
            final_slices = int(slices_to_remove * (outputs['image2_axial_vs'] / outputs['image1_axial_vs']))

        initial_image1, initial_mask = sitk.GetArrayFromImage(outputs['image1']), sitk.GetArrayFromImage(
            outputs['mask'])
        corrected_image1, corrected_mask = initial_image1[:-final_slices], initial_mask[:-final_slices]
        sitk_image1, sitk_mask = sitk.GetImageFromArray(corrected_image1), sitk.GetImageFromArray(corrected_mask)

        sitk_image1.SetSpacing(outputs['image1'].GetSpacing()), sitk_mask.SetSpacing(outputs['mask'].GetSpacing())
        sitk_image1.SetOrigin(outputs['image1'].GetOrigin()), sitk_mask.SetOrigin(outputs['mask'].GetOrigin())
        sitk_image1.SetDirection(outputs['image1'].GetDirection()), sitk_mask.SetDirection(
            outputs['mask'].GetDirection())

        assert sitk_image1.GetSize() == sitk_mask.GetSize(), 'Image and mask have different size.'
        assert sitk_image1.GetSpacing() == sitk_mask.GetSpacing(), 'Image and mask have different spacing.'
        assert sitk_image1.GetOrigin() == sitk_mask.GetOrigin(), 'Image and mask have different origin.'
        assert sitk_image1.GetDirection() == sitk_mask.GetDirection(), 'Image and mask have different direction.'

        outputs['image1'] = sitk_image1
        outputs['mask'] = sitk_mask

        if 'image2' in outputs.keys():
            initial_image2 = sitk.GetArrayFromImage(outputs['image2'])
            corrected_image2 = initial_image2[:-final_slices]
            sitk_image2 = sitk.GetImageFromArray(corrected_image2)

            sitk_image2.SetSpacing(outputs['image2'].GetSpacing())
            sitk_image2.SetOrigin(outputs['image2'].GetOrigin())
            sitk_image2.SetDirection(outputs['image2'].GetDirection())

            assert sitk_image1.GetSize() == sitk_image2.GetSize(), 'Image1 and Image2 have different size.'
            assert sitk_image1.GetSpacing() == sitk_image2.GetSpacing(), 'Image1 and Image2 have different spacing.'
            assert sitk_image1.GetOrigin() == sitk_image2.GetOrigin(), 'Image1 and Image2 have different origin.'
            assert sitk_image1.GetDirection() == sitk_image2.GetDirection(), 'Image1 and Image2 have different direction.'

            outputs['image2'] = sitk_image2

        print('        size after correction:', sitk_image1.GetSize())

    return outputs


def dcmrtstruct2nii(img1_path, target_labels, struct_path, img2_path=None):
    """
    Extract all the images and corresponding contours from rtstruct file

    Args:
        img1_path: path to current patient folder with dicom slices
        target_labels: dictionary with class_index -> class_label as keys, values
        struct_path: path to dicom rtstruct path
        img2_path: path to additional patient folder with dicom slices (optional)

    Returns: dictionary containing the image(s) and the mask

    """
    dicom_image = DcmInputAdapter().ingest(img1_path)
    class_masks = extract_rtstructs(dicom_image, struct_path, target_labels)

    masks_list = []
    for i in target_labels.keys():
        masks_list.append(sitk.GetArrayFromImage(class_masks[i]))

    # combine class masks to a single mask
    combined_mask = np.stack(masks_list)
    combined_mask_final = np.argmax(combined_mask, axis=0)
    mask = sitk.GetImageFromArray(combined_mask_final)
    mask.CopyInformation(dicom_image)

    outputs = {'image1': dicom_image, 'mask': mask, 'image1_axial_vs': dicom_image.GetSpacing()[-1]}

    if img2_path:
        dicom_2_image = DcmInputAdapter().ingest(img2_path)
        dicom_2_resampled = sitk.Resample(image1=dicom_2_image, size=dicom_image.GetSize(), transform=sitk.Transform(),
                                          interpolator=sitk.sitkLinear, outputOrigin=dicom_image.GetOrigin(),
                                          outputSpacing=dicom_image.GetSpacing(),
                                          outputDirection=dicom_image.GetDirection(),
                                          defaultPixelValue=0, outputPixelType=dicom_2_image.GetPixelID())
        dicom_2_resampled.CopyInformation(dicom_image)

        outputs['image2'] = dicom_2_resampled
        outputs['image2_axial_vs'] = dicom_2_image.GetSpacing()[-1]

    print('        size:', dicom_image.GetSize())
    print('        spacing:', dicom_image.GetSpacing())
    print('        origin:', dicom_image.GetOrigin())
    print('        direction:', dicom_image.GetDirection())

    return outputs


def extract_rtstructs(dicom_image, rtstruct_file_path, target_labels, mask_background_value=0, mask_foreground_value=1):
    """
    Extract contours from rtstruct files and convert them to simple itk masks

    Args:
        dicom_image: simple itk image
        rtstruct_file_path: path to rtstruct file
        target_labels: dictionary with class_index -> class_label as keys, values
        mask_background_value: value for background pixels
        mask_foreground_value: value for foreground pixels

    Returns: dictionary containing all the simple itk masks of contours as values, and corresponding class_index as keys

    """
    rtreader = RtStructInputAdapter()
    rtstructs = rtreader.ingest(rtstruct_file_path)
    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    dummy_zeros_mask_np = np.zeros_like(sitk.GetArrayFromImage(dicom_image)).astype('uint8')
    class_masks = {}

    for label_idx in target_labels:
        current_rtstruct = [rtstruct for rtstruct in rtstructs if rtstruct['name'] == target_labels[label_idx]]
        if current_rtstruct:
            assert len(current_rtstruct) == 1
            if 'sequence' not in current_rtstruct[0]:
                warnings.warn(f"Mask for {current_rtstruct[0]['name']} will be empty. No shape/polygon found.")
                mask = sitk.GetImageFromArray(dummy_zeros_mask_np)
            else:
                try:
                    mask = dcm_patient_coords_to_mask.convert(current_rtstruct[0]['sequence'], dicom_image,
                                                              mask_background_value, mask_foreground_value)
                except ContourOutOfBoundsException:
                    warnings.warn(f"Structure {current_rtstruct[0]['name']} is out of bounds, ignoring contour!")
                    mask = sitk.GetImageFromArray(dummy_zeros_mask_np)
        else:
            if target_labels[label_idx] != 'background':
                warnings.warn(f'Mask for {target_labels[label_idx]} will be empty.')
            mask = sitk.GetImageFromArray(dummy_zeros_mask_np)

        mask.CopyInformation(dicom_image)
        class_masks[label_idx] = mask
    return class_masks


def get_item_paths(folder_path):
    """
    Create a dictionary containing all the possible dicom paths needed for a given patient folder

    Args:
        folder_path: current patient folder path

    Returns: dictionary containing the dicom paths of the patient

    """
    mr_paths = glob.glob(folder_path + 'MR*/')
    struct_paths = glob.glob(folder_path + 'RTSTRUCT*/')
    dose_paths = glob.glob(folder_path + 'RTDOSE*/')
    plan_paths = glob.glob(folder_path + 'RTPLAN*/')
    app_paths = glob.glob(folder_path + 'applicator*/')

    if len(mr_paths) > 4: warnings.warn(f"Redundant MRIs found: {folder_path.split('/')[-3]}")
    if len(struct_paths) != 1: warnings.warn(f"Zero or more than 1 RTSTRUCTs found: {folder_path.split('/')[-3]}")
    if len(dose_paths) != 1: warnings.warn(f"Zero or more than 1 RTDOSEs found: {folder_path.split('/')[-3]}")
    if len(plan_paths) != 1: warnings.warn(f"Zero or more than 1 RTPLANs found: {folder_path.split('/')[-3]}")
    # if len(app_paths) != 1: warnings.warn(f"Zero or more than 1 applicators found: {folder_path.split('/')[-3]}")

    t2_path, bffe_path = None, None
    for mr_path in mr_paths:
        mri_slice_path = glob.glob(mr_path + '*.dcm')[0]
        mri_slice = pydicom.dcmread(mri_slice_path)
        mri_description = mri_slice['SeriesDescription'].value
        # print(f'{mri_description=}')
        if mri_description in ['T2 TSE','T2TSE', 'T2TSE/', 'T2 TSE CLEAR',
                                'T2', 'T2TSE extra vulling',
                               'T2W_TSE_tra', 'T2 TRA', 'T2 CLEAR', 'T2W_TSE_ TRA']:
            t2_path = mr_path
        elif mri_description in ['3D BFFE', '3D bFFE TRA']:
            bffe_path = mr_path

    rtstruct_folder = struct_paths[0]
    rtstruct_path = glob.glob(rtstruct_folder + '*.dcm')
    assert len(rtstruct_path) == 1
    rtstruct_path = rtstruct_path[0]

    if t2_path:
        verify_img_rtstruct(t2_path, rtstruct_path)
    #
    # rtdose_folder = dose_paths[0]
    # rtdose_path = glob.glob(rtdose_folder + '*.dcm')
    # assert len(rtdose_path) == 1
    # rtdose_path = rtdose_path[0]
    #
    # rtplan_folder = plan_paths[0]
    # rtplan_path = glob.glob(rtplan_folder + '*.dcm')
    # # assert len(rtplan_path) == 1
    # rtplan_path = rtplan_path[0]

    # app_folder = app_paths[0]
    # app_path = glob.glob(app_folder + '*.dcm')
    # assert len(app_path) == 1
    # app_path = app_path[0]

    item_paths = {'t2': t2_path, 'bffe': bffe_path,
                  'struct': rtstruct_path,
                  # 'dose': rtdose_path, 'plan': rtplan_path#, 'app': app_path
                  }

    return item_paths


def verify_img_rtstruct(image_path, rtstruct_path):
    """
    Check if given image and rtstruct file match

    Args:
        image_path: path to folder with dicom slices
        rtstruct_path: path of dicom rtstruct

    Returns:

    """
    if image_path and rtstruct_path:
        slice_ids = []
        mri_slices_paths = glob.glob(image_path + '*')
        for slice_path in mri_slices_paths:
            mri_slice = pydicom.dcmread(slice_path)
            slice_ids.append(mri_slice['SOPInstanceUID'].value)
        rtstruct_file = glob.glob(rtstruct_path + '*')
        assert len(rtstruct_file) == 1, f"{rtstruct_file}"
        rtstruct = pydicom.dcmread(rtstruct_file[0])
        all_reference_ids = []
        rtstruct_sequences = rtstruct['ROIContourSequence'].value
        for item1 in rtstruct_sequences:
            contour_sequence = item1['ContourSequence'].value
            for item2 in contour_sequence:
                contour_item = item2['ContourImageSequence'].value
                assert len(contour_item) == 1
                contour_item = contour_item[0]['ReferencedSOPInstanceUID'].value
                all_reference_ids.append(contour_item)
        unique_ref_ids = list(np.unique(all_reference_ids))
        try:
            assert all([item in slice_ids for item in unique_ref_ids]), f"{sorted(slice_ids)} \n {sorted(unique_ref_ids)}"
            print("    MRI and RTSTRUCT match!")
        except AssertionError:
            warnings.warn(f"MRI {image_path} and RTSTRUCT {rtstruct_path} don't match!")


def convert_patient_images_from_dcm_to_nii(patient_index, patient_row_data, target_labels, img_mode,
                                           dataset_id, tr_image_path, tr_labels_path):
    """
    Extract images and corresponding masks and save them to nii files

    Args:
        patient_index: integer index of the patient
        patient_row_data: pandas series with patient data information
        target_labels: dictionary with class_index -> class_label as keys, values
        img_mode: we might have two images, the T2, and the BFFE, so the possible values for img_mode are "t2",
            "t2_bffe", "bffe", and "t2_bffe", where the double values indicate which image has the priority
        dataset_id: automatically configured dataset id
        tr_image_path: path to save the image
        tr_labels_path: path to save the mask

    Returns:

    """
    print(f"Processing data from patient {patient_row_data['patient id']}...")
    path_to_save_img_nifti_file = tr_image_path + dataset_id.lower() + f"_{patient_index:0=3d}.nii.gz"
    path_to_save_mask_nifti_file = tr_labels_path + dataset_id.lower() + f"_{patient_index:0=3d}.nii.gz"

    item_paths = get_item_paths(patient_row_data['folder path'])
    img1_path = item_paths['t2'] if img_mode in ['t2', 't2_bffe'] else item_paths['bffe']
    img2_path = item_paths['t2'] if img_mode == 'bffe_t2' else item_paths['bffe'] if img_mode == 't2_bffe' else None
    kwargs = {'img1_path': img1_path, 'img2_path': img2_path, 'target_labels': target_labels}
    # save img1_path
    pd.DataFrame([img1_path], columns=['path']).to_csv(path_to_save_img_nifti_file.replace('.nii.gz', '.csv'),
                                                            index=False)

    if '_' not in img_mode:
        print(f"    {img_mode.upper()} series path: {item_paths[img_mode]}")
    else:
        img_modalities = img_mode.split('_')
        for img_mod in img_modalities:
            print(f"    {img_mod.upper()} series path: {item_paths[img_mod]}")
    kwargs['struct_path'] = item_paths['struct']
    print(f"    RTSTRUCT file path: {item_paths['struct']}")

    print("    Extracting masks...")
    outputs = dcmrtstruct2nii(**kwargs)
    outputs = corrected_image_mask(outputs, img_mode, patient_row_data['slices to remove'])
    final_image = sitk.JoinSeries([outputs['image1'], outputs['image2']]) if '_' in img_mode else outputs['image1']

    # write image and mask to nii files
    print(f'    Writing image to {path_to_save_img_nifti_file}...')
    print(f'    Writing mask to {path_to_save_mask_nifti_file}...')
    sitk.WriteImage(final_image, path_to_save_img_nifti_file)
    sitk.WriteImage(outputs['mask'], path_to_save_mask_nifti_file)


def get_patient_data_from_csv(patient_csv, read_path):
    """
    Check if all ids in patient_csv exist in the read_path, and adjust the "folder path" based on the read_path

    Args:
        patient_csv: csv with at least 2 columns, "patient id", and "folder path"
        read_path: path where the data are stored

    Returns: patient_csv with the right "folder path"

    """
    patient_data = pd.read_csv(patient_csv)

    # Check if all ids in the csv exist in the read path
    all_paths = glob.glob(read_path + '*/')
    all_read_ids = [patient_path.split('/')[-2] for patient_path in all_paths]
    ids_excluded = [patient_id for patient_id in patient_data['patient id'].unique() if patient_id not in all_read_ids]
    assert ids_excluded == [], f"There are patients missing in the dataset: {ids_excluded}"

    for i, row in patient_data.iterrows():
        # correct folder paths with the current read folder
        # Me: BREAK BACKWARD COMPATIBILITY
        if False:
            main_path = read_path + "/".join(row['folder path'].split('\\')[-3:-1]) + '/'
        else:
            main_path = os.path.join(read_path, row['patient id'], row['folder path']) + '/'
        patient_data.loc[i, 'folder path'] = main_path
    return patient_data


def _create_dataset_niigz_from_dicoms(data_path, dataset_name, img_mode, patient_csv, processes, save_path, target_labels):
    tr_image_path = save_path + 'imagesTr/'
    ts_image_path = save_path + 'imagesTs/'
    tr_labels_path = save_path + 'labelsTr/'

    assert os.path.exists(save_path)
    os.mkdir(tr_image_path), os.mkdir(ts_image_path), os.mkdir(tr_labels_path)
    patient_data = get_patient_data_from_csv(patient_csv, data_path)
    partial_func = partial(convert_patient_images_from_dcm_to_nii, target_labels=target_labels, img_mode=img_mode,
                           dataset_id=dataset_name, tr_image_path=tr_image_path, tr_labels_path=tr_labels_path)
    pool_args = [{'func': partial_func, 'patient_index': index + 1, 'patient_row_data': patient_row}
                 for index, patient_row in patient_data.iterrows()]
    pool = Pool(processes)
    pool.map(fun_wrapper, pool_args)
    return tr_image_path, ts_image_path, patient_data


def fun_wrapper(dict_args):
    internal_dict = {key: value for key, value in dict_args.items() if key != 'func'}
    dict_args['func'](**internal_dict)
