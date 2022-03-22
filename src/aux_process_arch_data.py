"""
Methods and constants to convert
"""

import os, json, csv
import pandas as pd
import numpy as np
import glob
import argparse
import pdb
from CONFIG import CONFIG


COMBINE_CLASS = {
    'pomegranate' : ['pom', 'pomegranate'], 'winged sandal' : ['winged sandals', 'winged sandal'], 'columns' : ['columns', 'column']
}

ABSTRACT_CLASS_classarch = {
    'elongated_objects' : ['sword', 'harpe', 'scepter', 'arrow', 'bow', 'trident', 'club', 'kerykeion', 'torch', 'spear', 'stick', 'thyrsos', 'axe', 'thymiaterion', 'aulos'],# included 'bow',
    'architecture' : ['columns', 'altar', 'door', 'bed'],
    'garment_accessory' : ['phrygian cap', 'lions skin (headdress)', 'winged sandal', 'wreath (worn)', 'stephane (bride)', 'taenia', 'phrygian cap', 'petasos'], # I included winged sandals in this category and renamed it to clothing, does it make sense to you?
    'musicalinstruments' : ['lyre', 'harp'],
    'roundobjects' : ['shield', 'hoop', 'wreath'],
    'containers' : ['box', 'basket', 'cornucopia', 'quiver'], # I added 'quiver' here since (although is a part of a weapon), it is a container for arrows
    'vessels' : ['kantharos', 'vessel (kantharos)', 'vessel (oinochoe)', 'vessel (amphora)', 'phiale', 'vessel (loutrophoros)', 'vessel'],
    'creatures' : ['tauros', 'lion', 'dolphin', 'cock', 'fish', 'sphinx', 'owl', 'dog', 'hippocamp', 'ram', 'octopus', 'centaur', 'pegasus', 'griffin', 'panther', 'chimaira'], # the list includes animals from land, sea and fiction, so I named the category 'organisms', however it can be changed to what you feel best
    'character' : ['Eros'],
    'others' : ['lions skin', 'tripod', 'thunderbolt', 'pomegranate', 'palmette', 'ship', 'hand-held fan']
}

ABSTRACT_CLASS_arthist = {
    'characters' : ['gabriel', 'mary', 'angel', 'god', 'putto', 'jesus child'],
    'architecture' : ['column', 'architecture', 'window'],
    'furniture' : ['bed', 'bookrest', 'stool', 'chair', 'door'],
    'animals_birds' : ['dove', 'cat', 'peacock'],
    # 'objects' : ['basket', 'book', 'vase'] -- from vision's perspective these are very different classes, hence combining the more pattern matching classes below
    'containers' : ['basket', 'vase', 'flower vase'],
    'scene' : ['annunciation'],
    'scrolls' : ['speech scroll', 'banderole', 'scroll'], # because the objects have similar shapes
    'others' : ['flower', 'scepter', 'book', 'branch']
}

ABSTRACT_CLASS_chrisarch = {}

CORRUPTED_PHOTOS = ['kantharos_1877.jpg', 'herakles_2017.jpg']

def combine_classes(_class):
    for i in COMBINE_CLASS:
        if _class in COMBINE_CLASS[i]:
            return i
    return _class

def abstract_classes_classarch(_class):
    for i in ABSTRACT_CLASS_classarch:
        if _class in ABSTRACT_CLASS_classarch[i]:
            return i
    return _class

def abstract_classes_arthist(_class):
    for i in ABSTRACT_CLASS_arthist:
        if _class in ABSTRACT_CLASS_arthist[i]:
            return i
    return _class

def abstract_classes_chrisarch(_class):
    for i in ABSTRACT_CLASS_chrisarch:
        if _class in ABSTRACT_CLASS_chrisarch[i]:
            return i
    return _class

def parse_json_to_csv_bounding_boxes(df, folder_path):
    csv_list = []
    flags = 0
    for index in df.index:
        if index not in ['ui', 'core', 'project', 'region', 'file']:
            index_dict = df['_via_img_metadata'][index]
            index_json = json.dumps(df['_via_img_metadata'][index])
            label = json.loads(index_json)

            filename = label['filename']
            if filename in CORRUPTED_PHOTOS:
                continue

            regions = label['regions']
            if len(regions) != 0:
                for annotation in range(len(regions)):
                    region_annotation = regions[annotation]
#                     print(region_annotation)
                    region_annotation_shape = region_annotation['shape_attributes']
                    region_annotation_name = region_annotation['shape_attributes']['name']
#                     print(region_annotation)

                    if 'object/attribute/figure' in region_annotation['region_attributes']:
                        region_annotation_attribute = region_annotation['region_attributes']['object/attribute/figure']
                    elif 'objects/attribute/figures' in region_annotation['region_attributes']:
                        region_annotation_attribute = region_annotation['region_attributes']['objects/attribute/figures']
                    elif 'objects/attributes/figures' in region_annotation['region_attributes']:
                        region_annotation_attribute = region_annotation['region_attributes']['objects/attributes/figures']
                    elif 'schema' in region_annotation['region_attributes']:
                        region_annotation_attribute = region_annotation['region_attributes']['schema']
                    else:
                        print('- issue with the file: {}'.format(filename))
                        print('- the region attribute -- {} -- is not consistent. Change "\
                              the code'.format([key for key in label['regions'][0]['region_attributes'].keys()]))
                        flags = flags + 1
                        continue

                    if  region_annotation_name == 'rect':
                        new_filename = os.path.join(folder_path, filename)
                        width = region_annotation_shape['width']
                        height = region_annotation_shape['height']
                        xmin = region_annotation_shape['x']
                        ymin = region_annotation_shape['y']
                        xmax = xmin + width
                        ymax = ymin + height
                        class_ = region_annotation_attribute


                    elif region_annotation_name == 'polygon' or region_annotation_name == 'polyline':
                        new_filename = os.path.join(folder_path, filename)
                        xmin = np.min(region_annotation_shape['all_points_x'])
                        xmax = np.max(region_annotation_shape['all_points_x'])
                        ymin = np.min(region_annotation_shape['all_points_y'])
                        ymax = np.max(region_annotation_shape['all_points_y'])
                        width = int(xmax - xmin)
                        height = int(ymax - ymin)
                        class_ = region_annotation_attribute

                    elif region_annotation_name == 'circle':
                        new_filename = os.path.join(folder_path, filename)
                        xmin = region_annotation_shape['cx'] - region_annotation_shape['r']
                        xmax = region_annotation_shape['cx'] + region_annotation_shape['r']
                        ymin = region_annotation_shape['cy'] - region_annotation_shape['r']
                        ymax = region_annotation_shape['cy'] + region_annotation_shape['r']
                        width = height = region_annotation_shape['r']*2
                        class_ = region_annotation_attribute

                    elif region_annotation_name == 'ellipse':
                        new_filename = os.path.join(folder_path, filename)
                        xmin = region_annotation_shape['cx'] - region_annotation_shape['rx']
                        xmax = region_annotation_shape['cx'] + region_annotation_shape['rx']
                        ymin = region_annotation_shape['cy'] - region_annotation_shape['ry']
                        ymax = region_annotation_shape['cy'] + region_annotation_shape['ry']
                        width = region_annotation_shape['rx']*2
                        height = region_annotation_shape['ry']*2
                        class_ = region_annotation_attribute

                    else: # for cases with 'circle' or any other as annotation name
                        print(filename, '- The region is neither rectangle nor polygon!')
                        continue

                    if args.abstract_classes == 'yes':
                        if args.domain == 'classarch':
                            class_ = combine_classes(class_)
                            class_ = abstract_classes_classarch(class_)
                        elif args.domain == 'arthist':
                            class_ = abstract_classes_arthist(class_)
                        elif args.domain == 'chrisarch':
                            class_ = abstract_classes_chrisarch(class_)

                    value = (new_filename, xmin, ymin, xmax, ymax, class_)
                    csv_list.append(value)
        else:
            continue

    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    print('(Missed): {}'.format(flags))
    print('-'*65)

    return csv_df


def get_bounding_box_csv(labeled_data_path, number_classes):
    list_of_folders = glob.glob(labeled_data_path + '/*')
    # json_paths = glob.glob(labeled_data_path + '/*/*mapped_annotations.json')
    json_paths = glob.glob(labeled_data_path + '/*/*.json')
    json_paths = [i for i in json_paths if '_coco' not in i]

    all_df = []

    for json_path, folder_path in zip(json_paths, list_of_folders):
        print(f"(PATH/FILE): {json_path}...\n")
        folder_json_df = pd.read_json(json_path, orient='records')
        folder_csv_df = parse_json_to_csv_bounding_boxes(folder_json_df, folder_path)
        all_df.append(folder_csv_df)
    all_data_df = pd.concat(all_df)
    all_data_df.index = range(0, len(all_data_df))
    all_classes = list(all_data_df['class'].value_counts().keys())
    all_classes_dict = {all_classes[i] : i for i in range(0, len(all_classes))}

    custom_classes = list(all_data_df['class'].value_counts()[:number_classes].keys())

    custom_classes_dict = { custom_classes[i] : i for i in range(0, len(custom_classes))}
    custom_data_df = all_data_df[all_data_df['class'].isin(custom_classes)]

    custom_faster_rcnn_df = pd.DataFrame()
    custom_faster_rcnn_df['filepath'] = custom_data_df['filename']
    custom_faster_rcnn_df['x1'] = custom_data_df['xmin']
    custom_faster_rcnn_df['y1'] = custom_data_df['ymin']
    custom_faster_rcnn_df['x2'] = custom_data_df['xmax']
    custom_faster_rcnn_df['y2'] = custom_data_df['ymax']
    custom_faster_rcnn_df['class_name'] = custom_data_df['class']

    all_faster_rcnn_df = pd.DataFrame()
    all_faster_rcnn_df['filepath'] = all_data_df['filename']
    all_faster_rcnn_df['x1'] = all_data_df['xmin']
    all_faster_rcnn_df['y1'] = all_data_df['ymin']
    all_faster_rcnn_df['x2'] = all_data_df['xmax']
    all_faster_rcnn_df['y2'] = all_data_df['ymax']
    all_faster_rcnn_df['class_name'] = all_data_df['class']

    return custom_data_df, custom_classes_dict, custom_faster_rcnn_df, all_faster_rcnn_df, all_data_df, all_classes_dict


def process_csv_to_json_and_save(data, savepath):
    """
    Converting a csv with the annotations: (img_path, x1, y1, x2, y2, class) to a json
    to later fit the person detector model

    Args:
    -----
    data: pandas DataFrame
        dataframe containing the data to conver to json
    savepath: string
        path to the directory and file to save the json data
    """


    annotations = []
    img_ids = {}
    imgs = []
    id_idx = 0
    class_idx = 1
    classes = []
    class_names = []
    for id, (index, row) in enumerate(data.iterrows()):

        # obtainign image name and image id
        img_name = os.path.basename(row["filename"])
        filename = row['filename'].split("class_arch_data/")[-1]

        # if("bf0116.jpg" in img_name):
            # print(row['filename'])

        if(img_name not in img_ids.keys()):
            img_ids[img_name] = id_idx
            img_dict = {
                "id": id_idx,
                "file_name": filename
            }
            imgs.append(img_dict)
            id_idx = id_idx + 1

        class_name = row["class"]
        if(class_name not in class_names):
            cur_cat_id = class_idx
        else:
            cur_cat_id = class_names.index(class_name) + 1

        cur_ann = {
            "id": len(annotations),
            "image_id": img_ids[img_name],
            "img_name": img_name,
            "filename": filename,
            "bbox": f"{row['xmin']},{row['ymin']},{row['xmax']},{row['ymax']}",
            "category_id": cur_cat_id,
            "iscrowd": 0,
            "area": (row['ymax']-row['ymin']) * (row['xmax']-row['xmin'])
        }
        annotations.append(cur_ann)
        if(class_name not in class_names):
            class_ = {
                "id": class_idx,
                "name": class_name
            }
            classes.append(class_)
            class_names.append(class_name)
            class_idx = class_idx + 1

    dict_data = {}
    dict_data["annotations"] = annotations
    dict_data["images"] = imgs
    dict_data["categories"] = classes

    with open(savepath, "w") as file:
        json.dump(dict_data, file)

    return


if __name__ == '__main__':

    os.system("clear")
    parser = argparse.ArgumentParser(description='Creating dataset arg parser')
    # parser.add_argument('--images_path', type=str, help='Path for images containing sub-directories')
    # parser.add_argument('--data_csv', type=str, help='Folder where we store the generated data_csv files')
    parser.add_argument('--number_classes', type=int, help='Number of training classes', default=10)
    parser.add_argument('--domain', type=str, help='The type of the data: arthist/chrisarch/classarch', default=True)
    parser.add_argument('--abstract_classes', type=str, help='yes/no to decide whether abstract classes are requried =', default='yes')

    args = parser.parse_args()
    # resources_folder = args.data_csv
    # labeled_data_path = args.images_path
    labeled_data_path = os.path.join(CONFIG["paths"]["data_path"], "class_arch_data")
    resources_folder = os.path.join(CONFIG["paths"]["data_path"], "annotations_arch_data")

    if(os.path.exists(resources_folder) and len(os.listdir(resources_folder)) > 0):
        print(f"Arch-data annotations directory is not empty. Previous files might  " +
                "be overwritten during the proccess...")
        txt = input("Do you want to proceed? (y/n)\n")
        if(txt != "y" and txt != "Y"):
            print("Aborting execution...")
            exit()

    # Generating the csv_file, class_dict
    custom_data_df, custom_class_dict, custom_faster_rcnn_df, all_faster_rcnn_df,\
        all_data_df, all_classes_dict = get_bounding_box_csv(labeled_data_path,
                                                             args.number_classes)

    print('Total annotations per class ({} classes)'.format(len(all_data_df['class'].unique())))
    print(all_data_df['class'].value_counts())
    print('Total annotations per image/file')
    print(all_data_df['filename'].value_counts())
    print('Total Images:')
    print(len(all_data_df['filename'].unique()))
    print('Total annotations: ')
    print(len(all_data_df))
    print('Mean annotations per image', all_data_df['filename'].value_counts().mean())

    ## Creating the resources folder if it doesn't exist
    if not os.path.exists(resources_folder):
        os.makedirs(resources_folder)

    ## Dumping the csv_file, custom_class_dict
    custom_data_csv_path = os.path.join(resources_folder, 'custom_data.csv')
    custom_data_json_path = os.path.join(resources_folder, 'custom_data.json')
    all_data_csv_path = os.path.join(resources_folder, 'all_data.csv')
    all_data_json_path = os.path.join(resources_folder, 'all_data.json')
    custom_faster_rcnn_csv_path = os.path.join(resources_folder, 'custom_faster_rcnn_data.csv')
    custom_faster_rcnn_json_path = os.path.join(resources_folder, 'custom_faster_rcnn_data.json')
    all_faster_rcnn_csv_path = os.path.join(resources_folder, 'all_faster_rcnn_data.csv')
    all_faster_rcnn_json_path = os.path.join(resources_folder, 'all_faster_rcnn_data.json')

    custom_class_dict_path = os.path.join(resources_folder, 'custom_class_dict.json')
    all_classes_dict_path = os.path.join(resources_folder, 'all_classes_dict.json')

    process_csv_to_json_and_save(data=custom_data_df, savepath=custom_data_json_path)
    custom_data_df.to_csv(custom_data_csv_path, index=None, header=None)
    process_csv_to_json_and_save(data=all_data_df, savepath=all_data_json_path)
    all_data_df.to_csv(all_data_csv_path, index=None, header=None)
    # process_csv_to_json_and_save(data=custom_faster_rcnn_df, savepath=custom_faster_rcnn_json_path)
    custom_faster_rcnn_df.to_csv(custom_faster_rcnn_csv_path, index=None, header=None)
    # process_csv_to_json_and_save(data=all_faster_rcnn_df, savepath=all_faster_rcnn_json_path)
    all_faster_rcnn_df.to_csv(all_faster_rcnn_csv_path, index=None, header=None)


    with open(os.path.join(resources_folder, 'custom_class_dict.json'), 'w') as out:
        json.dump(custom_class_dict, out)
    with open(os.path.join(resources_folder, 'custom_class_dict.csv'), 'w') as f:
        for key in custom_class_dict.keys():
            f.write("%s, %s\n"%(key, custom_class_dict[key]))


    with open(os.path.join(resources_folder, 'all_classes_dict.json'), 'w') as out:
        json.dump(all_classes_dict, out)

    with open(os.path.join(resources_folder, 'all_classes_dict.csv'), 'w') as f:
        for key in all_classes_dict.keys():
            f.write("%s,%s\n"%(key, all_classes_dict[key]))


#
