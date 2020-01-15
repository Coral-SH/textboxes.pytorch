import scipy.io as sio
import numpy as np
import xml.dom.minidom
import sys
import random
import os
import glob
import cv2 as cv
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import ipdb

SynthText_ROOT = '/home/Datasets/TextDet/SynthText/'
ANNOTATIONS = os.path.join(SynthText_ROOT, 'annotations')

class SynthTextDataset(data.Dataset):
    """SynthText Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to SynthText dataset folder.
        transform (callable, optional): transformation to perform on the
            input image
        dataset_name (string, optional): which dataset to load
            (default: 'SynthText')
    """
    def __init__(self, root, image_sets='train', transform=None, dataset_name='SynthText'):
        self.root = root
        self.name = dataset_name
        self.transform = transform
        self.image_sets = image_sets
        file_path = os.path.join(root, self.image_sets+'.txt')
        self.annotations = list()
        self.images = list()
        with open(file_path, 'r') as fp:
            files = fp.read().splitlines()
        for file in files:
            file = file.split(' ')
            self.images.append(file[0])
            self.annotations.append(file[1])
        print('Number of images in SynthText_{}: {:d}'.format(self.image_sets, len(self.images)))
        
    def __getitem__(self, index):
        img = cv.imread(self.images[index])
        height, width = img.shape[:2]
        data = ET.parse(self.annotations[index]).getroot()
        target = []
        for obj in data.iter('object'):
            content = obj.find('content').text
            if not content:
                #content = '###'
                continue
            else:
                #content = content.lower().strip()
                content = 0
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            bndbox.append(content) # content label 0 means it is text
            target += [bndbox]
            
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # BGR to RGB
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # HWC to CHW
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
    
    def __len__(self):
        return len(self.images)

    
def MatRead(matfile):
    data = sio.loadmat(matfile)
    train_file = open(os.path.join(ROOT, 'train.txt'), 'w')
    test_file = open(os.path.join(ROOT, 'test.txt'), 'w')
    
    for i in range(len(data['txt'][0])):
        contents = []
        for val in data['txt'][0][i]:
            v = [x.split("\n") for x in val.strip().split(" ")]
            contents.extend(sum(v, []))
        print("No.{} data".format(i))
        rec = np.array(data['wordBB'][0][i], dtype=np.int32)
        if len(rec.shape) == 3:
            rec = rec.transpose(2,1,0)
        else:
            rec = rec.transpose(1,0)[np.newaxis, :]

        doc = xml.dom.minidom.Document() 
        root = doc.createElement('annotation') 
        doc.appendChild(root) 
        print("start to process {} object".format(len(rec)))
        
        for j in range(len(rec)):
            nodeobject = doc.createElement('object')
            nodecontent = doc.createElement('content')
            nodecontent.appendChild(doc.createTextNode(str(contents[j])))

            nodename = doc.createElement('name')
            nodename.appendChild(doc.createTextNode('text'))

            bndbox = {}
            bndbox['x1'] = rec[j][0][0]
            bndbox['y1'] = rec[j][0][1]
            bndbox['x2'] = rec[j][1][0]
            bndbox['y2'] = rec[j][1][1]
            bndbox['x3'] = rec[j][2][0]
            bndbox['y3'] = rec[j][2][1]
            bndbox['x4'] = rec[j][3][0]
            bndbox['y4'] = rec[j][3][1]
            bndbox['xmin'] = min(bndbox['x1'], bndbox['x2'], bndbox['x3'], bndbox['x4'])
            bndbox['xmax'] = max(bndbox['x1'], bndbox['x2'], bndbox['x3'], bndbox['x4'])
            bndbox['ymin'] = min(bndbox['y1'], bndbox['y2'], bndbox['y3'], bndbox['y4'])
            bndbox['ymax'] = max(bndbox['y1'], bndbox['y2'], bndbox['y3'], bndbox['y4'])

            nodebndbox = doc.createElement('bndbox')
            for k in bndbox.keys():
                nodecoord =  doc.createElement(k)
                nodecoord.appendChild(doc.createTextNode(str(bndbox[k])))
                nodebndbox.appendChild(nodecoord)

            nodeobject.appendChild(nodecontent)
            nodeobject.appendChild(nodename)
            nodeobject.appendChild(nodebndbox)
            root.appendChild(nodeobject)

        filename = os.path.join(ANNOTATIONS, data['imnames'][0][i][0].replace('.jpg', '.xml'))
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fp = open(filename, 'w')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()
        rad = random.uniform(10,20)
        img_path = os.path.join(ROOT, data['imnames'][0][i][0])
        xml_path = os.path.join(ROOT, filename)
        file_line = img_path + " " + xml_path + '\n'
        if rad > 18:
            test_file.write(file_line)
        else:
            train_file.write(file_line)    

    train_file.close()
    test_file.close()

if __name__ == '__main__':
    # preprocess the SynthText dataset to get train/test subset
    # and save the annotation of each image
    MatRead(os.path.join(SynText_ROOT, 'gt.mat'))