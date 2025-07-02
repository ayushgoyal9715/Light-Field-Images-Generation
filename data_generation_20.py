import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from PIL import Image
import os
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.image import ssim



def gen_ran_cord(cntrl, n=51, m=51):
    ctrl=30
    num = (n * m) * (cntrl) // 100 
    list_of_coordinates = []
    
    while num:
        x = random.randint(1, 49)  
        y = random.randint(1, 49)  
        list_of_coordinates.append((x, y))
        num -= 1
        break;
    
    return list_of_coordinates

def find_sol_X(n, m, ctrl,fault_cameras=None,is_true=True):
    if is_true:
        fault_cameras = gen_ran_cord(ctrl, n, m)
   
    num = len(fault_cameras)
    replace_dict = {}  
    # print("done")
    
    for i in range(num):
        x = fault_cameras[i][0]
        y = fault_cameras[i][1]
        flag = False
        j = 5
        
        while not flag and ((x + j) < n and (x - j) >= 0) and ((y + j) < m and (y - j) >= 0):
            right_up = (x-j, y + j)
            left_up = (x-j, y - j)
            right_down = (x +j, y+j)
            left_down = (x + j, y-j)
            
            if right_up not in fault_cameras and left_up not in fault_cameras and right_down not in fault_cameras and left_down not in fault_cameras:
                replace_dict[(x, y)] = (right_up, left_up, right_down, left_down)
                flag = True
            else:
                j += 1
        
        if not flag:
            replace_dict[(x, y)] = "No replacable cameras found"
    
    return replace_dict


def find_sol_plus(n, m, ctrl,fault_cameras=None,is_true=True):
    if is_true:
        fault_cameras = gen_ran_cord(ctrl, n, m)
    
    num = len(fault_cameras)
    replace_dict = {}  
    # print("done")
    
    for i in range(num):
        x = fault_cameras[i][0]
        y = fault_cameras[i][1]
        flag = False
        j = 0
        
        while not flag and ((x + j) < n and (x - j) >= 0) and ((y + j) < m and (y - j) >= 0):
            right = (x, y + j)
            left = (x, y - j)
            up = (x - j, y)
            down = (x + j, y)
            
            if right not in fault_cameras and left not in fault_cameras and up not in fault_cameras and down not in fault_cameras:
                replace_dict[(x, y)] = (right, left, up, down)
                flag = True
            else:
                j += 1
        
        if not flag:
            replace_dict[(x, y)] = "No replacable cameras found"
    
    return replace_dict
# images_dict=find_sol_plus(51,51,5)
# images_dict


def regeneration(image_dict):
    defective=False
    fault=[]
    for i in image_dict:
        if image_dict[i]=='No replacable cameras found':
            defective=True
            fault.append(i)
    f=find_sol_plus(51,51,5,fault,False)
    image_dict.update(f)
    #2nd time
    defective=False
    fault=[]
    for i in image_dict:
        if image_dict[i]=='No replacable cameras found':
            defective=True
            fault.append(i)
    f=find_sol_X(51,51,5,fault,False)
    image_dict.update(f)
    #3rd time
    defective=False
    fault=[]
    for i in image_dict:
        if image_dict[i]=='No replacable cameras found':
            defective=True
            fault.append(i)
    f=find_sol_plus(51,51,5,fault,False)
    image_dict.update(f)
    #4th time
    defective=False
    fault=[]
    for i in image_dict:
        if image_dict[i]=='No replacable cameras found':
            defective=True
            fault.append(i)
    f=find_sol_X(51,51,5,fault,False)
    image_dict.update(f)
    #5th time
    defective=False
    fault=[]
    for i in image_dict:
        if image_dict[i]=='No replacable cameras found':
            defective=True
            fault.append(i)
    f=find_sol_X(51,51,5,fault,False)
    image_dict.update(f) 
    
    count=0;  
    
    for i in image_dict:
        if image_dict[i]=='No replacable cameras found':
            count+=1
    return image_dict,count




root_path = "/home/ag/lfi/dataset/lofimages"

data_folders = sorted([os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])


def load_and_resize_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    print(image.size())
#    image = image.resize(target_size)
    return np.array(image)


dataset = []



class LightFieldDataGenerator(Sequence):
    def __init__(self, data_folders, batch_size=8, target_size=(256, 256), shuffle=True):
        self.data_folders = data_folders
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.data_list = self._generate_data_list()
        self.on_epoch_end()

    def _generate_data_list(self):
        data_list = []
        for data_path in self.data_folders:
            total_images = []
            for _ in range(20):  # Generate 10 iterations per folder
                images_dict, _ = regeneration(find_sol_plus(51, 51, 50))
                total_images.append(images_dict)

            for item in total_images:
                for gt_coords, input_coords in item.items():
                    gt_file = f"x_{gt_coords[0]}_y_{gt_coords[1]}.jpg"
                    gt_path = os.path.join(data_path, gt_file)
                    input_paths = []
                    for coord in input_coords:
                        input_file = f"x_{coord[0]}_y_{coord[1]}.jpg"
                        input_path = os.path.join(data_path, input_file)
                        if os.path.exists(input_path):
                            input_paths.append(input_path)
                    if os.path.exists(gt_path) and input_paths:
                        data_list.append((gt_path, input_paths))

        return data_list

    def load_and_resize_image(self, image_path):
        image = Image.open(image_path).resize(self.target_size)
        return np.array(image)/255.0

    def __len__(self):
        """Number of batches per epoch."""
        return len(self.data_list) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_data = self.data_list[index * self.batch_size : (index + 1) * self.batch_size]
        return self.__data_generation(batch_data)

    def __data_generation(self, batch_data):
        batch_ground_truths = []
        batch_input_stacks = []
        for gt_path, input_paths in batch_data:
            ground_truth_image = self.load_and_resize_image(gt_path)
            ground_images=[]
            ground_images.append(ground_truth_image)
            input_images = []
            for p in input_paths:
                i=self.load_and_resize_image(p)
                ground_images.append(i)
                input_images.append(i)
            if not input_images:
                continue
            ground_stack=np.stack(ground_images,axis=-2)
            input_stack = np.stack(input_images,axis=-2)
            batch_ground_truths.append(ground_stack)
            batch_input_stacks.append(input_stack)
        batch_ground_truths = np.array(batch_ground_truths)
        batch_input_stacks = np.array(batch_input_stacks)
        return batch_input_stacks, batch_ground_truths
