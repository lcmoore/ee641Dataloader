from ast import Tuple
import torch
import torchio as tio
import os
import torch.utils.data as data

def get_subject_list(dataset_dir):
    '''
    dataset_dir must point to "Infection Segmentation Data Transformed/Train || /Test || /Val
    '''
    patient_list = []
    category_switch = { ## lookup dictionary for mapping the folder name to the file prefix
        "COVID-19":"covid",
        "Non-COVID" : "non_COVID",
        "Normal": "Normal"
                        


    }


    list_categories = os.listdir(dataset_dir) #dataset_dir = Train/Test/Val for a given dataset

    for category in list_categories: # categories = COVID-19 || Non-COVID-19 || Normal
        print(category)
        files_dir = os.path.join(dataset_dir,category) # files dir = images & infection masks & lung masks (maybe) 

        for patient_file in os.listdir(os.path.join(files_dir, "images")): ## images is the input folder, patient file is the individual image
            images_dir = os.path.join(files_dir, "images")
            full_dir = os.path.join(images_dir,patient_file) ## full dir is .../.../.../covid_1.nii.gz
            lung_dir = os.path.join(files_dir, "lung masks")
            infection_dir = os.path.join(files_dir, "infection masks")
            if "sub" in patient_file:
                patient_number = patient_file.strip(".nii.gz")
            elif category == "COVID-19":
                patient_number = patient_file.split("_")[1].strip(".nii.gz")
            else:
                patient_number = patient_file.split("(")[1].strip(").nii.gz")
            


            subject_dict = {} # this is a dictionary for an individual patient
            subject_dict["patient_id"] = patient_number ## this stores the patient number in the dictionary so that it can be refrenced later in the Subject instance
            subject_dict["category"] = category # stores the classifcation type 
            subject_dict["image"]=tio.ScalarImage(full_dir) # this creates a TorchIO scalar image from the patient image file
            
            # the Label Map class handles binary mask images, this is needed for the automatic seleciton of post-transformation
            # interpolation. Label Maps use a nearest neighbor interpolation, Scalar images by default use a b-spline
            if "sub" in patient_number:
                subject_dict["lung mask"] = tio.LabelMap(os.path.join(lung_dir,patient_file))    
            elif category == "COVID-19":
                subject_dict["lung mask"] = tio.LabelMap(os.path.join(lung_dir,category_switch[category] + "_"+str(patient_number) + ".nii.gz")) 
            else:
                subject_dict["lung mask"] = tio.LabelMap(os.path.join(lung_dir,category_switch[category] + " ("+str(patient_number) + ").nii.gz")) 
            
            if os.path.exists(infection_dir): # Lung segment dataset does not have infection masks
                if "sub" in patient_number:
                   subject_dict['infection mask'] = tio.LabelMap(os.path.join(infection_dir,category_switch[category] + "_"+str(patient_number) + ".nii.gz"))  
                elif category == "COVID-19":
                       
                    subject_dict['infection mask'] = tio.LabelMap(os.path.join(infection_dir,category_switch[category] + "_"+str(patient_number) + ".nii.gz"))
 
                else:
                    subject_dict['infection mask'] = tio.LabelMap(os.path.join(infection_dir,category_switch[category] + " ("+str(patient_number) + ").nii.gz"))
 
            subject = tio.Subject(subject_dict) # creates the "Subject" instance from the dictinoary. This is the wrapper for all of
            # the stored images and metadata (patient number, category, whatever else you want)
            patient_list.append(subject)
    return patient_list

def get_loader(target_directory:str =None, train_bs=1, val_bs=1, num_works=10) -> Tuple[data.DataLoader,data.DataLoader,data.DataLoader]: 
    '''
    Parameters
    target_directory: absolute path to location of either Infection Segmentation Dataset or Lung Segmentation Dataset
    train_bs: batch size for training dataloader, default 1
    val_bs: batch size for validation dataloader, default 1
    num_works: attempts to allocate addtional cpu workers for dataloading (unconfirmed if functional)

    Returns
    train_dataloader, validation dataloader, test dataloader

    '''

    if target_directory is not None: ## if the target directory is provided
        data_dir_train = os.path.join(target_directory,"Train")
        data_dir_val = os.path.join(target_directory,"Val")
        data_dir_test = os.path.join(target_directory, "Test")
    else: 
        data_dir_train = "./Train"
        data_dir_val = './Validation'   
        data_dir_test = "./Test"

    train_subjects = get_subject_list(data_dir_train)
    val_subjects = get_subject_list(data_dir_val)
    test_subjects = get_subject_list(data_dir_test)


    transforms = tio.Compose([ # composition of the torchIO transforms, applies each transform with probability p to images each time they are loaded

    tio.RandomFlip(p=0.5,axes=0), # axis=0 is horizontal flip
    tio.RandomAffine(p=1.0,scales=0.,degrees=(0,0,40),translation=(50,0,0), isotropic=True, default_pad_value=0) # rotates the image +/- 40 degrees and translates in a random direction between 0-50 pixels
 
      ])  

    train_dataset = tio.SubjectsDataset(train_subjects, transform=transforms)

    val_dataset = tio.SubjectsDataset(val_subjects)
    test_dataset = tio.SubjectsDataset(test_subjects)


    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works)
                                   #pin_memory=False,collate_fn=tio.utils.history_collate)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works)
                                 #pin_memory=False,collate_fn=tio.utils.history_collate)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_works)

    return train_loader, val_loader, test_loader 

