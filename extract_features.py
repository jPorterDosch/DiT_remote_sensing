import torch
from tqdm import tqdm
import os
from src.flux.feat_flux import Featurizer4Eval
from src.flux.eurosat_dataloader import get_eurosat_dataloader, get_eurosat_categories, get_eurosat_class_to_prompt

def extract_and_save_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing FLUX Featurizer...")
    categories = get_eurosat_categories()
    featurizer = Featurizer4Eval(flux_id='flux-dev', null_prompt='', cat_list=categories)
    class_to_prompt = get_eurosat_class_to_prompt()
    DATA_ROOT = "/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/EuroSAT"

    # NEW: Loop through both splits!
    splits = ['train', 'test']
    
    for current_split in splits:
        print(f"\n========== Processing {current_split.upper()} Split ==========")
        
        dataloader = get_eurosat_dataloader(
            root=DATA_ROOT, 
            split=current_split,  # Dynamically use 'train' then 'test'
            batch_size=1,
            shuffle=False 
        )

        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting {current_split}"):
                image = batch["img"].squeeze(0).to(device)
                label = batch["label"]
                class_name = batch["class_name"][0] 
                
                #category = class_to_prompt[class_name] 
                caption = ""
                
                features, _ = featurizer.forward(
                    args=None,              
                    img_tensor=image, 
                    caption=caption,
                    category="", 
                    block_idx=[1]
                )
                
                features = features.mean(dim=[2, 3]) 
                features = features.to(torch.float32).cpu()
                
                all_features.append(features)
                all_labels.append(label.cpu())

        print(f"Stacking tensors for {current_split}...")
        all_features = torch.cat(all_features, dim=0)
        
        if isinstance(all_labels[0], torch.Tensor) and all_labels[0].dim() > 0:
            all_labels = torch.cat(all_labels, dim=0) 
        else:
            all_labels = torch.tensor(all_labels)
        
        # Save dynamically named files based on the split
        save_name = f'eurosat_flux_features_{current_split}.pt'
        save_dict = {
            'features': all_features,
            'labels': all_labels
        }
        torch.save(save_dict, save_name)
        print(f"Successfully saved {current_split} features to '{save_name}'")

if __name__ == "__main__":
    extract_and_save_features()