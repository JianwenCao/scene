import os
import glob
import torch
import numpy as np
from PIL import Image
import open_clip

class GoatDatasetEnhanced:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.gt_root = os.path.join(root_dir, 'groundtruth')
        self.sensor_root = os.path.join(root_dir, 'dataset')
        self.samples = []
        self._scan_dataset()

    def _scan_dataset(self):
        if not os.path.exists(self.gt_root):
            print(f"Warning: Ground truth root {self.gt_root} not found.")
            return

        scenes = sorted(os.listdir(self.gt_root))
        for scene in scenes:
            scene_gt_path = os.path.join(self.gt_root, scene)
            scene_sensor_path = os.path.join(self.sensor_root, scene)
            
            for ep_id in range(6):
                ep_path = os.path.join(scene_gt_path, str(ep_id))
                if not os.path.exists(ep_path): continue
                
                for obj_name in sorted(os.listdir(ep_path)):
                    obj_path = os.path.join(ep_path, obj_name)
                    if not os.path.isdir(obj_path): continue
                    
                    # 1. Task Type
                    task_type_file = os.path.join(obj_path, 'task_type.txt')
                    if not os.path.exists(task_type_file): continue
                    with open(task_type_file, 'r') as f:
                        task_type = f.read().strip()
                    
                    # 2. Goals
                    goals = []
                    pos_file = os.path.join(obj_path, 'pos.txt')
                    if os.path.exists(pos_file):
                        with open(pos_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
                                    goals.append(torch.tensor([float(x) for x in clean_line.split()]))

                    # 3. Query
                    query_data = {}
                    language_file = os.path.join(obj_path, 'language.txt')
                    if os.path.exists(language_file):
                        with open(language_file, 'r') as f:
                            query_data['text'] = f.read().strip()
                    
                    if task_type == 'image':
                        img_files = glob.glob(os.path.join(obj_path, '*.png'))
                        if img_files:
                            query_data['image_paths'] = [img_files[0]]
                    elif task_type == 'object':
                        # Object query often has subdirectories 0, 1, 2 with multiple images
                        img_files = []
                        for sub in ['0', '1', '2']:
                            sub_path = os.path.join(obj_path, sub)
                            if os.path.isdir(sub_path):
                                img_files.extend(glob.glob(os.path.join(sub_path, '*.png')))
                        # Fallback to root images if any
                        img_files.extend(glob.glob(os.path.join(obj_path, '*.png')))
                        query_data['image_paths'] = sorted(list(set(img_files)))

                    self.samples.append({
                        'scene': scene,
                        'episode': ep_id,
                        'target_name': obj_name,
                        'task_type': task_type,
                        'goals': goals,
                        'query_data': query_data,
                        'sensor_path': scene_sensor_path
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class OpenFunGraphGoatQuerier:
    def __init__(self, model_name="ViT-H-14", pretrained="laion2b_s32b_b79k"):
        print(f"Loading CLIP model {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.templates = ["a photo of {}", "{}", "this is {}"]

    def encode_query(self, sample):
        task_type = sample['task_type']
        query_data = sample['query_data']
        
        if task_type == 'language':
            text = query_data['text']
            return self._encode_text(text)
        
        elif task_type == 'image':
            img_path = query_data['image_paths'][0]
            return self._encode_image(img_path)
            
        elif task_type == 'object':
            # If we have images, we can use them. If not, fallback to text.
            if query_data.get('image_paths'):
                # Average features of all images (up to 5 for efficiency)
                feats = []
                for img_path in query_data['image_paths'][:5]:
                    feats.append(self._encode_image(img_path))
                stacked = torch.stack(feats)
                mean_feat = stacked.mean(dim=0)
                return mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            else:
                return self._encode_text(query_data['text'])
        
        return None

    def _encode_text(self, text):
        with torch.no_grad():
            prompts = [t.format(text) for t in self.templates]
            tokens = self.tokenizer(prompts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
            mean_feat = feats.mean(dim=0)
            return mean_feat / mean_feat.norm(dim=-1, keepdim=True)

    def _encode_image(self, img_path):
        with torch.no_grad():
            img = Image.open(img_path).convert("RGB")
            img_input = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(img_input)
            return (feat / feat.norm(dim=-1, keepdim=True)).squeeze()

    def query_map(self, map_objects, query_feat):
        """
        map_objects: MapObjectList from OpenFunGraph
        query_feat: output of encode_query
        """
        if query_feat is None: return None
        
        # OpenFunGraph's MapObjectList already has compute_similarities
        similarities = map_objects.compute_similarities(query_feat.cpu().numpy())
        best_idx = torch.argmax(similarities).item()
        return map_objects[best_idx], similarities[best_idx].item()

if __name__ == "__main__":
    # Example usage
    dataset = GoatDatasetEnhanced("/root/autodl-tmp/Goat-core")
    print(f"Total samples: {len(dataset)}")
    
    # Filter by task type to verify
    for t in ['language', 'image', 'object']:
        count = sum(1 for s in dataset.samples if s['task_type'] == t)
        print(f"Task {t}: {count} samples")
