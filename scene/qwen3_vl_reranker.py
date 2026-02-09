import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging
import re

from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.utils import TransformersKwargs
from qwen_vl_utils.vision_process import process_vision_info

logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS

def sample_frames(frames: List[Union[str, Image.Image]], num_segments: int, max_segments: int) -> List[str]:
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            sampled_frames.append(frames[frame_idx])
        except:
            break
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]

class Qwen3VLReranker():
    def __init__(
        self, 
        model_name_or_path: str, 
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path.lower()
        self.is_thinking_model = "thinking" in self.model_name_or_path
            
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames

        print(f"  [Init] Loading {model_name_or_path} (Thinking: {self.is_thinking_model}) with kwargs: {kwargs}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        )
        if "device_map" not in kwargs:
            self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='left'
        )
        self.model.eval()
        
        # Token IDs for Reranker-style scoring (Yes/No)
        if not self.is_thinking_model:
            self.yes_tokens = [
                self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0],
                self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
            ]
            self.no_tokens = [
                self.processor.tokenizer.encode("No", add_special_tokens=False)[0],
                self.processor.tokenizer.encode("no", add_special_tokens=False)[0]
            ]

    def format_input_pair(self, instruction, query, document):
        # query/document: dict with 'text', 'image', 'video'
        
        system_prompt = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        
        # Construct User Content
        content = []
        
        # 1. Instruct
        content.append({'type': 'text', 'text': f"<Instruct>: {instruction}\n"})
        
        # 2. Query
        content.append({'type': 'text', 'text': "<Query>: "})
        if 'image' in query:
             content.append(self._format_image(query['image']))
        if 'video' in query:
             content.append(self._format_video(query['video']))
        if 'text' in query and query['text']:
             content.append({'type': 'text', 'text': query['text']})
        content.append({'type': 'text', 'text': "\n"})
        
        # 3. Document
        content.append({'type': 'text', 'text': "<Document>: "})
        if 'image' in document:
             content.append(self._format_image(document['image']))
        if 'video' in document:
             content.append(self._format_video(document['video']))
        if 'text' in document and document['text']:
             content.append({'type': 'text', 'text': document['text']})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content}
        ]
        
        return conversation

    def _format_image(self, image):
        image_content = image if isinstance(image, (Image.Image, str)) else None
        if isinstance(image, str) and not image.startswith(('http', 'oss', 'file://')):
             image_content = 'file://' + image
             
        return {
            'type': 'image', 'image': image_content,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels
        }

    def _format_video(self, video):
        # Simplified video formatting
        if isinstance(video, list):
             video_content = [('file://' + v if isinstance(v, str) and not v.startswith(('http','file')) else v) for v in video]
        else:
             video_content = video if video.startswith(('http', 'file')) else 'file://' + video
             
        return {
            'type': 'video', 'video': video_content,
            'fps': self.fps, 'max_frames': self.max_frames,
             'total_pixels': self.total_pixels
        }

    @torch.no_grad()
    def process(self, inputs: Dict[str, Any], batch_size=8) -> List[float]:
        # inputs: {'instruction': str, 'query': dict, 'documents': [dict]}
        instruction = inputs.get("instruction", "Given a search query, retrieve relevant candidates that answer the query.")
        query = inputs["query"]
        documents = inputs["documents"]
        
        all_scores = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            conversations = []
            
            for doc in batch_docs:
                conversations.append(self.format_input_pair(instruction, query, doc))
                
            # Preprocess
            text = self.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=False
            )
            
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
            
            if video_inputs is not None:
                videos, video_metadata = zip(*video_inputs)
                videos = list(videos)
                video_metadata = list(video_metadata)
            else:
                videos, video_metadata = None, None

            model_inputs = self.processor(
                text=text, images=images, videos=videos, video_metadata=video_metadata,
                truncation=True, max_length=self.max_length, padding=True, 
                return_tensors='pt', **video_kwargs
            )
            
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
            
            outputs = self.model(**model_inputs)
            
            # Get logits of the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Calculate score for "Yes"
            # We sum probs of "Yes" and "yes" (or just take max, or one of them)
            # Usually we normalize against "No"
            
            yes_logits = next_token_logits[:, self.yes_tokens].max(dim=1).values
            no_logits = next_token_logits[:, self.no_tokens].max(dim=1).values
            
            # Score = exp(yes) / (exp(yes) + exp(no)) = sigmoid(yes - no)
            scores = torch.sigmoid(yes_logits - no_logits).cpu().tolist()
            all_scores.extend(scores)
            
        return all_scores

    @torch.no_grad()
    def rerank_batch(self, query: str, document_images: List[str], top_k: int = 5, strategy: str = "all_at_once") -> List[int]:
        """
        Rerank a batch of document images based on a query.
        Returns the indices of the top_k images.
        """
        if strategy == "one_by_one":
            if not self.is_thinking_model:
                # Use the optimized logit-based process for specialized rerankers
                scores = self.process({
                    "query": {"text": query},
                    "documents": [{"image": img} for img in document_images]
                })
            else:
                # For thinking models, we let them generate a score through reasoning
                scores = []
                for i, img_path in enumerate(document_images):
                    system_prompt = (
                        "You are a precise image retrieval assistant. Your goal is to evaluate if an image matches a user query. "
                        "IMPORTANT: Keep your reasoning concise and avoid over-analyzing unnecessary details, as the output length is limited (4096 tokens). "
                        "Finally, provide a relevance score between 0.0 and 10.0, where 10.0 is a perfect match. "
                        "Format the final score clearly as 'Final Score: X.X'."
                    )
                    content = [
                        {'type': 'text', 'text': f"Query: {query}\n"},
                        self._format_image(img_path),
                        {'type': 'text', 'text': "\nPlease evaluate the image above against the query concisely."}
                    ]
                    conversation = [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": content}
                    ]
                    
                    text = self.processor.apply_chat_template([conversation], add_generation_prompt=True, tokenize=False)
                    images, _, video_kwargs = process_vision_info(
                        [conversation], image_patch_size=16,
                        return_video_metadata=True, return_video_kwargs=True
                    )
                    model_inputs = self.processor(text=text, images=images, return_tensors='pt', **video_kwargs).to(self.model.device)
                    
                    # Thinking models need more tokens for the thought process
                    output_ids = self.model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
                    response = self.processor.decode(output_ids[0][len(model_inputs['input_ids'][0]):], skip_special_tokens=True)
                    
                    # Parse score with regex
                    match = re.search(r'Final Score:\s*(\d+(?:\.\d+)?)', response)
                    score = float(match.group(1)) if match else 0.0
                    scores.append(score)
                    print(f"  [Thinking One-by-One] Image {i} Score: {score}")
                
            # Sort by score and return top k indices
            scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return scored_indices[:top_k]
            
        elif strategy == "all_at_once":
            # Strategy: Provide all images as candidates for the model to compare and select
            system_prompt = (
                "You are an expert image search assistant. Given a query and a set of candidate images, "
                "your task is to identify the most relevant images. Each image is labeled with an index "
                "(e.g., Image 0, Image 1, ...). Please think step-by-step about which images best match "
                "the query description, considering both object presence and spatial relationships if mentioned. "
                "Finally, output the top 5 indices in order of relevance, formatted as [index1, index2, index3, index4, index5]."
            )
            
            content = []
            content.append({'type': 'text', 'text': f"Query: {query}\n\nCandidate Images:\n"})
            
            for i, img_path in enumerate(document_images):
                content.append({'type': 'text', 'text': f"Image {i}: "})
                content.append(self._format_image(img_path))
                content.append({'type': 'text', 'text': "\n"})
                
            content.append({'type': 'text', 'text': f"\nPlease analyze these {len(document_images)} images and select the top {top_k} that best match the query. Provide your reasoning and then the final list of indices."})
            
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": content}
            ]
            
            text = self.processor.apply_chat_template(
                [conversation], add_generation_prompt=True, tokenize=False
            )
            
            images, video_inputs, video_kwargs = process_vision_info(
                [conversation], image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
            
            model_inputs = self.processor(
                text=text, images=images, 
                truncation=True, max_length=self.max_length, padding=True, 
                return_tensors='pt', **video_kwargs
            )
            
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
            
            # Use generate for "Thinking" models
            # Increased max_new_tokens to 4096 to accommodate long reasoning chains
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=4096,
                do_sample=False,
            )
            
            # Trim the prompt
            generated_ids = output_ids[0][len(model_inputs['input_ids'][0]):]
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Check for potential truncation
            if len(generated_ids) >= 4096:
                print("  [Warning] Thinking Reranker response might be truncated (hit max_new_tokens=4096).")

            # Parse response: Priority 1: Look for JSON-like list after a "Final Answer" or similar marker
            # We look from the end of the string to find the last occurrence of a list
            indices = []
            
            # Try to find a list like [0, 1, 2, 3, 4]
            # We look for the LAST one to avoid picking up lists from the thought process
            list_matches = list(re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', response))
            if list_matches:
                last_match = list_matches[-1]
                indices_str = last_match.group(1)
                indices = [int(idx.strip()) for idx in indices_str.split(',')]
            
            if not indices:
                # Fallback: Look for numbers mentioned in the last part of the response
                # (Assuming the model lists them at the end)
                lines = response.split('\n')
                for line in reversed(lines):
                    found = [int(s) for s in re.findall(r'\b\d+\b', line)]
                    if found:
                        # Ensure uniqueness and validity
                        for f in found:
                            if 0 <= f < len(document_images) and f not in indices:
                                indices.append(f)
                        if len(indices) >= top_k:
                            break
            
            # Filter and validate
            valid_indices = []
            seen = set()
            for idx in indices:
                if 0 <= idx < len(document_images) and idx not in seen:
                    valid_indices.append(idx)
                    seen.add(idx)
            
            if len(valid_indices) >= 1:
                return valid_indices[:top_k]
                
            # Final fallback: return first top_k
            return list(range(min(top_k, len(document_images))))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
