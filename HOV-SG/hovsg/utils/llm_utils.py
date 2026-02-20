from functools import lru_cache
import os
from typing import Dict, List, Tuple, Union
import json
import time
import random
from omegaconf import DictConfig

# Try importing providers
try:
    import google.generativeai as genai_legacy
except ImportError:
    genai_legacy = None

try:
    from google import genai
    from google.genai import types
    import httpx
except ImportError:
    genai = None
    httpx = None

try:
    from openai import OpenAI
    import httpx
except ImportError:
    OpenAI = None
    httpx = None
    

def get_llm_config(cfg: DictConfig = None):
    """Extract LLM config with defaults"""
    provider = "gemini"
    model = "gemini-2.0-flash"
    api_key = None
    
    if cfg and hasattr(cfg, 'models') and hasattr(cfg.models, 'llm'):
        llm_cfg = cfg.models.llm
        provider = getattr(llm_cfg, 'provider', provider)
        model = getattr(llm_cfg, 'model', model)
        api_key = getattr(llm_cfg, 'api_key', api_key)
        
    return provider, model, api_key

def get_client(cfg: DictConfig = None):
    provider, _, api_key = get_llm_config(cfg)
    
    if provider == "gemini":
        # Prefer new SDK if available and working, else legacy
        if genai:
             # Fallback to hardcoded key if not provided
            if not api_key:
                 api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDzcaxpz7Qm5m6vWZxOZCs4lPHw6OF_S5U")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            return genai.Client(api_key=api_key)
        elif genai_legacy:
            if not api_key:
                 api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDzcaxpz7Qm5m6vWZxOZCs4lPHw6OF_S5U")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai_legacy.configure(api_key=api_key)
            return genai_legacy
        else:
            raise ImportError("Google Generative AI module not found. Please install `google-generativeai`.")
        
    elif provider == "openai" or provider == "gpt":
        if OpenAI is None:
            raise ImportError("OpenAI module not found. Please install `openai`.")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        client_kwargs = {"api_key": api_key}
        if httpx:
            client_kwargs["http_client"] = httpx.Client()
        return OpenAI(**client_kwargs)
        
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def generate_with_retry(client, contents, cfg: DictConfig = None, temperature=0.0):
    """
    Unified generation function for both Gemini and OpenAI with retry logic.
    """
    provider, model_name, _ = get_llm_config(cfg)
    
    # Proactive delay
    time.sleep(2 if provider == "gemini" else 0.5)
    
    max_retries = 3
    base_wait = 20 # seconds
    
    for i in range(max_retries):
        try:
            if provider == "gemini":
                if genai and isinstance(client, genai.Client):
                     return _generate_gemini_new(client, model_name, contents, temperature)
                else:
                     return _generate_gemini_legacy(client, model_name, contents, temperature)
            elif provider == "openai" or provider == "gpt":
                return _generate_openai(client, model_name, contents, temperature)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "Rate limit" in error_str or "Quota" in error_str
            
            if is_rate_limit:
                wait_time = base_wait * (1.5 ** i) + random.uniform(0, 2)
                print(f"Rate limit hit ({provider}). Retrying in {wait_time:.1f}s... (Attempt {i+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    
    # Final attempt
    if provider == "gemini":
        if genai and isinstance(client, genai.Client):
                return _generate_gemini_new(client, model_name, contents, temperature)
        else:
                return _generate_gemini_legacy(client, model_name, contents, temperature)
    else:
        return _generate_openai(client, model_name, contents, temperature)

def _generate_gemini_new(client, model, contents, temperature):
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
        )
    )
    return response.text

def _generate_gemini_legacy(client, model_name, contents, temperature):
    # client is genai_legacy module
    model = client.GenerativeModel(model_name)
    response = model.generate_content(
        contents,
        generation_config=client.types.GenerationConfig(
            temperature=temperature
        )
    )
    return response.text

def _generate_openai(client, model, contents, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": contents}],
        temperature=temperature
    )
    return response.choices[0].message.content

def infer_floor_id_from_query(floor_ids: List[int], query: str, cfg: DictConfig = None) -> int:
    """return the floor id from the floor_ids_list that match with the query"""
    floor_ids_str = ", ".join([str(i) for i in floor_ids])
    
    prompt = f"""
You are a floor detector. You can infer the floor number based on a query.
The query is: {query}.
The floor number list is: {floor_ids_str}.
Please answer the floor number in one integer.
"""
    try:
        client = get_client(cfg)
        result_text = generate_with_retry(client, prompt, cfg=cfg)
        result = result_text.strip()
        try:
            result = int(result)
        except:
            # Try to find integer in the string
            import re
            match = re.search(r'\d+', result)
            if match:
                result = int(match.group())
            else:
                print(f"The return answer is not an integer. The answer is: {result}")
                # Default to first floor if fails
                result = floor_ids[0]
    except Exception as e:
        print(f"Error in infer_floor_id_from_query: {e}")
        return floor_ids[0]
            
    if result not in floor_ids:
         # simple heuristic if LLM fails
         return floor_ids[0]
         
    return result


def infer_room_type_from_object_list_chat(
    object_list: List[str], default_room_type: List[str] = None, cfg: DictConfig = None
) -> str:
    """generate a room type based on a list of objects contained in the room"""
    try:
        client = get_client(cfg)
        
        room_types_str = ""
        if default_room_type is not None:
            room_types_str = ", ".join(default_room_type)
            room_instruction = f"Please pick the most matching room type from the following list: {room_types_str}."
        else:
            room_instruction = "What is the room type? Please just answer the room name."

        objects_str = ", ".join(object_list)
        
        prompt = f"""
You are a room type detector. You can infer a room type based on a list of objects.

Example 1:
User: The list of objects contained in this room are: bed, wardrobe, chair, sofa. What is the room type? Please just answer the room name.
Assistant: bedroom

Example 2:
User: The list of objects contained in this room are: tv, table, chair, sofa. Please pick the most matching room type from the following list: living room, bedroom, bathroom, kitchen. What is the room type? Please just answer the room name.
Assistant: living room

User: The list of objects contained in this room are: {objects_str}. {room_instruction}
Assistant:
"""
        result_text = generate_with_retry(client, prompt, cfg=cfg)
        result = result_text.strip()
        print("The room type is: ", result)
        return result
    except Exception as e:
        print(f"Error in infer_room_type_from_object_list_chat: {e}")
        return "unknown"


def parse_hier_query(params, instruction: str) -> Tuple[str, str, str]:
    """
    Parse long language query into a list of short queries at floor, room, and object level
    """
    try:
        client = get_client(params)
        
        # Determine output format based on params
        specs = set(params.main.long_query.spec)
        
        if specs == {"obj"}:
            print("floor, room, object:", None, None, instruction)
            return (None, None, instruction.strip())
        
        if specs == {"obj", "room", "floor"}:
            prompt = f"Please parse the following sentence into a floor, a room and an object: '{instruction}'. Output format: [floor, room, object]. Example: [floor 2, living room, couch]"
        elif specs == {"obj", "room"}:
            prompt = f"Please parse the following sentence into a room and an object: '{instruction}'. Output format: [room, object]. Example: [living room, couch]"
        elif specs == {"obj", "floor"}:
            prompt = f"Please parse the following sentence into a floor and an object: '{instruction}'. Output format: [floor, object]. Example: [floor 2, couch]"
        else:
            # Default fallback
            prompt = f"Please parse the following: {instruction}. Output format: [floor, room, object]"

        result_text = generate_with_retry(client, prompt, cfg=params)
        
        result = result_text.strip().replace("[", "").replace("]", "")
        parts = [x.strip() for x in result.split(",")]
        
        if specs == {"floor", "room", "obj"}:
            # handle missing parts if model output less than 3
            if len(parts) < 3: parts += [""] * (3 - len(parts))
            print("floor, room, object:", parts)
            return tuple(parts[:3])
        elif specs == {"room", "obj"}:
            if len(parts) < 2: parts += [""] * (2 - len(parts))
            print("floor, room, object:", None, parts[0], parts[1])
            return (None, parts[0], parts[1])
        elif specs == {"floor", "obj"}:
            if len(parts) < 2: parts += [""] * (2 - len(parts))
            print("floor, room, object:", parts[0], None, parts[1])
            return (parts[0], None, parts[1])
            
    except Exception as e:
        print(f"Error in parse_hier_query: {e}")
        # Fallback to simple object query if parsing fails
        print("Falling back to object query only.")
        return (None, None, instruction.strip())
    
    return (None, None, instruction)


def parse_floor_room_object_gpt35(instruction: str, cfg: DictConfig = None) -> Tuple[str, str, str]:
    """
    Parse long language query into a list of short queries at floor, room, and object level
    """
    try:
        client = get_client(cfg)
        
        prompt = f"""
You are a hierarchical concept parser. You need to parse a description of an object into floor, region and object. Return ONLY a comma-separated list enclosed in brackets: [floor,region,object]. If a field is missing, leave it empty.

Examples:
Input: chair in region living room on the 0 floor
Output: [floor 0,living room,chair]

Input: floor in living room on floor 0
Output: [floor 0,living room,floor]

Input: table in kitchen on floor 3
Output: [floor 3,kitchen,table]

Input: cabinet in region bedroom on floor 1
Output: [floor 1,bedroom,cabinet]

Input: bedroom on floor 1
Output: [floor 1,bedroom,]

Input: bed
Output: [,,bed]

Input: bedroom
Output: [,bedroom,]

Input: I want to go to bed, where should I go?
Output: [,bedroom,]

Input: I want to go for something to eat upstairs. I am currently at floor 0, where should I go?
Output: [floor 1,dining,]

Input: {instruction}
Output:
"""
        result_text = generate_with_retry(client, prompt, cfg=cfg)
        
        result_text = result_text.strip()
        # Extract content inside brackets if present
        if "[" in result_text and "]" in result_text:
            result = result_text[result_text.find("[")+1:result_text.find("]")]
        else:
            result = result_text.replace("[", "").replace("]", "")
            
        print("floor, room, object:", result)
        decomposition = [x.strip() for x in result.split(",")]
        
        # Ensure 3 elements
        if len(decomposition) < 3:
            decomposition += [""] * (3 - len(decomposition))
        elif len(decomposition) > 3:
            decomposition = decomposition[:3]
            
        return decomposition
    except Exception as e:
        print(f"Error in parse_floor_room_object_gpt35: {e}")
        return [None, None, instruction]

def main():
    while True:
        try:
            instruction = input("Enter instruction: ")
            result = parse_floor_room_object_gpt35(instruction)
            print(result)
        except EOFError:
            break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
