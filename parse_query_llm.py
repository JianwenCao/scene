import os
import sys
import json

# Add HOV-SG to path
sys.path.append('/home/scene/HOV-SG')
from hovsg.utils.llm_utils import parse_floor_room_object_gpt35

instruction = sys.argv[1]
result = parse_floor_room_object_gpt35(instruction)
print(json.dumps(result))
