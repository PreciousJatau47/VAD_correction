import enum

class VADMask(enum.Enum):
    default = 0
    biological = 1
    insects = 2
    birds = 3
    weather = 4
    external_l3_vad_profile = 5

def GetVADMaskDescription(in_enum):
    if in_enum == VADMask.default:
        return "default"
    elif in_enum == VADMask.biological:
        return "biological"
    elif in_enum == VADMask.insects:
        return "insects"
    elif in_enum == VADMask.birds:
        return "birds"
    elif in_enum == VADMask.weather:
        return "weather"
    elif in_enum == VADMask.external_l3_vad_profile:
        return "l3_vad"